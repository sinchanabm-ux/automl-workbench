from flask import Flask, render_template, request, send_file
import io
import os
import re
import time
import uuid
import pandas as pd
from werkzeug.utils import secure_filename
from preprocessing import preprocess_data, clean_data
from training import train_models, AVAILABLE_MODELS
from database import create_table, log_run

UPLOAD_FOLDER = 'uploads'
MAX_AGE_SECONDS = 1800                 # uploads are kept 30 minutes (download + train)
MAX_TOTAL_BYTES = 100 * 1024 * 1024    # hard cap on total upload storage
STORED_NAME = re.compile(r'^[0-9a-f]{32}__[\w.\-]+$')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # reject uploads over 10 MB
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
create_table()


def cleanup_old_uploads():
    """Remove expired uploads, then evict oldest files if storage is over the cap."""
    now = time.time()
    kept = []
    for name in os.listdir(UPLOAD_FOLDER):
        path = os.path.join(UPLOAD_FOLDER, name)
        try:
            if not os.path.isfile(path) or name == '.gitkeep':
                continue
            if now - os.path.getmtime(path) > MAX_AGE_SECONDS:
                os.remove(path)
            else:
                kept.append((os.path.getmtime(path), os.path.getsize(path), path))
        except OSError:
            pass
    total = sum(size for _, size, _ in kept)
    for _, size, path in sorted(kept):          # oldest first
        if total <= MAX_TOTAL_BYTES:
            break
        try:
            os.remove(path)
            total -= size
        except OSError:
            pass


def resolve_upload(stored_name):
    """Validate a stored filename. Returns (filepath, display_name) or (None, None)."""
    stored_name = os.path.basename(stored_name or '')   # blocks ../ paths
    if not STORED_NAME.match(stored_name):
        return None, None
    filepath = os.path.join(UPLOAD_FOLDER, stored_name)
    if not os.path.isfile(filepath):
        return None, None
    return filepath, stored_name.split('__', 1)[1]


# home page
@app.route('/')
def index():
    return render_template('index.html')


# Machine Learning page: lists the available models + dataset upload
@app.route('/machine-learning')
def machine_learning():
    return render_template('machine_learning.html', models=AVAILABLE_MODELS)


@app.route('/uploads/', methods=['POST'])
def upload():
    file = request.files.get('csv_file')
    if file is None or file.filename == '':
        return "Please select a file before uploading."
    if not file.filename.lower().endswith('.csv'):
        return "Please upload a .csv file."

    cleanup_old_uploads()

    # Unique stored name avoids collisions and path-traversal tricks
    original = secure_filename(file.filename) or 'dataset.csv'
    stored_name = f"{uuid.uuid4().hex}__{original}"
    filepath = os.path.join(UPLOAD_FOLDER, stored_name)
    file.save(filepath)

    try:
        df = pd.read_csv(filepath)
    except Exception:
        os.remove(filepath)
        return "Could not read that file as a CSV."

    try:
        _, report = clean_data(df)      # only the summary is needed on this page
    except ValueError as e:
        os.remove(filepath)
        return f"Could not clean this dataset: {e}"

    columns = df.columns.to_list()      # targets are chosen from the original columns
    return render_template('select_target.html',
                           columns=columns,
                           filename=stored_name,     # used by the form and download link
                           display_name=original,    # nice name to show the user
                           report=report)


@app.route('/download/<stored_name>')
def download_cleaned(stored_name):
    filepath, display_name = resolve_upload(stored_name)
    if filepath is None:
        return "This file has expired. Please upload it again.", 404

    try:
        cleaned, _ = clean_data(pd.read_csv(filepath))
    except ValueError as e:
        return f"Could not clean this dataset: {e}", 400

    # Built in memory and streamed to the user: nothing extra is written to disk.
    # utf-8-sig makes Excel open non-English characters correctly.
    buffer = io.BytesIO(cleaned.to_csv(index=False).encode('utf-8-sig'))
    base_name = os.path.splitext(display_name)[0]
    return send_file(buffer, mimetype='text/csv', as_attachment=True,
                     download_name=f"cleaned_{base_name}.csv")


@app.route('/preprocess', methods=['POST'])
def preprocess():
    target_column = request.form['target_column']
    filepath, display_name = resolve_upload(request.form['filename'])
    if filepath is None:
        return "Your uploaded file has expired. Please upload it again."

    try:
        df = pd.read_csv(filepath)
        X, y, problem_type, error_message = preprocess_data(df, target_column)
        if error_message:
            return error_message
        del df

        results = train_models(X, y, problem_type)
    except ValueError as e:
        return f"Could not train models on this dataset: {e}"
    # The upload is NOT deleted here: the user may still want to download the
    # cleaned CSV or try another target. cleanup_old_uploads() removes it later.

    best_model = max(results, key=results.get)

    for model_name, score in results.items():
        log_run(display_name, target_column, problem_type, model_name, score)

    return render_template('results.html', results=results, best_model=best_model,
                           filename=display_name,                 # readable name (as before)
                           stored_name=request.form['filename'],  # for the download link
                           target_column=target_column, problem_type=problem_type)


if __name__ == '__main__':
    app.run(debug=True)