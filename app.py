from flask import Flask, render_template, request
import os
import time
import uuid
import pandas as pd
from werkzeug.utils import secure_filename
from preprocessing import preprocess_data
from training import train_models
from database import create_table, log_run

UPLOAD_FOLDER = 'uploads'
MAX_AGE_SECONDS = 3600  # abandoned uploads are removed after 1 hour

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # reject uploads over 10 MB
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
create_table()


def cleanup_old_uploads():
    """Delete uploads that were never finished (user left before training)."""
    now = time.time()
    for name in os.listdir(UPLOAD_FOLDER):
        path = os.path.join(UPLOAD_FOLDER, name)
        try:
            if (os.path.isfile(path) and name != '.gitkeep'
                    and now - os.path.getmtime(path) > MAX_AGE_SECONDS):
                os.remove(path)
        except OSError:
            pass


# home page
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/uploads/', methods=['POST'])
def upload():
    file = request.files.get('csv_file')
    if file is None or file.filename == '':
        return "Please select a file before uploading."
    if not file.filename.lower().endswith('.csv'):
        return "Please upload a .csv file."

    cleanup_old_uploads()

    # Unique stored name avoids collisions and path-traversal tricks
    stored_name = f"{uuid.uuid4().hex}__{secure_filename(file.filename)}"
    filepath = os.path.join(UPLOAD_FOLDER, stored_name)
    file.save(filepath)

    try:
        # nrows=0 reads only the header, so it uses almost no memory
        columns = pd.read_csv(filepath, nrows=0).columns.to_list()
    except Exception:
        os.remove(filepath)
        return "Could not read that file as a CSV."

    # The template's hidden "filename" field now carries the stored name
    return render_template('select_target.html', columns=columns, filename=stored_name)


@app.route('/preprocess', methods=['POST'])
def preprocess():
    target_column = request.form['target_column']
    stored_name = os.path.basename(request.form['filename'])  # blocks ../ paths
    display_name = stored_name.split('__', 1)[-1]
    filepath = os.path.join(UPLOAD_FOLDER, stored_name)

    if not os.path.exists(filepath):
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
    finally:
        # Always free the disk space, whether training succeeded or failed
        if os.path.exists(filepath):
            os.remove(filepath)

    best_model = max(results, key=results.get)

    for model_name, score in results.items():
        log_run(display_name, target_column, problem_type, model_name, score)

    return render_template('results.html', results=results, best_model=best_model,
                           filename=display_name, target_column=target_column,
                           problem_type=problem_type)


if __name__ == '__main__':
    app.run(debug=True)