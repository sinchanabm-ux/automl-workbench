from flask import Flask, render_template, request
import os
import pandas as pd

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/', methods = ['POST'] )
def upload():
    file = request.files['csv_file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    #converting CSV into DataFrame to obtain columns
    df = pd.read_csv(filepath)
    columns = df.columns.to_list()
    return render_template('select_target.html', columns=columns, filename = file.filename)

@app.route('/preprocess', methods = ['POST'])
def preprocess():
    target_column = request.form['target_column']
    filename = request.form['filename']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)
    return f"Target Column : {target_column} \n Shape : {df.shape}"

if __name__ == '__main__':
    app.run(debug = True)