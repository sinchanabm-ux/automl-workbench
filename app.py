from flask import Flask, render_template, request
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

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

    #separate target column
    X = df.drop(columns=[target_column])
    y = df[target_column]

    #separate numerical and categorical columns into two lists
    num = X.select_dtypes(include='number').columns.to_list()
    obj = X.select_dtypes(include= ['object', 'str']).columns.to_list()

    #checking for ID columns which are to be dropped as encoding is not needed for columns with completely unique values
    ID = []
    for col in obj:
        ratio = X[col].nunique()/len(X)
        if ratio > 0.9:
            ID.append(col)
    X = X.drop(columns = ID)
    for col in ID:
        obj.remove(col)

    #Step-1 of preprocessing : Handle missing values, median for numerical data, mode for categorical data
    for col in num:
        median = X[col].median()
        X[col] = X[col].fillna(median)
    for col in obj:
        mode = X[col].mode()[0]
        X[col] = X[col].fillna(mode)

    #Step-2 of preprocessing : Encoding categorical data 
    X = pd.get_dummies(X, columns= obj)

    #Step-3 of preprocessing : Scaling numerical columns
    if num: 
        scaler = StandardScaler()
        X[num] = scaler.fit_transform(X[num])

    return X.head().to_html()

if __name__ == '__main__':
    app.run(debug = True)