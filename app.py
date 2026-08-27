from flask import Flask, render_template, request
import os

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
    return f"File {file.filename} uploaded successfully!"

if __name__ == '__main__':
    app.run(debug = True)