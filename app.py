from flask import Flask, request, render_template
import os
from werkzeug.utils import secure_filename
from generate_spectrogram import generate_spectrogram
from compare import compare_with_dataset

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    plagiarism_percent = None
    error = None

    if request.method == 'POST':
        file = request.files.get('audio_file')
        if not file:
            error = "No file uploaded"
        else:
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            try:
                user_spec_path = generate_spectrogram(file_path)
                prediction, plagiarism_percent = compare_with_dataset(user_spec_path)
            except Exception as e:
                error = str(e)

    return render_template('index.html',
                           prediction=prediction,
                           plagiarism_percent=plagiarism_percent,
                           error=error)

if __name__ == '__main__':
    app.run(debug=True)
