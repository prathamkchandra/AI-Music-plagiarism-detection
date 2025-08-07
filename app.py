from flask import Flask, request, render_template
import os
from werkzeug.utils import secure_filename
from compare import compare_with_dataset
from generate_spectrograms import generate_spectrogram

app = Flask(__name__)

UPLOAD_FOLDER = 'static/user_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    plagiarism_percent = None

    if request.method == 'POST':
        file = request.files['audio_file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # ✅ Generate spectrogram of uploaded audio
            user_spec_path = generate_spectrogram(file_path, save_dir=UPLOAD_FOLDER)

            # ✅ Compare it with stored spectrograms
            prediction, plagiarism_percent = compare_with_dataset(user_spec_path)

    return render_template('index.html', prediction=prediction, plagiarism_percent=plagiarism_percent)

if __name__ == '__main__':
    app.run(debug=True)
