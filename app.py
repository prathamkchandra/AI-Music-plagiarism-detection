from flask import Flask, request, render_template, url_for
import os
from generate_spectrogram import generate_spectrogram
from compare import compare_with_dataset, generate_fingerprint
import json

app = Flask(__name__)

# Paths
UPLOAD_FOLDER = os.path.join("static", "uploads")
DATASET_AUDIO_DIR = os.path.join("static", "dataset", "Audio_files")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load precomputed fingerprints
FINGERPRINTS_FILE = "fingerprints.json"
if os.path.exists(FINGERPRINTS_FILE):
    with open(FINGERPRINTS_FILE, "r", encoding="utf-8") as f:
        FINGERPRINTS = json.load(f)
else:
    FINGERPRINTS = {}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")

        if not file:
            return render_template(
                "index.html",
                uploaded_song=None,
                uploaded_song_path=None,
                match_label="No file uploaded",
                match_song=None,
                match_song_path=None,
                final_score=None
            )

        # Save uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        uploaded_song_path = f"uploads/{file.filename}"

        # Step 1 — Try fingerprint match
        try:
            user_fp = generate_fingerprint(file_path)
            if user_fp:
                for dataset_file, dataset_fp in FINGERPRINTS.items():
                    if dataset_fp == user_fp:
                        # Fingerprint match → direct return
                        return render_template(
                            "index.html",
                            uploaded_song=file.filename,
                            uploaded_song_path=uploaded_song_path,
                            match_label="Exact Match (Fingerprint)",
                            match_song=dataset_file,
                            match_song_path=f"dataset/Audio_files/{dataset_file}",
                            final_score=100.0
                        )
        except Exception as e:
            print(f"⚠ Fingerprint check failed: {e}")

        # Step 2 — No hash match, do spectrogram + CNN + melody
        user_spec_path = generate_spectrogram(file_path)
        if not user_spec_path:
            return render_template(
                "index.html",
                uploaded_song=file.filename,
                uploaded_song_path=uploaded_song_path,
                match_label="Error generating spectrogram",
                match_song=None,
                match_song_path=None,
                final_score=None
            )

        try:
            result = compare_with_dataset(
                user_spec_path,
                user_audio_path=file_path,
                dataset_audio_dir=DATASET_AUDIO_DIR,
                dataset_spec_dir="./Spectrograms",
                melody_embedding_dir="./Melody_Embeddings",
                threshold=0.75
            )
        except Exception as e:
            return render_template(
                "index.html",
                uploaded_song=file.filename,
                uploaded_song_path=uploaded_song_path,
                match_label=f"Error during comparison: {e}",
                match_song=None,
                match_song_path=None,
                final_score=None
            )

        # Build dataset song path if found
        match_song_path = None
        if result.get("match_song") and result.get("match_label") != "No match found":
            match_song_path = f"dataset/Audio_files/{result['match_label']}/{result['match_song']}.mp3"

        return render_template(
            "index.html",
            uploaded_song=file.filename,
            uploaded_song_path=uploaded_song_path,
            match_label=result.get("match_label"),
            match_song=result.get("match_song"),
            match_song_path=match_song_path,
            final_score=result.get("final_score")
        )

    return render_template(
        "index.html",
        uploaded_song=None,
        uploaded_song_path=None,
        match_label=None,
        match_song=None,
        match_song_path=None,
        final_score=None
    )

if __name__ == "__main__":
    app.run(debug=True)
