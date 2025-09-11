from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
import os, json
from generate_spectrogram import generate_spectrogram
from compare import compare_with_dataset, generate_fingerprint

# ------------------------------
# Flask app setup
# ------------------------------
app = Flask(__name__)

# ------------------------------
# Paths
# ------------------------------
UPLOAD_FOLDER = os.path.join("static", "uploads")
DATASET_AUDIO_DIR = os.path.join("static", "Audio_files")
FINGERPRINTS_FILE = "fingerprints.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------------------
# Preload fingerprints
# ------------------------------
if os.path.exists(FINGERPRINTS_FILE):
    with open(FINGERPRINTS_FILE, "r", encoding="utf-8") as f:
        FINGERPRINTS = json.load(f)
    print(f" Loaded {len(FINGERPRINTS)} fingerprints.")
else:
    FINGERPRINTS = {}
    print("⚠ No fingerprints.json found. Fingerprint matching disabled.")


# ------------------------------
# Helper: Safe Audio URL Builder
# ------------------------------
def build_audio_url(folder, song_name):
    """Ensure .mp3 only once and file exists"""
    print("Whats in folder", folder,"---" ,song_name)
    if not song_name.lower().endswith(".mp3"):
        song_name = song_name + ".mp3"

    abs_path = os.path.join("static", "Audio_files", folder, song_name)
    if not os.path.exists(abs_path):
        print("⚠ File missing in dataset:", abs_path)
        return None

    return url_for("static", filename=f"Audio_files/{folder}/{song_name}")


# ------------------------------
# Home / Upload route
# ------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return render_template("index.html",
                uploaded_song=None, uploaded_song_path=None,
                match_label="No file uploaded", match_song=None,
                match_song_path=None, final_score=None
            )

        # Save uploaded file safely
        safe_filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, safe_filename)
        file.save(file_path)
        print("Saved file at:", file_path)

        # URL for uploaded file
        uploaded_song_path = url_for('static', filename=f"uploads/{safe_filename}")

        # ------------------------------
        # Step 1 — Fingerprint exact match
        # ------------------------------
        try:
            user_fp = generate_fingerprint(file_path)
            if user_fp:
                for dataset_file, dataset_fp in FINGERPRINTS.items():
                    if dataset_fp == user_fp:
                        folder, song_name = os.path.split(dataset_file)
                        match_song_path = build_audio_url(folder, song_name)

                        return render_template("index.html",
                            uploaded_song=safe_filename, uploaded_song_path=uploaded_song_path,
                            match_label="Exact Match (Fingerprint)",
                            match_song=song_name, match_song_path=match_song_path,
                            final_score=100.0
                        )
        except Exception as e:
            print(f"⚠ Fingerprint check failed: {e}")

        # ------------------------------
        # Step 2 — Spectrogram + CNN + Melody
        # ------------------------------
        user_spec_path = generate_spectrogram(file_path)
        if not user_spec_path:
            return render_template("index.html",
                uploaded_song=safe_filename, uploaded_song_path=uploaded_song_path,
                match_label="Error generating spectrogram",
                match_song=None, match_song_path=None, final_score=None
            )

        try:
            result = compare_with_dataset(user_spec_path, user_audio_path=file_path, threshold=0.70)
        except Exception as e:
            return render_template("index.html",
                uploaded_song=safe_filename, uploaded_song_path=uploaded_song_path,
                match_label=f"Error during comparison: {e}",
                match_song=None, match_song_path=None, final_score=None
            )

        # ------------------------------
        # Build dataset song path if found
        # ------------------------------
        match_song_path = None
        if result.get("match_song") and result.get("match_folder"):
            match_song_path = build_audio_url(result["match_folder"], result["match_song"])

        return render_template("index.html",
            uploaded_song=safe_filename, uploaded_song_path=uploaded_song_path,
            match_label=result.get("match_label"), match_song=result.get("match_song"),
            match_song_path=match_song_path, final_score=result.get("final_score")
        )

    # ------------------------------
    # GET request → render empty form
    # ------------------------------
    return render_template("index.html",
        uploaded_song=None, uploaded_song_path=None, match_label=None,
        match_song=None, match_song_path=None, final_score=None
    )

# ------------------------------
# Start server
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)
