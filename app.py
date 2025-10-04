from flask import Flask, request, jsonify, url_for
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os, json
from generate_spectrogram import generate_spectrogram
from compare import compare_with_dataset, generate_fingerprint

# ------------------------------
# Flask app setup
# ------------------------------
app = Flask(__name__)
CORS(app)  # allow frontend (React) to connect

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
    print(f"✅ Loaded {len(FINGERPRINTS)} fingerprints.")
else:
    FINGERPRINTS = {}
    print("⚠ No fingerprints.json found. Fingerprint matching disabled.")

# ------------------------------
# Helper: Build dataset song URL
# ------------------------------
def build_audio_url(folder, song_name):
    """Ensure .mp3 extension and check file exists"""
    if not song_name.lower().endswith(".mp3"):
        song_name = song_name + ".mp3"

    abs_path = os.path.join(DATASET_AUDIO_DIR, folder, song_name)
    if not os.path.exists(abs_path):
        print("⚠ Missing dataset file:", abs_path)
        return None

    return url_for("static", filename=f"Audio_files/{folder}/{song_name}", _external=True)

# ------------------------------
# API: Upload + Analyze
# ------------------------------
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    # Save uploaded file
    safe_filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, safe_filename)
    file.save(file_path)

    uploaded_song_url = url_for("static", filename=f"uploads/{safe_filename}", _external=True)

    # Step 1 — Fingerprint exact match
    try:
        user_fp = generate_fingerprint(file_path)
        if user_fp:
            for dataset_file, dataset_fp in FINGERPRINTS.items():
                if dataset_fp == user_fp:
                    folder, song_name = os.path.split(dataset_file)
                    match_song_url = build_audio_url(folder, song_name)
                    return jsonify({
                        "uploaded_song": safe_filename,
                        "uploaded_song_path": uploaded_song_url,
                        "match_label": "Exact Match (Fingerprint)",
                        "match_song": song_name,
                        "match_song_path": match_song_url,
                        "final_score": 100.0
                    })
    except Exception as e:
        print("⚠ Fingerprint check failed:", e)

    # Step 2 — Spectrogram + CNN
    user_spec_path = generate_spectrogram(file_path)
    if not user_spec_path:
        return jsonify({
            "uploaded_song": safe_filename,
            "uploaded_song_path": uploaded_song_url,
            "match_label": "Error generating spectrogram",
            "match_song": None,
            "match_song_path": None,
            "final_score": None
        })

    try:
        result = compare_with_dataset(user_spec_path, user_audio_path=file_path, threshold=0.70)
    except Exception as e:
        return jsonify({
            "uploaded_song": safe_filename,
            "uploaded_song_path": uploaded_song_url,
            "match_label": f"Error during comparison: {e}",
            "match_song": None,
            "match_song_path": None,
            "final_score": None
        })

    # Build dataset match URL if exists
    match_song_url = None
    if result.get("match_song") and result.get("match_folder"):
        match_song_url = build_audio_url(result["match_folder"], result["match_song"])

    return jsonify({
        "uploaded_song": safe_filename,
        "uploaded_song_path": uploaded_song_url,
        "match_label": result.get("match_label"),
        "match_song": result.get("match_song"),
        "match_song_path": match_song_url,
        "final_score": result.get("final_score")
    })

# ------------------------------
# Run server
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
    