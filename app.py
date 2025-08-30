from flask import Flask, request, render_template
import os, json, urllib.parse
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
FINGERPRINTS = {}
if os.path.exists(FINGERPRINTS_FILE):
    try:
        with open(FINGERPRINTS_FILE, "r", encoding="utf-8") as f:
            FINGERPRINTS = json.load(f)
        print(f"✅ Loaded {len(FINGERPRINTS)} fingerprints.")
    except Exception as e:
        print(f"⚠ Error reading fingerprints.json: {e}")
else:
    print("⚠ No fingerprints.json found. Fingerprint matching disabled.")

# ------------------------------
# Home / Upload route
# ------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return render_template("index.html",
                uploaded_song=None,
                uploaded_song_path=None,
                match_label="No file uploaded",
                match_song=None,
                match_song_path=None,
                final_score=None
            )

        # ✅ Save uploaded file
        safe_filename = urllib.parse.quote(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        uploaded_song_path = f"uploads/{safe_filename}"

        # ------------------------------
        # Step 1 — Fingerprint exact match
        # ------------------------------
        try:
            user_fp = generate_fingerprint(file_path)
            if user_fp:
                for dataset_file, dataset_fp in FINGERPRINTS.items():
                    if dataset_fp == user_fp:
                        folder, song_name = (dataset_file.split("/") if "/" in dataset_file else ("", dataset_file))
                        folder_safe = urllib.parse.quote(folder)
                        song_safe = urllib.parse.quote(song_name + ".mp3")
                        match_song_path = f"Audio_files/{folder_safe}/{song_safe}".replace("\\", "/")

                        return render_template("index.html",
                            uploaded_song=file.filename,
                            uploaded_song_path=uploaded_song_path,
                            match_label="Exact Match (Fingerprint)",
                            match_song=song_name,
                            match_song_path=match_song_path,
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
                uploaded_song=file.filename,
                uploaded_song_path=uploaded_song_path,
                match_label="Error generating spectrogram",
                match_song=None,
                match_song_path=None,
                final_score=None
            )

        try:
            result = compare_with_dataset(user_spec_path, user_audio_path=file_path, threshold=0.70)
        except Exception as e:
            return render_template("index.html",
                uploaded_song=file.filename,
                uploaded_song_path=uploaded_song_path,
                match_label=f"Error during comparison: {e}",
                match_song=None,
                match_song_path=None,
                final_score=None
            )

        # ------------------------------
        # Build dataset song path if found
        # ------------------------------
        match_song_path = None
        if result.get("match_song") and result.get("match_folder"):
            folder_safe = urllib.parse.quote(result.get("match_folder"))
            song_safe = urllib.parse.quote(result.get("match_song") + ".mp3")
            match_song_path = f"Audio_files/{folder_safe}/{song_safe}".replace("\\", "/")

        return render_template("index.html",
            uploaded_song=file.filename,
            uploaded_song_path=uploaded_song_path,
            match_label=result.get("match_label"),
            match_song=result.get("match_song"),
            match_song_path=match_song_path,
            final_score=result.get("final_score")
        )

    # ------------------------------
    # GET request → render empty form
    # ------------------------------
    return render_template("index.html",
        uploaded_song=None,
        uploaded_song_path=None,
        match_label=None,
        match_song=None,
        match_song_path=None,
        final_score=None
    )

# ------------------------------
# Start server
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)
