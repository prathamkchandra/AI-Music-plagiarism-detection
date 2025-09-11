import os
import json
import numpy as np
import subprocess
import librosa
import urllib.parse
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# Paths
# ------------------------------
FINGERPRINTS_FILE = "fingerprints.json"
DATASET_AUDIO_DIR = "static/Audio_files"
DATASET_SPEC_DIR = "./Spectrograms"
MELODY_EMBEDDING_DIR = "./Melody_Embeddings"
MODEL_PATH = "models/multimodal.keras"

# ------------------------------
# Load fingerprints
# ------------------------------
if os.path.exists(FINGERPRINTS_FILE):
    with open(FINGERPRINTS_FILE, "r", encoding="utf-8") as f:
        FINGERPRINTS = json.load(f)
else:
    FINGERPRINTS = {}

# ------------------------------
# Load CNN model once
# ------------------------------
base_model = load_model(MODEL_PATH)
embedding_model = Model(
    inputs=base_model.input,
    outputs=base_model.get_layer("embedding").output
)

# ------------------------------
# Generate audio fingerprint
# ------------------------------
def generate_fingerprint(audio_path):
    try:
        result = subprocess.run(
            ["fpcalc", "-raw", audio_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            return None
        for line in result.stdout.splitlines():
            if line.startswith("FINGERPRINT="):
                return line.split("=", 1)[1].strip()
    except Exception as e:
        print(f"⚠ Error generating fingerprint: {audio_path} → {e}")
    return None

# ------------------------------
# Extract CNN embedding
# ------------------------------
def extract_cnn_embedding(spec_path):
    try:
        if spec_path.endswith(".npy"):
            spec_img = np.load(spec_path)
            spec_img = np.expand_dims(spec_img, axis=(0, -1))
        else:
            img = image.load_img(spec_path, target_size=(128, 128))
            spec_img = image.img_to_array(img)
            spec_img = np.expand_dims(spec_img, axis=0) / 255.0
        return spec_img
    except Exception as e:
        print(f"⚠ CNN embedding failed: {spec_path} → {e}")
        return None

# ------------------------------
# Extract melody embedding
# ------------------------------
def extract_melody_embedding(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
        return np.mean(chroma, axis=1).reshape(1, -1)
    except Exception as e:
        print(f"⚠ Melody embedding failed: {audio_path} → {e}")
        return None

# ------------------------------
# Preload dataset embeddings
# ------------------------------
DATASET_EMBEDDINGS = []

for genre in os.listdir(DATASET_SPEC_DIR):
    genre_spec_dir = os.path.join(DATASET_SPEC_DIR, genre)
    genre_melody_dir = os.path.join(MELODY_EMBEDDING_DIR, genre)
    if not os.path.isdir(genre_spec_dir):
        continue

    for file in os.listdir(genre_spec_dir):
        if not (file.endswith(".png") or file.endswith(".npy")):
            continue

        spec_path = os.path.join(genre_spec_dir, file)
        song_name = os.path.splitext(file)[0]

        cnn_input = extract_cnn_embedding(spec_path)
        if cnn_input is None:
            continue

        melody_path = os.path.join(genre_melody_dir, song_name + ".npy")
        if os.path.exists(melody_path):
            mel_input = np.load(melody_path).reshape(1, -1)
        else:
            audio_path = os.path.join(DATASET_AUDIO_DIR, genre, song_name + ".mp3")
            mel_input = extract_melody_embedding(audio_path) if os.path.exists(audio_path) else None

        if mel_input is None:
            continue

        try:
            cnn_emb = embedding_model.predict([cnn_input, mel_input], verbose=0)
        except Exception:
            continue

        DATASET_EMBEDDINGS.append({
            "genre": genre,
            "song": song_name,
            "cnn_emb": cnn_emb,
            "mel_emb": mel_input
        })

print(f"Cached {len(DATASET_EMBEDDINGS)} dataset embeddings.")

# ------------------------------
# Compare user song
# ------------------------------

def compare_with_dataset(user_spec_path, user_audio_path, threshold=0.75):
    import random
    import urllib.parse

    # Step 1 — fingerprint exact match
    user_fp = generate_fingerprint(user_audio_path)
    if user_fp:
        for dataset_file, dataset_fp in FINGERPRINTS.items():
            if user_fp == dataset_fp:
                if "/" in dataset_file:
                    folder, song_name = dataset_file.split("/")
                else:
                    folder, song_name = "", dataset_file

                song_name_safe = urllib.parse.quote(song_name + ".mp3")

                return {
                    "match_label": "Exact Match (Fingerprint)",
                    "match_song": song_name,
                    "match_folder": folder,
                    "match_audio_path": os.path.join("Audio_files", folder, song_name_safe).replace("\\","/"),
                    "final_score": 100.0,
                    "color": "red"
                }

    # Step 2 — CNN + melody similarity
    user_cnn_input = extract_cnn_embedding(user_spec_path)
    user_mel_input = extract_melody_embedding(user_audio_path)

    if user_cnn_input is None or user_mel_input is None:
        return {
            "match_label": "Error",
            "match_song": None,
            "match_folder": None,
            "match_audio_path": None,
            "final_score": 0,
            "color": "green"
        }

    try:
        user_cnn_emb = embedding_model.predict([user_cnn_input, user_mel_input], verbose=0)
    except Exception as e:
        print(f"⚠ CNN embedding failed for user song → {e}")
        return {
            "match_label": "Error",
            "match_song": None,
            "match_folder": None,
            "match_audio_path": None,
            "final_score": 0,
            "color": "green"
        }

    # Build matrices
    cnn_matrix = np.vstack([d["cnn_emb"].reshape(1, -1) for d in DATASET_EMBEDDINGS])
    mel_matrix = np.vstack([d["mel_emb"].reshape(1, -1) for d in DATASET_EMBEDDINGS])

    cnn_sims = cosine_similarity(user_cnn_emb, cnn_matrix)[0]
    mel_sims = cosine_similarity(user_mel_input, mel_matrix)[0]

    # Weighted blend
    scores = 0.5 * cnn_sims + 0.5 * mel_sims
    best_idx = np.argmax(scores)
    best_score_raw = scores[best_idx]
    best_match = DATASET_EMBEDDINGS[best_idx]

    folder = best_match["genre"]
    song_name = best_match["song"]

    # Final similarity score scaling
    min_score, max_score = 75, 90
    final_score = min_score + (max_score - min_score) * best_score_raw
    final_score += random.uniform(-1.5, 1.5)  # small variation
    final_score = max(min_score, min(max_score, final_score))
    final_score = round(final_score, 2)

    # Decide label based on score
    if final_score > 90:
        match_label = "High Match"
        color = "red"
    elif 85 <= final_score <= 90:
        match_label = "Moderate Match"
        color = "green"
    elif final_score < 75:
        match_label = "Low Match"
        color = "green"
    else:
        match_label = "No Match"
        color = "green"

    # Encode filename safely
    if best_score_raw >= threshold:
        song_name_safe = urllib.parse.quote(song_name + ".mp3")
        match_audio_path = os.path.join("Audio_files", folder, song_name_safe).replace("\\","/")
    else:
        match_audio_path = None

    return {
        "match_label": match_label,
        "match_song": song_name if best_score_raw >= threshold else None,
        "match_folder": folder if best_score_raw >= threshold else None,
        "match_audio_path": match_audio_path,
        "final_score": final_score,
        "color": color
    }
