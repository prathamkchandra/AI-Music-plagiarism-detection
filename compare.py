import json
import os
import subprocess
import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing import image
from keras.models import load_model, Model

FINGERPRINTS_FILE = "fingerprints.json"
DATASET_AUDIO_DIR = "static/dataset/Audio_files"
DATASET_SPEC_DIR = "./Spectrograms"
MELODY_EMBEDDING_DIR = "./Melody_Embeddings"
MODEL_PATH = "models/latest_model.keras"

# ------------------------------
# Load precomputed fingerprints
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

def generate_fingerprint(audio_path):
    """Generate an audio fingerprint using fpcalc."""
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
        print(f"⚠ Error generating fingerprint for {audio_path}: {e}")
    return None

def extract_cnn_embedding(spec_path):
    """Extract CNN embedding for a spectrogram image."""
    try:
        img = image.load_img(spec_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        return embedding_model.predict(img_array, verbose=0)
    except Exception as e:
        print(f"⚠ CNN embedding failed for {spec_path}: {e}")
        return None

def extract_melody_embedding(audio_path):
    """Extract melody embedding using chroma features."""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
        return np.mean(chroma, axis=1).reshape(1, -1)
    except Exception as e:
        print(f"⚠ Melody embedding failed for {audio_path}: {e}")
        return None

# ------------------------------
# Precompute dataset embeddings once
# ------------------------------
DATASET_EMBEDDINGS = []

for genre in os.listdir(DATASET_SPEC_DIR):
    genre_spec_dir = os.path.join(DATASET_SPEC_DIR, genre)
    genre_melody_dir = os.path.join(MELODY_EMBEDDING_DIR, genre)

    if not os.path.isdir(genre_spec_dir):
        continue

    for file in os.listdir(genre_spec_dir):
        if not file.endswith(".png"):
            continue

        spec_path = os.path.join(genre_spec_dir, file)
        melody_embedding_path = os.path.join(
            genre_melody_dir, file.replace(".png", ".npy")
        )

        if not os.path.exists(melody_embedding_path):
            continue

        cnn_emb = extract_cnn_embedding(spec_path)
        if cnn_emb is None:
            continue

        mel_emb = np.load(melody_embedding_path).reshape(1, -1)

        DATASET_EMBEDDINGS.append({
            "genre": genre,
            "song": file.replace(".png", ""),
            "cnn_emb": cnn_emb,
            "mel_emb": mel_emb
        })

print(f"✅ Cached {len(DATASET_EMBEDDINGS)} dataset embeddings.")

# ------------------------------
# Main function
# ------------------------------
def compare_with_dataset(user_spec_path, user_audio_path, threshold=0.85):
    # Step 1 — Fingerprint exact match
    user_fingerprint = generate_fingerprint(user_audio_path)
    if user_fingerprint:
        for dataset_file, dataset_fp in FINGERPRINTS.items():
            if dataset_fp.strip() == user_fingerprint.strip():
                return {
                    "match_label": "Exact Match (Fingerprint)",
                    "match_song": dataset_file,
                    "match_audio_path": os.path.join(DATASET_AUDIO_DIR, dataset_file + ".mp3"),
                    "final_score": 100.0
                }

    # Step 2 — CNN + Melody similarity
    user_cnn_embedding = extract_cnn_embedding(user_spec_path)
    user_melody_embedding = extract_melody_embedding(user_audio_path)

    if user_cnn_embedding is None or user_melody_embedding is None:
        return {
            "match_label": "Error",
            "match_song": None,
            "match_audio_path": None,
            "final_score": 0
        }

    # --- Vectorize dataset embeddings ---
    dataset_cnn_embs = np.vstack([item["cnn_emb"] for item in DATASET_EMBEDDINGS])
    dataset_mel_embs = np.vstack([item["mel_emb"] for item in DATASET_EMBEDDINGS])

    # Cosine similarity in one go
    cnn_sims = cosine_similarity(user_cnn_embedding, dataset_cnn_embs)[0]
    melody_sims = cosine_similarity(user_melody_embedding, dataset_mel_embs)[0]

    # Weighted blend
    blended_scores = cnn_sims * 0.4 + melody_sims * 0.6

    # Pick best match
    best_idx = np.argmax(blended_scores)
    best_score = blended_scores[best_idx]
    best_match = DATASET_EMBEDDINGS[best_idx]

    return {
        "match_label": best_match["genre"] if best_score >= threshold else "No match found",
        "match_song": best_match["song"] if best_score >= threshold else None,
        "match_audio_path": (os.path.join(DATASET_AUDIO_DIR, best_match["song"] + ".mp3")
                             if best_score >= threshold else None),
        "final_score": round(float(best_score) * 100, 2)
    }
