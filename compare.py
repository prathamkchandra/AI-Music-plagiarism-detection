import json
import os
import subprocess
import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing import image
from keras.models import load_model, Model

FINGERPRINTS_FILE = "fingerprints.json"

# Load precomputed fingerprints
if os.path.exists(FINGERPRINTS_FILE):
    with open(FINGERPRINTS_FILE, "r", encoding="utf-8") as f:
        FINGERPRINTS = json.load(f)
else:
    FINGERPRINTS = {}


def generate_fingerprint(audio_path):
    """
    Generate an audio fingerprint for a given file using fpcalc.
    """
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
                return line.split("=", 1)[1]
    except Exception as e:
        print(f" Error generating fingerprint for {audio_path}: {e}")
        return None


def compare_with_dataset(user_spec_path, user_audio_path,
                         dataset_audio_dir="./model/Audio_files",
                         dataset_spec_dir="./Spectrograms",
                         melody_embedding_dir="./Melody_Embeddings",
                         threshold=0.80):

    # Step 1 — Try HASH match first
    user_fingerprint = generate_fingerprint(user_audio_path)
    if user_fingerprint:
        for dataset_file, dataset_fp in FINGERPRINTS.items():
            if dataset_fp == user_fingerprint:  # Exact match
                return {
                    "match_label": "Exact Match (Fingerprint)",
                    "match_song": dataset_file,
                    "final_score": 100.0
                }

    # Step 2 — CNN + Melody similarity
    base_model = load_model("models/latest_model.keras")
    embedding_model = Model(
        inputs=base_model.input,
        outputs=base_model.get_layer("embedding").output
    )

    def extract_cnn_embedding(spec_path):
        img = image.load_img(spec_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        return embedding_model.predict(img_array, verbose=0)

    def extract_melody_embedding(audio_path):
        y, sr = librosa.load(audio_path, sr=None)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
        return np.mean(chroma, axis=1).reshape(1, -1)

    user_cnn_embedding = extract_cnn_embedding(user_spec_path)
    user_melody_embedding = extract_melody_embedding(user_audio_path)

    best_match_label = "No match found"
    best_match_song = None
    best_score = 0

    for genre in os.listdir(dataset_spec_dir):
        genre_spec_dir = os.path.join(dataset_spec_dir, genre)
        genre_melody_dir = os.path.join(melody_embedding_dir, genre)

        if not os.path.isdir(genre_spec_dir):
            continue

        for file in os.listdir(genre_spec_dir):
            if file.endswith(".png"):
                spec_path = os.path.join(genre_spec_dir, file)
                melody_embedding_path = os.path.join(
                    genre_melody_dir, file.replace(".png", ".npy")
                )

                if not os.path.exists(melody_embedding_path):
                    continue

                dataset_cnn_embedding = extract_cnn_embedding(spec_path)
                cnn_sim = cosine_similarity(user_cnn_embedding, dataset_cnn_embedding)[0][0]

                dataset_melody_embedding = np.load(melody_embedding_path).reshape(1, -1)
                melody_sim = cosine_similarity(user_melody_embedding, dataset_melody_embedding)[0][0]

               # Weighted blend
                blended_score = cnn_sim * 0.4 + melody_sim * 0.6

                # Aggressive normalization: treat anything under 0.75 as "very different"
                norm_score = max(0, blended_score - 0.80) / (1.0 - 0.80)

                # Strong exaggeration so small cosine changes = big percentage differences
                exaggerated = norm_score ** 7.0  # You can try 5.0 for even more extreme scaling

                # Map back to plagiarism % range (50–100)
                mapped_score = 0.50 + exaggerated * 0.50

                # Cap at 100%
                mapped_score = min(mapped_score, 1.0)

                if mapped_score > best_score:
                    best_score = mapped_score
                    best_match_label = genre
                    best_match_song = file.replace(".png", "")

    return {
        "match_label": best_match_label if best_score >= threshold else "No match found",
        "match_song": best_match_song if best_score >= threshold else None,
        "final_score": round(best_score * 100, 2)
    }


