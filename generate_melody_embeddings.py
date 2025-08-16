import librosa
import numpy as np
import os

def extract_melody_embedding(audio_path):
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
    return np.mean(chroma, axis=1)

dataset_audio_dir = "./model/Audio_files/Movie_songs/Navarasm"
output_dir = "./Melody_Embeddings"
os.makedirs(output_dir, exist_ok=True)

for genre in os.listdir(dataset_audio_dir):
    genre_path = os.path.join(dataset_audio_dir, genre)
    if not os.path.isdir(genre_path):
        continue

    output_genre_dir = os.path.join(output_dir, genre)
    os.makedirs(output_genre_dir, exist_ok=True)

    for file in os.listdir(genre_path):
        # ✅ Skip MP4 files completely
        if file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):  
            audio_path = os.path.join(genre_path, file)
            try:
                embedding = extract_melody_embedding(audio_path)
                np.save(os.path.join(output_genre_dir, file.rsplit('.', 1)[0] + ".npy"), embedding)
                print(f" Generated embedding for {file}")
            except Exception as e:
                print(f" Skipping {file} — {e}")
        else:
            print(f"⚠ Skipping unsupported format: {file}")

print("🎯 Melody embedding generation completed.")
