import os
import librosa
import numpy as np
import hashlib

# Paths
dataset_audio_dir = './static/dataset/Audio_files'
hash_output_file = './audio_hashes.txt'

# Function to create a hash from chroma features
def hash_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)  # Load audio
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)     # Extract chroma
        chroma_mean = np.mean(chroma, axis=1)                # Average over time
        return hashlib.sha256(chroma_mean.round(3).tobytes()).hexdigest()
    except Exception as e:
        print(f"❌ Error hashing {file_path}: {e}")
        return None

# Generate hashes for dataset
with open(hash_output_file, 'w', encoding='utf-8') as f:  # Force UTF-8 encoding
    for root, _, files in os.walk(dataset_audio_dir):
        for file in files:
            if file.lower().endswith((".mp3", ".wav", ".flac", ".ogg", ".m4a")):
                file_path = os.path.join(root, file)
                audio_hash = hash_audio(file_path)
                if audio_hash:
                    f.write(f"{file_path}|{audio_hash}\n")
                    print(f" Hashed {file}")
                else:
                    print(f" Skipped {file}")

print(f"\n🎯 Audio hashing completed. Hashes saved to {hash_output_file}")
