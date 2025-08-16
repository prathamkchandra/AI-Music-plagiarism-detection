import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Input audio dataset directory
dataset_audio_dir = './model/Audio'
dataset_spectrogram_dir = './Spectrograms'

os.makedirs(dataset_spectrogram_dir, exist_ok=True)

def generate_spectrogram(audio_path, output_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f" Skipping {audio_path} — {e}")
        return False

    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(3, 3))
    librosa.display.specshow(S_dB, sr=sr, cmap='magma')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return True

for genre in os.listdir(dataset_audio_dir):
    genre_path = os.path.join(dataset_audio_dir, genre)
    if not os.path.isdir(genre_path):
        continue

    output_genre_dir = os.path.join(dataset_spectrogram_dir, genre)
    os.makedirs(output_genre_dir, exist_ok=True)

    for file in os.listdir(genre_path):
        ext = file.lower()
        
        # Skip MP4 and unsupported formats
        if not ext.endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
            print(f" Skipping unsupported format: {file}")
            continue

        input_path = os.path.join(genre_path, file)
        output_path = os.path.join(output_genre_dir, os.path.splitext(file)[0] + '.png')

        if generate_spectrogram(input_path, output_path):
            print(f" Generated spectrogram for {file}")
        else:
            print(f" Failed to process {file}")

print(" Spectrogram generation completed.")
