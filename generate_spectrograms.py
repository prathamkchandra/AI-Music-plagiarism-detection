import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def generate_spectrogram(audio_path, save_path):
    try:
        # Load the audio file
        y, sr = librosa.load(audio_path, sr=None, mono=True)

        # Create Mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Plot and save
        plt.figure(figsize=(2.56, 2.56), dpi=50)  # 128x128 pixels
        librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None, cmap='magma')
        plt.axis('off')  # Remove axis

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        return save_path

    except Exception as e:
        print(f"[ERROR] Spectrogram generation failed: {e}")
        return None
