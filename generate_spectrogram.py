import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import uuid

def generate_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Unique filename
    filename = f"{uuid.uuid4().hex}.png"
    output_dir = 'static/user_specs'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    # Plot and save
    plt.figure(figsize=(3, 3))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return output_path
