import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import uuid

def generate_spectrogram(audio_path, save_dir="model/user_specs"):
    """Generates and saves a spectrogram from the uploaded audio"""
    y, sr = librosa.load(audio_path, sr=22050)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)

    os.makedirs(save_dir, exist_ok=True)

    filename = f"{uuid.uuid4().hex}.png"
    spec_path = os.path.join(save_dir, filename)

    plt.figure(figsize=(2, 2))
    librosa.display.specshow(S_DB, sr=sr, cmap='magma')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(spec_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return spec_path
