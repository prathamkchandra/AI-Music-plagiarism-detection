import os
import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for server
import matplotlib.pyplot as plt

# ------------------------------
# Paths
# ------------------------------
SPEC_DIR = "./Spectrograms"
os.makedirs(SPEC_DIR, exist_ok=True)

# ------------------------------
# Generate spectrogram function
# ------------------------------
def generate_spectrogram(audio_path, output_dir=SPEC_DIR):
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # Generate mel-spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Prepare output path
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        genre_folder = os.path.basename(os.path.dirname(audio_path))
        out_dir = os.path.join(output_dir, genre_folder)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, base_name + ".png")

        # Plot & save spectrogram (non-GUI)
        plt.figure(figsize=(2.56, 2.56), dpi=50)  # 128x128 px
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.imshow(S_dB, aspect='auto', origin='lower', cmap='magma')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # Important to free memory

        return output_path
    except Exception as e:
        print(f"⚠ Spectrogram generation failed: {audio_path} → {e}")
        return None

# ------------------------------
# Optional: Generate .npy array for CNN
# ------------------------------
def generate_spectrogram_array(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        # Normalize between 0-1 for CNN input
        S_norm = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())
        return S_norm.astype(np.float32)
    except Exception as e:
        print(f"⚠ Spectrogram array failed: {audio_path} → {e}")
        return None
