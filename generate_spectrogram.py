import matplotlib
matplotlib.use('Agg')  #  Prevents Tkinter GUI issues in Flask

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import uuid

def generate_spectrogram(audio_path):
    """
    Generate a melody-based spectrogram for the uploaded song.
    Matches the same processing used for the dataset.
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)

        # Isolate harmonic (melody) part
        y_harmonic, _ = librosa.effects.hpss(y)

        # Estimate pitch (melody line)
        f0, voiced_flag, _ = librosa.pyin(
            y_harmonic,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7')
        )

        # Replace NaN with 0 for plotting
        f0 = np.nan_to_num(f0)

        # Unique filename
        filename = f"{uuid.uuid4().hex}.png"
        output_dir = 'static/user_specs'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)

        # Plot melody contour (no GUI backend)
        fig, ax = plt.subplots(figsize=(3, 3))
        times = librosa.times_like(f0)
        ax.plot(times, f0, color='cyan')
        ax.set(yticks=[], xticks=[])  # Hide ticks
        ax.axis('off')

        # Save figure
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        return output_path

    except Exception as e:
        print(f" Failed to generate spectrogram for {audio_path} — {e}")
        return None
