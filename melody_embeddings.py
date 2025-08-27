import librosa
import numpy as np

def extract_melody_embeddings(audio_path, n_mfcc=20):
    """
    Extracts melody-based embeddings from audio using MFCC features.
    Returns a fixed-length vector for similarity comparison.
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)

        # Extract MFCCs (Mel-Frequency Cepstral Coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        # Mean across time axis → fixed-size vector
        mfcc_mean = np.mean(mfcc, axis=1)

        return mfcc_mean
    except Exception as e:
        print(f"Error extracting melody embeddings from {audio_path}: {e}")
        return np.zeros(n_mfcc)
