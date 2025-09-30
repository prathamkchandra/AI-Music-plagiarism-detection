# AI-based Music Plagiarism Detection System 🎶

A sophisticated, web-based platform designed to detect potential plagiarism in musical compositions. This system uses a **hybrid analysis model**, providing a final weighted score based on similarity from both deep learning embeddings and manually extracted melodic features.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-black?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com/)
[![Keras](https://img.shields.io/badge/Keras-2.x-red?style=for-the-badge&logo=keras)](https://keras.io/)

## Features

-   **Web Interface:** Simple and intuitive interface for uploading audio files.
-   **Hybrid Analysis:** Employs a dual-pathway system for robust analysis:
    -   **Deep Learning:** A CNN trained on spectrograms generates dense melody embeddings.
    -   **Algorithmic Extraction:** Manually defined algorithms extract key melodic and rhythmic features.
-   **Weighted Scoring System:** Combines the results from both analysis methods to produce a more accurate and nuanced final similarity score.
-   **Cosine Similarity:** Uses cosine similarity to compare the vector representations from the CNN.

---

## How It Works

The plagiarism detection process follows a multi-stage approach to ensure comprehensive analysis:

1.  **Upload & Preprocessing:** A user uploads one audio file. The backend converts each file into a spectrogram, a visual representation of the audio's frequency spectrum over time. 

2.  **Dual Analysis Pathway:** The system processes the spectrograms through two parallel pipelines to extract musical features:
    * **A) CNN-based Embedding Generation:** The spectrograms are fed into a pre-trained Convolutional Neural Network (CNN). The CNN generates dense **melody embeddings**—high-dimensional vectors that capture abstract, learned patterns in the music.
    * **B) Manual Feature Extraction:** Simultaneously, separate algorithms directly analyze the audio signals to extract a set of predefined melodic features (e.g., pitch contours, note sequences, rhythmic patterns).

3.  **Similarity Calculation:** A similarity score is calculated independently for each pathway:
    * $S_{\text{cnn}}$: The **cosine similarity** is computed between the two CNN-generated melody embeddings.
    * $S_{\text{manual}}$: A similarity score is computed for the manually extracted feature sets.

4.  **Weighted Score Combination:** The final plagiarism score is a weighted sum of the similarities from both pathways. This allows for fine-tuning the system's sensitivi  ty   to different aspects of musical composition. 

5.  **Result Display:** The final weighted score is converted to a percentage and displayed to the user.

---

## Tech Stack

This project is built with the following technologies:

-   **Backend:** Python, Flask
-   **Machine Learning:** TensorFlow, Keras
-   **Frontend:** HTML, CSS, JavaScript
-   **Audio Processing:** Libraries like `Librosa` or `SciPy`

---

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

-   Python 3.9+
-   `pip` (Python package installer)

### Installation

  **Clone the repository for Windows System:**
    ```
    git clone https://github.com/prathamkchandra/AI-Music-plagiarism-detection.git
    cd AI-Music-plagiarism-detection
    ```

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
