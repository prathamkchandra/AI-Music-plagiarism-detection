import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing import image
from keras.models import load_model, Model

# Paths
dataset_spec_dir = "./Spectrograms"
melody_embedding_dir = "./Melody_Embeddings"
model_path = "models/latest_model.keras"

# Load CNN model for embedding extraction
base_model = load_model(model_path)
embedding_model = Model(inputs=base_model.input, outputs=base_model.get_layer("embedding").output)

def extract_cnn_embedding(spec_path):
    img = image.load_img(spec_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return embedding_model.predict(img_array, verbose=0)

# Collect all embeddings
cnn_embeddings = []
melody_embeddings = []

for genre in os.listdir(dataset_spec_dir):
    genre_spec_path = os.path.join(dataset_spec_dir, genre)
    genre_melody_path = os.path.join(melody_embedding_dir, genre)

    if not os.path.isdir(genre_spec_path):
        continue

    for file in os.listdir(genre_spec_path):
        if file.endswith(".png"):
            spec_path = os.path.join(genre_spec_path, file)
            melody_path = os.path.join(genre_melody_path, file.replace(".png", ".npy"))

            if not os.path.exists(melody_path):
                continue

            cnn_embeddings.append(extract_cnn_embedding(spec_path))
            melody_embeddings.append(np.load(melody_path).reshape(1, -1))

# Convert to arrays
cnn_embeddings = np.vstack(cnn_embeddings)
melody_embeddings = np.vstack(melody_embeddings)

# Compare random pairs
import random
import matplotlib.pyplot as plt

scores = []
for _ in range(500):  # sample 500 random pairs
    i, j = random.sample(range(len(cnn_embeddings)), 2)
    cnn_sim = cosine_similarity(cnn_embeddings[i].reshape(1, -1), cnn_embeddings[j].reshape(1, -1))[0][0]
    melody_sim = cosine_similarity(melody_embeddings[i].reshape(1, -1), melody_embeddings[j].reshape(1, -1))[0][0]
    blended = cnn_sim * 0.3 + melody_sim * 0.7
    scores.append(blended)

# Plot histogram
plt.hist(scores, bins=20, color='skyblue', edgecolor='black')
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.title("Distribution of Cosine Similarities in Dataset")
plt.show()

print(f"Min: {min(scores):.4f}, Max: {max(scores):.4f}, Avg: {np.mean(scores):.4f}")
