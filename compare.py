from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# Load the full model
base_model = load_model(r"C:\Users\Pratham K Chandra\Desktop\Music-plaigarism\cnn_similarity_model.keras")

# Create a new model that outputs from the named embedding layer
embedding_model = Model(inputs=base_model.input, outputs=base_model.get_layer("embedding").output)

def extract_embedding(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    embedding = embedding_model.predict(img_array, verbose=0)
    return embedding[0]

def compare_with_dataset(user_spec_path, dataset_dir="./model/Spectrograms/"):
    user_embedding = extract_embedding(user_spec_path)
    similarities = []

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".png"):
                db_path = os.path.join(root, file)
                db_embedding = extract_embedding(db_path)
                sim = cosine_similarity([user_embedding], [db_embedding])[0][0]
                similarities.append({"filename": file, "similarity": round(sim * 100, 2)})

    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    top_match = similarities[0] if similarities else None
    plagiarism_percent = top_match["similarity"] if top_match else 0.0

    return similarities[:5], plagiarism_percent
