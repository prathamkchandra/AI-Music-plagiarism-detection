def compare_with_dataset(user_spec_path, dataset_dir="generated_spectrograms"):
    from keras.models import load_model, Model
    import numpy as np
    from tensorflow.keras.preprocessing import image
    import os
    from sklearn.metrics.pairwise import cosine_similarity

    base_model = load_model("models/latest_model.keras")
    embedding_model = Model(inputs=base_model.input,
                            outputs=base_model.get_layer("embedding").output)

    def extract_embedding(spec_path):
        img = image.load_img(spec_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        return embedding_model.predict(img_array, verbose=0)

    user_embedding = extract_embedding(user_spec_path)

    similarities = []
    labels = []

    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.png'):
                path = os.path.join(root, file)
                label = os.path.basename(root)
                dataset_embedding = extract_embedding(path)
                sim = cosine_similarity(user_embedding, dataset_embedding)[0][0]
                similarities.append(sim)
                labels.append(label)

    if similarities:
        best_match_index = np.argmax(similarities)
        return labels[best_match_index], round(similarities[best_match_index] * 100, 2)
    else:
        return "No match found", 0.0
