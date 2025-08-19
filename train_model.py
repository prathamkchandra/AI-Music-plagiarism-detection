# train_model.py
import os
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

# Set paths
data_dir = './Spectrograms'
model_path = 'models/latest_model.keras'

# Filter only valid class folders (directories only)
class_folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
num_classes = len(class_folders)

# Params 
input_shape = (128, 128, 3)
batch_size = 32
epochs = 100

# Image generators
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data = datagen.flow_from_directory(
    data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)
val_data = datagen.flow_from_directory(
    data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# 🔍 Debugging: Show detected classes
print("Detected Classes:", train_data.class_indices)

# Model definition using Functional API
input_tensor = Input(shape=input_shape, name="input_layer")
x = Conv2D(32, (3, 3), activation='relu')(input_tensor)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
embedding = Dense(128, activation='relu', name='embedding')(x)
output = Dense(num_classes, activation='softmax')(embedding)

model = Model(inputs=input_tensor, outputs=output)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Training
model.fit(train_data, validation_data=val_data, epochs=epochs)

# Save the model
os.makedirs('models', exist_ok=True)
model.save(model_path)
print(" Model saved at", model_path)
