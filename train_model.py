import os
import numpy as np
import random
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Concatenate, LeakyReLU
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ------------------------
# Paths
# ------------------------
spectrogram_dir = './Spectrograms'
melody_dir = './Melody_Embeddings'
model_path = 'models/multimodal.keras'

# ------------------------
# Params
# ------------------------
input_shape = (128, 128, 3)
melody_shape = (12,)
batch_size = 32
epochs = 100
train_ratio = 0.80   # using 85% for training now
test_ratio = 0.20    # 15% for testing
random.seed(42)

# ------------------------
# Load data
# ------------------------
all_samples = []
class_names = sorted([d for d in os.listdir(spectrogram_dir) if os.path.isdir(os.path.join(spectrogram_dir, d))])
class_to_index = {c: i for i, c in enumerate(class_names)}

for cls in class_names:
    spec_class_dir = os.path.join(spectrogram_dir, cls)
    for fname in os.listdir(spec_class_dir):
        if not fname.endswith('.png'):
            continue
        mel_path = os.path.join(melody_dir, cls, fname.replace('.png', '.npy'))
        if not os.path.exists(mel_path):
            continue
        all_samples.append({'spectrogram': os.path.join(spec_class_dir, fname),
                            'melody': mel_path,
                            'label': class_to_index[cls]})

random.shuffle(all_samples)
n_total = len(all_samples)
n_train = int(train_ratio * n_total)
n_test = n_total - n_train

train_samples = all_samples[:n_train]
test_samples = all_samples[n_train:]

print(f"Total: {n_total}, Train: {len(train_samples)}, Test: {len(test_samples)}")

# ------------------------
# Data generator
# ------------------------
def multimodal_generator_tf(samples, batch_size, num_classes, input_shape, melody_shape, shuffle=True):
    while True:
        if shuffle:
            random.shuffle(samples)
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size]
            imgs = np.zeros((len(batch), *input_shape), dtype=np.float32)
            melodies = np.zeros((len(batch), *melody_shape), dtype=np.float32)
            labels = np.zeros((len(batch), num_classes), dtype=np.float32)

            for idx, s in enumerate(batch):
                img = img_to_array(load_img(s['spectrogram'], target_size=input_shape[:2])) / 255.0
                mel = np.load(s['melody'])
                imgs[idx] = img
                melodies[idx] = mel
                labels[idx, s['label']] = 1.0

            yield (tf.convert_to_tensor(imgs), tf.convert_to_tensor(melodies)), tf.convert_to_tensor(labels)

# ------------------------
# Model architecture
# ------------------------
def conv_block(x, filters):
    x = Conv2D(filters, (3,3), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    return x

input_image = Input(shape=input_shape, name='spectrogram_input')
x = conv_block(input_image, 32)
x = conv_block(x, 64)
x = conv_block(x, 128)
x = conv_block(x, 256)

x = Flatten()(x)
cnn_embedding = Dense(512, name="embedding")(x)
cnn_embedding = LeakyReLU(alpha=0.1)(cnn_embedding)
cnn_embedding = Dropout(0.5)(cnn_embedding)

input_melody = Input(shape=melody_shape, name='melody_input')
melody_dense = Dense(128)(input_melody)
melody_dense = LeakyReLU(alpha=0.1)(melody_dense)
melody_dense = Dropout(0.3)(melody_dense)

combined = Concatenate()([cnn_embedding, melody_dense])
combined = Dense(256)(combined)
combined = LeakyReLU(alpha=0.1)(combined)
combined = Dropout(0.5)(combined)
output = Dense(len(class_names), activation='softmax')(combined)

model = Model(inputs=[input_image, input_melody], outputs=output)
model.compile(optimizer=Adam(learning_rate=0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ------------------------
# Callbacks
# ------------------------
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-6)

# ------------------------
# Generators
# ------------------------
train_gen = multimodal_generator_tf(train_samples, batch_size, len(class_names), input_shape, melody_shape)
test_gen = multimodal_generator_tf(test_samples, batch_size, len(class_names), input_shape, melody_shape, shuffle=False)

steps_per_epoch = (len(train_samples) + batch_size - 1) // batch_size
test_steps = (len(test_samples) + batch_size - 1) // batch_size

# ------------------------
# Training
# ------------------------
history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    callbacks=[reduce_lr]
)

# ------------------------
# Save model
# ------------------------
os.makedirs('models', exist_ok=True)
model.save(model_path)
print(f"Model saved at {model_path}")

# ------------------------
# Evaluate on Train & Test
# ------------------------
train_loss, train_acc = model.evaluate(train_gen, steps=steps_per_epoch)
print(f"Training Accuracy: {train_acc*100:.2f}%")

test_loss, test_acc = model.evaluate(test_gen, steps=test_steps)
print(f"Test Accuracy: {test_acc*100:.2f}%")
