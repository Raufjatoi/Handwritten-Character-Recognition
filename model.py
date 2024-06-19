# model.py
import numpy as np
import tensorflow as tf
from emnist import extract_training_samples, extract_test_samples
import os
import zipfile

def download_and_extract_emnist():
    try:
        (x_train, y_train), (x_test, y_test) = extract_training_samples('letters')
        x_train = np.expand_dims(x_train, -1) / 255.0
        x_test = np.expand_dims(x_test, -1) / 255.0
        y_train = tf.keras.utils.to_categorical(y_train, 27)  # 26 letters + 1 for digit
        y_test = tf.keras.utils.to_categorical(y_test, 27)
        return (x_train, y_train), (x_test, y_test)
    except zipfile.BadZipFile:
        print("Bad zip file, please try downloading again.")
        # Clean up the corrupted file and try again
        cache_path = os.path.join(os.path.expanduser("~"), ".cache", "emnist", "emnist.zip")
        if os.path.exists(cache_path):
            os.remove(cache_path)
        return download_and_extract_emnist()

# Download and preprocess the dataset
(x_train, y_train), (x_test, y_test) = download_and_extract_emnist()

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(27, activation='softmax')  # 26 letters + 1 for digit
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# Save the model
model.save('handwritten_character_recognition_model.h5')
