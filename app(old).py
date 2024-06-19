# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Load the model
model = tf.keras.models.load_model('handwritten_character_recognition_model.h5')

st.title('Handwritten Character Recognition')
st.write('Upload an image of a handwritten digit (0-9).')

uploaded_file = st.file_uploader("Choose an image...", type="png")

def predict(image):
    img = ImageOps.grayscale(image)
    img = img.resize((28, 28))
    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28, 1)
    prediction = model.predict(img)
    return np.argmax(prediction)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = predict(image)
    st.write(f'Predicted digit: {label}')
