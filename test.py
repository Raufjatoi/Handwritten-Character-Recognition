import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import os

# Ensure the model file exists and add debugging information
model_path = 'handwritten_character_recognition_model.h5'
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.stop()

# Load the model with error handling
try:
    model = tf.keras.models.load_model(model_path)
    st.success("Model loaded successfully!")
except tf.errors.OpError as e:
    st.error(f"Operational error during model loading: {e}")
    st.stop()
except AttributeError as e:
    st.error(f"Attribute error during model loading: {e}")
    st.stop()
except Exception as e:
    st.error(f"Unexpected error during model loading: {e}")
    st.stop()

st.title('Handwritten Character Recognition (Hello World in DL)')
st.write('Upload an image of a handwritten digit (0-9) or draw a digit.')

def predict(image):
    img = ImageOps.grayscale(image)
    img = img.resize((28, 28))
    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28, 1)
    prediction = model.predict(img)
    return np.argmax(prediction)

# File upload option
uploaded_file = st.file_uploader("Choose an image...", type="png")

# Drawing canvas option
st.write("Or draw a digit below:")
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fill color with some transparency
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=150,
    width=150,
    drawing_mode="freedraw",
    key="canvas",
)

# Placeholder for the prediction result
prediction_placeholder = st.empty()

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    with st.spinner('Classifying uploaded image...'):
        try:
            label = predict(image)
            prediction_placeholder.markdown(f"<div style='text-align: center; font-size: 48px; border: 2px solid black; padding: 20px;'>Predicted digit: {label}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
elif canvas_result.image_data is not None:
    image_data = canvas_result.image_data
    if np.sum(image_data) > 0:  # Check if something was drawn
        st.image(image_data, caption='Drawn Image', use_column_width=True)
        st.write("")
        with st.spinner('Classifying drawn image...'):
            try:
                # Convert the canvas image to a PIL image
                image = Image.fromarray((255 - image_data).astype(np.uint8))  # Invert colors for better visibility
                label = predict(image)
                prediction_placeholder.markdown(f"<div style='text-align: center; font-size: 48px; border: 2px solid black; padding: 20px;'>Predicted digit: {label}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error during prediction: {e}")
else:
    prediction_placeholder.markdown("<div style='text-align: center; font-size: 48px; border: 2px solid black; padding: 20px;'>No prediction yet</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: black;
        text-align: center;
        padding: 10px;
    }
    </style>
    <div class="footer">
        Developed by <a href="https://example.com" target="_blank">Rauf</a>
    </div>
    """, unsafe_allow_html=True)
