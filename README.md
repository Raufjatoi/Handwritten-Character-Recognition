# Handwritten Character Recognition

This project is a simple handwritten character recognition system built using TensorFlow and Keras. The model is trained on the MNIST dataset and deployed using Streamlit.

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/handwritten-character-recognition.git
    cd handwritten-character-recognition
    ```
2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
2. Open your web browser and go to the local URL provided by Streamlit (usually `http://localhost:8501`).

3. Upload an image of a handwritten digit or draw a digit on the canvas, and the app will classify the digit.

## Features

- Upload an image of a handwritten digit.
- Draw a digit on a canvas.
- Real-time digit classification.
- User-friendly interface with a loading indicator during processing.

## Model

The TensorFlow model used in this app should be trained on handwritten digit datasets such as MNIST and saved as `handwritten_character_recognition_model.h5`.

## Acknowledgments

- Made by [Rauf](https://example.com)

## License

This project is licensed under the MIT License.
