# app.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from tensorflow.keras import models
import numpy as np
from PIL import Image

# Load the trained model
model = models.load_model('mnist_model.h5')

# Function to preprocess an image
def preprocess_image(image):
    img = Image.open(image).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img).reshape((1, 28, 28, 1)).astype('float32') / 255.0
    return img_array

# Function to preprocess a drawn image
def preprocess_drawn_image(drawn_image):
    # Convert to grayscale and resize to (28, 28)
    img = Image.fromarray((drawn_image * 255).astype('uint8')).convert('L')
    img = img.resize((28, 28))
    
    # Convert to array and reshape
    img_array = np.array(img).reshape((1, 28, 28, 1)).astype('float32') / 255.0
    return img_array

# Set page configuration
st.set_page_config(
    page_title="Digit Recognizer",
    page_icon="🖍️",
    layout="wide",
)

# Streamlit app
col1, col2 = st.columns([1, 3])

# Set logo in the right column
col2.image("logo.png", width=100)

# Title and introduction in the left column
col1.title("Digit Recognizer")
col1.header("Draw or Upload a Digit Image and Predict!")

# Sidebar configuration
st.sidebar.title("Choose Input Type")

# Choose input type (upload image or draw)
input_type = st.sidebar.radio("Select Input Type", ["Upload Image", "Draw Digit"])

if input_type == "Upload Image":
    # Upload image through the sidebar
    uploaded_file = st.sidebar.file_uploader("Choose a digit image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        col1.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        # Predict button
        if col1.button("Predict"):
            # Preprocess the uploaded image and make predictions
            image_array = preprocess_image(uploaded_file)
            prediction = model.predict(image_array)
            predicted_label = np.argmax(prediction)

            # Display the prediction
            col1.write("Prediction:")
            col1.write(f"Predicted Digit: {predicted_label}")
            col1.write("Prediction Probabilities:")
            col1.bar_chart(prediction.flatten())

elif input_type == "Draw Digit":
    # Create a canvas for drawing
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.1)",  # Background color of the canvas
        stroke_width=10,
        stroke_color="#000",
        update_streamlit=True,
        height=150,
        drawing_mode="freedraw",
    )

    # Predict button for drawing
    if col1.button("Predict"):
        if canvas_result.image_data is not None:
            # Preprocess the drawn image and make predictions
            drawn_image_array = preprocess_drawn_image(canvas_result.image_data)
            prediction = model.predict(drawn_image_array)
            predicted_label = np.argmax(prediction)

            # Display the prediction
            col1.write("Prediction:")
            col1.write(f"Predicted Digit: {predicted_label}")
            col1.write("Prediction Probabilities:")
            col1.bar_chart(prediction.flatten())

# Instructions and model summary
st.sidebar.header("How to Use")
st.sidebar.write(
    "1. Select the input type: 'Upload Image' or 'Draw Digit'.\n"
    "2. Upload a grayscale image of a digit or draw a digit on the canvas.\n"
    "3. Press the 'Predict' button to make predictions.\n"
    "4. View the predicted digit and probabilities."
)

# Streamlit app run command: streamlit run app.py
