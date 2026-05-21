import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Potato Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>

.main {
    background-color: #f4fff4;
}

h1 {
    color: #2e7d32;
    text-align: center;
    font-size: 45px;
}

.stButton>button {
    background-color: #2e7d32;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}

.stFileUploader {
    border: 2px dashed #2e7d32;
    padding: 20px;
    border-radius: 10px;
    background-color: #ffffff;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    background-color: #e8f5e9;
    color: black;
    font-size: 20px;
    text-align: center;
    margin-top: 20px;
}

.info-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #ffffff;
    color: black;
    margin-top: 10px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# Load model
model = tf.keras.models.load_model("potato_disease_model.h5")

# Categories
categories = [
    "Early Blight",
    "Late Blight",
    "Healthy"
]

# Disease information
disease_info = {

    "Early Blight": {
        "Cause": "Fungal infection caused by Alternaria solani.",
        "Prevention": "Use fungicides and remove infected leaves."
    },

    "Late Blight": {
        "Cause": "Disease caused by Phytophthora infestans.",
        "Prevention": "Avoid excessive moisture and use resistant seeds."
    },

    "Healthy": {
        "Cause": "No disease detected.",
        "Prevention": "Maintain proper crop care and irrigation."
    }
}

# Sidebar
st.sidebar.title("🌿 About Project")

st.sidebar.write("""
This AI-powered web application detects potato leaf diseases using Deep Learning and CNN.
""")

st.sidebar.success("Model Accuracy: 96.23%")

# Main title
st.title("🌿 Potato Disease Detection System")

st.write("Upload a potato leaf image to detect disease.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Display image
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Convert image to array
    image_array = np.array(image)

    # Resize image
    image_resized = cv2.resize(image_array, (128,128))

    # Normalize image
    image_normalized = image_resized / 255.0

    # Reshape image
    image_reshaped = np.reshape(image_normalized, (1,128,128,3))

    # Prediction
    prediction = model.predict(image_reshaped)

    class_index = np.argmax(prediction)

    predicted_label = categories[class_index]

    # Confidence score
    confidence = np.max(prediction) * 100

    # Result box
    st.markdown(
        f"""
        <div class="result-box">
        <h2>Prediction: {predicted_label}</h2>
        <h3>Confidence: {confidence:.2f}%</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Disease Information
    st.subheader("Disease Information")

    st.markdown(
        f"""
        <div class="info-box">
        <h4>Cause</h4>
        <p>{disease_info[predicted_label]["Cause"]}</p>

        <h4>Prevention</h4>
        <p>{disease_info[predicted_label]["Prevention"]}</p>
        </div>
        """,
        unsafe_allow_html=True
    )