import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

# Load model.
model = load_model("mobilenet_model.keras")

classes = ["Bird drop panel","Cleaned panel","Dust panel","Electrical problem","Physical damage","Snow problem"]


# User input preprocessing.
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Main title.
st.set_page_config(page_title="Solar Defect Detection", layout="wide")
st.title("☀️ Solar Panel Defect Detection ")
st.markdown("---")

# Sidebar.
st.sidebar.title("📊 About")
st.sidebar.markdown("---")
st.sidebar.info("Upload a Solar Panel Image, To Detect Defects Using Deep Learning.")

uploaded_file = st.file_uploader("📤**Upload Solar Panel Image**", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.subheader("📷 Uploaded Image")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(img, width=400)


    if st.button("🔍 Predict"):                            # For prediction.

        with st.spinner("Predicting... ⏳"):

            processed = preprocess_image(img)

            prediction = model.predict(processed)
            class_index = np.argmax(prediction)

        st.subheader("🔮 Prediction Result")
        st.success(f"Prediction: {classes[class_index]}")
    





  