
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Disease Detection")
st.write(
    "Upload a plant leaf image to predict its disease "
    "using a trained CNN model."
)

# -------------------------------
# Load Trained Model
# -------------------------------
MODEL_PATH = Path(__file__).parent / "plant_disease_cnn.keras"


@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_cnn_model()
except Exception as e:
    st.error(f"Unable to load the model: {e}")
    st.stop()


# -------------------------------
# Class Names
# Same order as your test notebook
# -------------------------------
class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]


# -------------------------------
# Format Predicted Class
# -------------------------------
def format_disease_name(name):
    name = name.replace("___", " - ")
    name = name.replace("_", " ")
    return name


# -------------------------------
# Upload Image
# -------------------------------
uploaded_file = st.file_uploader(
    "Choose a plant leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, caption="Your plant leaf", width=350)

    if st.button("Predict Disease", type="primary"):

        with st.spinner("Analyzing the leaf image..."):

            # Same preprocessing as your test notebook
            image = image.resize((128, 128))

            input_array = tf.keras.utils.img_to_array(image)
            input_array = np.expand_dims(input_array, axis=0)

            # Model prediction
            predictions = model.predict(
                input_array,
                verbose=0
            )

            result_index = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][result_index]) * 100

            predicted_class = class_names[result_index]
            disease_name = format_disease_name(predicted_class)

        st.success("Prediction completed!")

        st.subheader("Prediction Result")
        st.write(f"**Predicted class:** {disease_name}")
        st.metric("Model confidence", f"{confidence:.2f}%")

        st.caption(
            "Confidence is the model's predicted score, "
            "not a guarantee that the diagnosis is correct."
        )

        if "healthy" in predicted_class.lower():
            st.info("The model predicts that the leaf belongs to a healthy class.")
        else:
            st.warning(
                "The model predicts a disease class. "
                "Confirm the result before taking plant treatment decisions."
            )
