import streamlit as st
from Utils.predict import predict_text, predict_image, load_text_model_and_scaler
from Utils.lime_explainer import explain_with_lime, create_lime_explainer
from Utils.gradcam_utils import generate_gradcam
import joblib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import warnings
import logging
import os

logging.getLogger("streamlit").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=RuntimeWarning)
st.set_page_config(page_title="Sickle Cell Anemia Detector", layout="centered")
# Custom CSS for dark theme and styling
st.markdown("""
    <style>
    body {
        background-color: #000000;
    }
    .main {
        background-color: #000000;
        color: white;
    }
    div.stSelectbox > label, .stTextInput > label, .stNumberInput > label, .stFileUploader > label {
        color: #FFA500 !important;
    }
    .stButton button {
        background-color: #FFA500;
        color: black;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 0.5rem;
    }
    .stButton button:hover {
        background-color: #FF8C00;
        color: white;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #FFA500;
    }
    .stSubheader {
        color: #FFA500;
    }
    .css-1d391kg {
        background-color: #1e1e1e !important;
        border-radius: 10px;
        padding: 10px;
    }
    .reportview-container {
        background-color: black;
    }
    </style>
""", unsafe_allow_html=True)

# Page setup

st.title("🧬 Sickle Cell Anemia Detection")
st.markdown("Choose input type, upload your data, and hit **Predict**.")

# Load models
svm, scaler = load_text_model_and_scaler(
    r"C:\Users\Lovely Upadhyay\Desktop\Major Project\Models\svm_model.pkl",
    r"C:\Users\Lovely Upadhyay\Desktop\Major Project\Models\scaler.pkl"
)
image_model_path = r"C:\Users\Lovely Upadhyay\Desktop\Major Project\Models\fine_tuned_efficientnet.pth"
X_train_res = joblib.load(r"C:\Users\Lovely Upadhyay\Desktop\Major Project\Models\X_train_res.pkl")

explainer = create_lime_explainer(
    X_train_res,
    feature_names=['RBC','PCV','MCV','MCH','MCHC','RDW','TLC','PLT_/MM3','HGB'],
    class_names=['Normal', 'Sickle Cell']
)

# Input type
input_type = st.selectbox("Choose type of input:", ["Text Only", "Image Only", "Text + Image"])
user_input = {}
image_file = None

# Input layout
if input_type == "Text + Image":
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🔢 Enter Numeric Features")
        for col in ['RBC', 'PCV', 'MCV', 'MCH', 'MCHC', 'RDW', 'TLC', 'PLT_/MM3', 'HGB']:
            user_input[col] = st.number_input(col, value=0.0, format="%.2f")
    with col2:
        st.markdown("#### 🖼️ Upload Image")
        image_file = st.file_uploader("Browse Image", type=["jpg", "jpeg", "png"])

elif input_type == "Text Only":
    st.markdown("#### 🔢 Enter Numeric Features")
    for col in ['RBC', 'PCV', 'MCV', 'MCH', 'MCHC', 'RDW', 'TLC', 'PLT_/MM3', 'HGB']:
        user_input[col] = st.number_input(col, value=0.0, format="%.2f")

elif input_type == "Image Only":
    st.markdown("#### 🖼️ Upload Image")
    image_file = st.file_uploader("Browse Image", type=["jpg", "jpeg", "png"])

# Centered Predict Button
st.markdown("<div style='text-align: center; padding-top: 10px;'>", unsafe_allow_html=True)
predict_triggered = st.button("🚀 Predict")
st.markdown("</div>", unsafe_allow_html=True)

# Prediction Logic
if predict_triggered:
    use_text = input_type in ["Text Only", "Text + Image"]
    use_image = input_type in ["Image Only", "Text + Image"]

    if use_text:
        probs_text, pred_text, input_scaled = predict_text(user_input, svm, scaler)
        st.subheader("🧠 Text Prediction")
        st.markdown(f"<div style='font-size:18px;'>Prediction: <b style='color:#FFA500'>{'Sickle Cell' if pred_text else 'Normal'}</b></div>", unsafe_allow_html=True)
        lime_exp, lime_fig = explain_with_lime(input_scaled, svm, explainer)
        st.markdown("**LIME Explanation:**")
        st.pyplot(lime_fig, clear_figure=True)

    if use_image and image_file is not None:
        temp_path = "temp_uploaded.jpg"
        with open(temp_path, "wb") as f:
            f.write(image_file.read())

        probs_img, pred_img, model, input_tensor, image_pil = predict_image(temp_path, image_model_path)
        st.subheader("🖼️ Image Prediction")
        st.markdown(f"<div style='font-size:18px;'>Prediction: <b style='color:#FFA500'>{'Sickle Cell' if pred_img else 'Normal'}</b></div>", unsafe_allow_html=True)

        cam_img = generate_gradcam(model, input_tensor, image_pil, target_class=1)
        st.image(cam_img, caption="Grad-CAM Explanation", width=400)

        os.remove(temp_path)

    if use_text and use_image:
        st.subheader("🔁 Combined Prediction")
        final_probs = (np.array(probs_text) + np.array(probs_img)) / 2
        final_pred = np.argmax(final_probs)
        st.markdown(f"<div style='font-size:18px;'>Combined Prediction: <b style='color:#FFA500'>{'Sickle Cell' if final_pred else 'Normal'}</b></div>", unsafe_allow_html=True)
