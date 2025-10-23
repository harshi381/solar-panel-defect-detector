import streamlit as st
from PIL import Image
import io
import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

# -------------------- SETTINGS --------------------
MODEL_PATH = "model/best_model.pth"
CLASS_NAMES = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------------------------------------

st.set_page_config(page_title="Solar Panel Classifier", layout="centered")
st.title("⚡ Solar Panel Damage Classifier")
st.write("Upload an image to classify it into one of six categories.")

# --- load model once ---
@st.cache_resource(show_spinner=False)
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    state_dict = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to('cpu')
    model.eval()
    return model

model = load_model()

# --- preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- upload image ---
uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output[0], dim=0).numpy()

    # Show predictions
    st.subheader("Predictions:")
    topk = 3
    top_indices = np.argsort(probs)[::-1][:topk]
    for idx in top_indices:
        st.write(f"{CLASS_NAMES[idx]} — {probs[idx]*100:.2f}% confidence")

st.markdown("---")
st.caption("Created with Streamlit & PyTorch 🧠")
