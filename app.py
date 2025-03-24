import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import sys
import time

# Setup paths
ROOT = Path(__file__).resolve().parent
YOLOV5_PATH = ROOT / "yolov5"
sys.path.append(str(YOLOV5_PATH))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox

# Streamlit page config
st.set_page_config(page_title="🚗 Foggy Detector", layout="centered", page_icon="🌫️")

# Title
st.markdown("<h1 style='text-align: center; color: #00ffcc;'>🚗 Foggy the Object Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload foggy/rainy images to detect objects using YOLOv5.</p>", unsafe_allow_html=True)
st.markdown("---")

# Load YOLOv5 model
@st.cache_resource
def load_model():
    model_path = ROOT / "Trained_Model_2.pt"
    model = DetectMultiBackend(str(model_path), device="cpu")
    return model

model = load_model()
names = model.names if hasattr(model, 'names') else [str(i) for i in range(100)]

# Upload image
uploaded_file = st.file_uploader("📁 Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    np_image = np.array(image)
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    if st.button("🔍 Run Detection"):
        with st.spinner("Running detection..."):
            start = time.time()

            # Preprocess
            img = np_image.copy()
            img0 = img.copy()
            img = letterbox(img, new_shape=640)[0]
            img = img.transpose((2, 0, 1))[::-1]
            img = np.ascontiguousarray(img)

            im = torch.from_numpy(img).to("cpu")
            im = im.float() / 255.0
            if im.ndimension() == 3:
                im = im.unsqueeze(0)

            pred = model(im, augment=False, visualize=False)
            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

            total_detected = 0
            detected_classes = []

            if pred is not None and len(pred):
                pred[:, :4] = scale_boxes(im.shape[2:], pred[:, :4], img0.shape).round()

                for *xyxy, conf, cls in pred:
                    label = f"{names[int(cls)]} {conf:.2f}"
                    total_detected += 1
                    detected_classes.append(names[int(cls)])

                    xyxy = list(map(int, xyxy))
                    cv2.rectangle(img0, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                    cv2.putText(img0, label, (xyxy[0], xyxy[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                end = time.time()
                st.success(f"✅ {total_detected} object(s) detected in {end - start:.2f} seconds.")
                st.info("📦 Classes Detected: " + ", ".join(set(detected_classes)))
                st.image(img0, caption="🔍 Detection Result", use_container_width=True, channels="BGR")
            else:
                st.warning("⚠️ No objects detected. Try a different image.")