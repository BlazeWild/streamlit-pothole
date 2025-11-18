import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import time

# Page config
st.set_page_config(
    page_title="Real-Time Object Detection",
    page_icon="🎯",
    layout="wide"
)

# Title
st.title("🎯 Real-Time Object Detection with YOLO")

# Load YOLO model
@st.cache_resource
def load_model():
    """Load the YOLO model - auto-downloads if not present"""
    try:
        model = YOLO('yolov8n.pt')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

# Sidebar settings
st.sidebar.header("⚙️ Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05
)

# Main content
if model:
    run = st.checkbox('Start Camera')
    FRAME_WINDOW = st.image([])
    
    if run:
        cap = cv2.VideoCapture(0)
        
        while run:
            ret, frame = cap.read()
            
            if not ret:
                st.error("Failed to access camera")
                break
            
            # Run YOLO detection
            results = model(frame, conf=confidence_threshold, verbose=False)
            
            # Draw detections
            annotated_frame = results[0].plot()
            
            # Convert BGR to RGB
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Display frame
            FRAME_WINDOW.image(annotated_frame)
        
        cap.release()
else:
    st.error("Model failed to load")

st.markdown("---")
st.markdown("Built with Streamlit, OpenCV, and YOLO")
