import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from ultralytics import YOLO
import cv2
import numpy as np
import av

# Page config
st.set_page_config(
    page_title="Real-Time Object Detection",
    page_icon="🎯",
    layout="wide"
)

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

# Video transformer class
class YOLOTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model
        
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if self.model:
            # Run YOLO detection
            results = self.model(img, conf=confidence_threshold, verbose=False)
            
            # Draw detections
            img = results[0].plot()
        
        return img

# Main content
if model:
    st.info("👇 Click START to allow camera access")
    
    webrtc_streamer(
        key="object-detection",
        video_transformer_factory=YOLOTransformer,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
else:
    st.error("Model failed to load")

st.markdown("---")
st.markdown("Built with Streamlit, OpenCV, and YOLO")
