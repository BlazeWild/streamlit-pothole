import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# Page config
st.set_page_config(
    page_title="Real-Time Object Detection",
    page_icon="🎯",
    layout="wide"
)

# Title
st.title("🎯 Real-Time Object Detection with YOLO")
st.markdown("Live webcam feed with object detection and localization")

# Load YOLO model
@st.cache_resource
def load_model():
    """Load the YOLO model - auto-downloads if not present"""
    try:
        model = YOLO('yolov8n.pt')  # Will auto-download
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
    step=0.05,
    help="Minimum confidence for detection"
)

show_labels = st.sidebar.checkbox("Show Labels", value=True)
show_confidence = st.sidebar.checkbox("Show Confidence", value=True)

# Video processor class
class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.confidence = confidence_threshold
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if self.model:
            # Run YOLO detection
            results = self.model(img, conf=confidence_threshold, verbose=False)
            
            # Get detections
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get confidence and class
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = self.model.names[cls]
                    
                    # Draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Prepare label
                    label_parts = []
                    if show_labels:
                        label_parts.append(class_name)
                    if show_confidence:
                        label_parts.append(f"{conf:.2f}")
                    
                    if label_parts:
                        label = " ".join(label_parts)
                        
                        # Draw label background
                        (label_width, label_height), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                        )
                        cv2.rectangle(
                            img,
                            (x1, y1 - label_height - 10),
                            (x1 + label_width, y1),
                            (0, 255, 0),
                            -1
                        )
                        
                        # Draw label text
                        cv2.putText(
                            img,
                            label,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 0),
                            2
                        )
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Main content
st.markdown("### 📹 Live Detection")

if model:
    # WebRTC configuration
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        video_processor_factory=YOLOVideoProcessor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    st.markdown("---")
    st.info("💡 Allow camera access when prompted. Adjust settings in the sidebar.")
else:
    st.error("⚠️ Model failed to load. Please check your internet connection.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, OpenCV, and YOLO")
