import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
from PIL import Image

# Page config
st.set_page_config(
    page_title="Object Detection",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Object Detection with YOLO")

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

detection_type = st.sidebar.radio(
    "Upload Type",
    ["Video", "Image"]
)

# Main content
if model:
    if detection_type == "Image":
        st.subheader("📷 Image Detection")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            # Read image
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Image**")
                st.image(image, use_container_width=True)
            
            with col2:
                st.markdown("**Detected Objects**")
                with st.spinner("Detecting..."):
                    # Run detection
                    results = model(img_array, conf=confidence_threshold, verbose=False)
                    
                    # Get annotated image
                    annotated_img = results[0].plot()
                    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                    
                    st.image(annotated_img, use_container_width=True)
                    
                    # Show detection count
                    num_detections = len(results[0].boxes)
                    st.success(f"✅ Detected {num_detections} objects")
    
    else:  # Video
        st.subheader("🎥 Video Detection")
        uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
        
        if uploaded_file:
            # Save uploaded video temporarily
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Video**")
                st.video(video_path)
            
            with col2:
                st.markdown("**Detected Objects**")
                
                if st.button("🚀 Process Video", type="primary"):
                    # Create output path
                    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='_detected.mp4').name
                    
                    with st.spinner("Processing video..."):
                        cap = cv2.VideoCapture(video_path)
                        
                        # Get video properties
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        
                        # Define codec and create VideoWriter
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                        
                        # Progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        frame_count = 0
                        
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            # Run YOLO detection
                            results = model(frame, conf=confidence_threshold, verbose=False)
                            
                            # Draw bounding boxes on frame
                            annotated_frame = results[0].plot()
                            
                            # Write frame to output video
                            out.write(annotated_frame)
                            
                            frame_count += 1
                            if total_frames > 0:
                                progress = frame_count / total_frames
                                progress_bar.progress(progress)
                                status_text.text(f"Processing: {frame_count}/{total_frames} frames")
                        
                        cap.release()
                        out.release()
                        progress_bar.empty()
                        status_text.empty()
                    
                    st.success(f"✅ Processed {frame_count} frames!")
                    
                    # Store output path in session state
                    st.session_state['output_path'] = output_path
                
                # Display processed video if it exists
                if 'output_path' in st.session_state:
                    st.video(st.session_state['output_path'])
                    
                    # Download button
                    with open(st.session_state['output_path'], 'rb') as f:
                        st.download_button(
                            label="⬇️ Download Processed Video",
                            data=f.read(),
                            file_name="detected_video.mp4",
                            mime="video/mp4"
                        )
else:
    st.error("Model failed to load")

st.markdown("---")
st.markdown("Built with Streamlit and YOLO")
