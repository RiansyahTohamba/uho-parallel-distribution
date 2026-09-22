import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import urllib.request
import os
from pathlib import Path
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Fish Detection with YOLOv8",
    page_icon="🐟",
    layout="wide"
)

st.title("🐟 Fish Detection & Classification with YOLOv8")
st.markdown("Detect and classify fish using YOLOv8 object detection model")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

model_size = st.sidebar.selectbox(
    "Model Size",
    options=["nano", "small", "medium", "large"],
    index=1,
    help="Smaller = faster, Larger = more accurate"
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Minimum confidence score to display detections"
)

iou_threshold = st.sidebar.slider(
    "IOU Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.45,
    step=0.05,
    help="Intersection over Union threshold for NMS"
)

st.sidebar.markdown("---")

# Model mapping
model_mapping = {
    "nano": "yolov8n.pt",
    "small": "yolov8s.pt",
    "medium": "yolov8m.pt",
    "large": "yolov8l.pt"
}

model_file = model_mapping[model_size]

@st.cache_resource
def load_yolo_model(size):
    """Load YOLOv8 model with caching"""
    try:
        model = YOLO(model_mapping[size])
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def detect_free_obj(image, model, conf, iou):
    """Run YOLOv8 inference on image"""
    try:
        # Run inference
        results = model(
            image,
            conf=conf,
            iou=iou,
            imgsz=640,
            verbose=False
        )
        
        return results
    except Exception as e:
        st.error(f"Error during detection: {e}")
        return None

def process_results(image, results):
    """Process YOLO results and annotate image"""
    
    # Get annotated frame
    annotated_frame = results[0].plot()
    
    # Extract detection information
    detections = []
    
    if results[0].boxes is not None:
        for box in results[0].boxes:
            detection_info = {
                "class": results[0].names[int(box.cls)],
                "confidence": float(box.conf),
                "x1": float(box.xyxy[0][0]),
                "y1": float(box.xyxy[0][1]),
                "x2": float(box.xyxy[0][2]),
                "y2": float(box.xyxy[0][3]),
                "width": float(box.xyxy[0][2] - box.xyxy[0][0]),
                "height": float(box.xyxy[0][3] - box.xyxy[0][1])
            }
            detections.append(detection_info)
    
    return annotated_frame, detections

def classify_detected_object(class_name, confidence):
    """Enhanced classification with species info"""
    
    # Fish species database
    fish_database = {
        "fish": {
            "name": "Generic Fish",
            "habitat": "Varies",
            "characteristics": "Aquatic vertebrate"
        },
        "bird": {
            "name": "Seabird",
            "habitat": "Coastal",
            "characteristics": "Flying vertebrate"
        },
        "dog": {
            "name": "Canine",
            "habitat": "Terrestrial",
            "characteristics": "Mammal"
        }
    }
    
    # Search for fish-related classes
    class_lower = class_name.lower()
    
    if "fish" in class_lower or class_lower in fish_database:
        return fish_database.get(class_lower, fish_database["fish"])
    
    return {
        "name": class_name,
        "habitat": "Unknown",
        "characteristics": f"Detected with {confidence:.1%} confidence"
    }

# Load model
st.sidebar.write("**Model Info:**")
st.sidebar.write(f"- Model: YOLOv8{model_size}")
st.sidebar.write(f"- File: {model_file}")
st.sidebar.write("- Task: Object Detection")

with st.spinner(f"Loading YOLOv8{model_size} model..."):
    model = load_yolo_model(model_size)

if model is None:
    st.error("Failed to load model. Please check your installation.")
    st.stop()

st.success("✅ Model loaded successfully")

# Main content
tab1, tab2, tab3 = st.tabs(["Image Upload", "Webcam", "Info"])

with tab1:
    st.header("📸 Image Upload & Detection")
    
    uploaded_file = st.file_uploader(
        "Upload a fish image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Clear image of fish for best results"
    )
    
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        # Display original and settings
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Original Image")
            st.image(image, width="stretch")
            
        with col2:
            st.subheader("Image Details")
            st.write(f"**Size:** {image.size[0]} × {image.size[1]} pixels")
            st.write(f"**Format:** {image.format}")
            st.write(f"**Mode:** {image.mode}")
        
        # Run detection
        if st.button("🔍 Detect Fish", key="detect_image"):
            with st.spinner("Running inference..."):
                results = detect_free_obj(image_np, model, confidence_threshold, iou_threshold)
                
                if results is not None and len(results) > 0:
                    # Process results
                    annotated_image, detections = process_results(image_np, results)
                    
                    # Display annotated image
                    st.subheader("Detection Results")
                    st.image(
                        annotated_image,
                        caption=f"Detected {len(detections)} object(s)",
                        width="stretch"
                    )
                    
                    # Display detection details
                    if len(detections) > 0:
                        st.subheader("📊 Detection Details")
                        
                        for idx, det in enumerate(detections, 1):
                            with st.expander(f"Detection #{idx}: {det['class']} ({det['confidence']:.1%})"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write("**Detection Info:**")
                                    st.write(f"- Class: {det['class']}")
                                    st.write(f"- Confidence: {det['confidence']:.1%}")
                                    st.write(f"- Bounding Box Width: {det['width']:.0f}px")
                                    st.write(f"- Bounding Box Height: {det['height']:.0f}px")
                                
                                with col2:
                                    fish_info = classify_detected_object(det['class'], det['confidence'])
                                    st.write("**Object Classification:**")
                                    st.write(f"- Name: {fish_info['name']}")
                                    st.write(f"- Habitat: {fish_info['habitat']}")
                                    st.write(f"- Characteristics: {fish_info['characteristics']}")
                        
                        # Export results
                        st.subheader("📥 Export Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Save annotated image
                            img_pil = Image.fromarray(annotated_image)
                            buffer = BytesIO()
                            img_pil.save(buffer, format="PNG")
                            st.download_button(
                                label="Download Annotated Image",
                                data=buffer.getvalue(),
                                file_name="fish_detection.png",
                                mime="image/png"
                            )
                        
                        with col2:
                            # Export as text summary
                            summary = "Fish Detection Results\n" + "="*50 + "\n\n"
                            for idx, det in enumerate(detections, 1):
                                summary += f"Detection {idx}:\n"
                                summary += f"  - Class: {det['class']}\n"
                                summary += f"  - Confidence: {det['confidence']:.1%}\n"
                                summary += f"  - Size: {det['width']:.0f}×{det['height']:.0f}px\n\n"
                            
                            st.download_button(
                                label="Download Text Summary",
                                data=summary,
                                file_name="fish_detection.txt",
                                mime="text/plain"
                            )
                    else:
                        st.warning("No objects detected. Try adjusting the confidence threshold.")
                else:
                    st.warning("No detections found. Try a different image or lower the confidence threshold.")

with tab2:
    st.header("📹 Webcam Detection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("⚠️ Webcam detection requires local environment. Use the image upload tab for cloud usage.")
        
        if st.button("Start Webcam Detection", key="start_webcam"):
            st.write("Note: This feature works best in local Streamlit deployments.")
            st.code("""
# To use webcam locally, install:
pip install streamlit-webrtc

# Then use:
from streamlit_webrtc import webrtc_streamer
""")
    
    with col2:
        st.write("**Setup Requirements:**")
        st.write("- Local Python environment")
        st.write("- Webcam access")
        st.write("- streamlit-webrtc library")

with tab3:
    st.header("ℹ️ About YOLOv8")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Specifications")
        st.write("""
        **YOLOv8 (You Only Look Once v8)**
        - Real-time object detection
        - Single-stage detector
        - Fast inference on CPU/GPU
        - Pre-trained on COCO dataset
        
        **Model Sizes:**
        - nano (n): Fastest, smallest
        - small (s): Balanced performance
        - medium (m): Good accuracy
        - large (l): Best accuracy
        """)
    
    with col2:
        st.subheader("Performance Tips")
        st.write("""
        ✅ **For Best Results:**
        - Use clear, well-lit images
        - Fish should be clearly visible
        - Avoid blurry or side-angle shots
        
        ⚡ **Speed Optimization:**
        - Nano model for fastest results
        - Lower confidence threshold
        - Smaller images (640×640)
        
        🎯 **Accuracy Optimization:**
        - Large model for accuracy
        - Higher confidence threshold
        - Larger resolution images
        """)
    
    st.subheader("Installation & Usage")
    st.code("""
# Install requirements
pip install streamlit ultralytics opencv-python pillow

# Run app
streamlit run fish_classifier_yolo.py
    """, language="bash")
    
    st.subheader("Features")
    st.write("""
    ✓ Real-time fish detection and localization
    ✓ Adjustable confidence and IOU thresholds
    ✓ Multiple model sizes (nano to large)
    ✓ Bounding box visualization
    ✓ Detailed detection statistics
    ✓ Export annotated images and reports
    ✓ Cross-platform (Windows, Mac, Linux)
    """)

# Footer
st.markdown("---")
st.markdown("""
    **Powered by:** YOLOv8 | **Framework:** Streamlit | **Version:** v1.0
    
    For more information: [YOLOv8 Documentation](https://docs.ultralytics.com/)
""")