# Import necessary libraries

# Python In-built packages
from pathlib import Path
import PIL.Image

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Object Detection And Tracking using YOLOv8")

# Sidebar configuration
st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio("Select Task", ['Detection', 'Segmentation'])

confidence = st.sidebar.slider("Select Model Confidence", 25, 100, 40) / 100

# Selecting Detection Or Segmentation model path
model_path = Path(settings.DETECTION_MODEL if model_type == 'Detection' else settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
    st.sidebar.success("Model loaded successfully.")
except Exception as ex:
    st.sidebar.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

# Sidebar Image/Video Config
st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio("Select Source", settings.SOURCES_LIST)

def display_image(image, caption, use_column_width=True):
    """Helper function to display an image in Streamlit."""
    st.image(image, caption=caption, use_column_width=use_column_width)

def get_highest_confidence_box(boxes):
    """Helper function to get the highest confidence box from predictions."""
    highest_conf_box = max(boxes, key=lambda box: box.conf.item())
    return highest_conf_box

def load_and_display_image():
    """Function to load and display the uploaded or default image."""
    source_img = st.sidebar.file_uploader("Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = Path(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                display_image(default_image_path, "Default Image")
            else:
                uploaded_image = PIL.Image.open(source_img)
                display_image(source_img, "Uploaded Image")
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = Path(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(default_detected_image_path)
            display_image(default_detected_image_path, 'Detected Image')
        else:
            try:
                res = model.predict(uploaded_image, conf=confidence)
                boxes = res[0].boxes

                if len(boxes) > 0:
                    highest_conf_box = get_highest_confidence_box(boxes)
                    res[0].boxes = [highest_conf_box]
                    highest_conf_plot = res[0].plot()[:, :, ::-1]
                    display_image(highest_conf_plot, 'Detected Image')

                    with st.expander("Detection Results"):
                        st.write("Highest Confidence Box:")
                        st.write(highest_conf_box.data)
                else:
                    st.warning("No objects detected.")
            except Exception as ex:
                st.error("Error occurred during detection.")
                st.error(ex)

if source_radio == settings.IMAGE:
    load_and_display_image()
else:
    st.error("Please select a valid source type!")
