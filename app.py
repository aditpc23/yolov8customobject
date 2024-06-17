# Import necessary libraries

# Python In-built packages
from pathlib import Path
import PIL

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
model_type = st.sidebar.radio(
    "Select Task", ['Detection', 'Segmentation'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation model path
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
else:
    model_path = Path(settings.SEGMENTATION_MODEL)  # Assuming you have a segmentation model path

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
    st.sidebar.success("Model loaded successfully.")
except Exception as ex:
    st.sidebar.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

# Sidebar Image/Video Config
st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = Path(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = Path(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            try:
                # Perform prediction automatically when an image is uploaded
                res = model.predict(uploaded_image, conf=confidence)
                boxes = res[0].boxes

                if len(boxes) > 0:

                    # Extract confidence scores and find the highest confidence box
                    highest_conf_box = None
                    highest_confidence = 0.0

                    for box in boxes:
                        # Assuming the confidence score is present in the data array (often the 5th element)
                        confidence_score = box.conf.item()  # Adjust the index if necessary
                        if confidence_score > highest_confidence:
                            highest_conf_box = box
                            highest_confidence = confidence_score

                    if highest_conf_box is not None:
                        highest_conf_box_data = highest_conf_box.data

                        # Plot only the highest confidence box
                        res[0].boxes = [highest_conf_box]  # Replace boxes with only the highest confidence box
                        highest_conf_plot = res[0].plot()[:, :, ::-1]  # Ensure correct color format
                        st.image(highest_conf_plot, caption='Detected Image', use_column_width=True)

                        with st.expander("Detection Results"):
                            st.write("Highest Confidence Box:")
                            st.write(highest_conf_box_data)
                    else:
                        st.warning("No objects detected with the required confidence.")
                else:
                    st.warning("No objects detected.")
            except Exception as ex:
                st.error("Error occurred during detection.")
                st.error(ex)

else:
    st.error("Please select a valid source type!")
