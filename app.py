from pathlib import Path
import PIL.Image
import streamlit as st
import settings
import helper

# Directory where FastAPI saves images
UPLOAD_DIR = 'uploaded_images'

# Setting page layout
st.set_page_config(
    page_title="Door Lock Access",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Door Lock Access Using YOLOv8")

# Sidebar configuration
st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio(" ", ['Detection'])

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
st.sidebar.header("Image Config")
source_radio = st.sidebar.radio(" ", ["Upload Image", "Upload from URL", "Load from Uploaded Images"])

def display_image(image, caption, use_column_width=True):
    """Helper function to display an image in Streamlit."""
    st.image(image, caption=caption, use_column_width=use_column_width)

def get_highest_confidence_box(boxes):
    """Helper function to get the highest confidence box from predictions."""
    highest_conf_box = max(boxes, key=lambda box: box.conf.item())
    return highest_conf_box

def load_and_display_image(image_path):
    """Function to load and display an image."""
    try:
        image = PIL.Image.open(image_path)
        display_image(image, "Uploaded Image")
        res = model.predict(image, conf=confidence)
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

def load_and_display_image_from_url(url):
    """Function to load and display an image from a URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = PIL.Image.open(BytesIO(response.content))
        display_image(image, "Uploaded Image from URL")
        res = model.predict(image, conf=confidence)
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

def get_latest_images(directory):
    """Function to get the latest images in the directory."""
    image_paths = list(Path(directory).glob("*.jpg"))
    return sorted(image_paths, key=lambda p: p.stat().st_mtime, reverse=True)

# Handling other upload methods
if source_radio == "Upload Image":
    source_img = st.sidebar.file_uploader("Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    if source_img:
        load_and_display_image(source_img)

elif source_radio == "Upload from URL":
    image_url = st.sidebar.text_input("Enter image URL")
    if st.sidebar.button("Load Image from URL"):
        if image_url:
            load_and_display_image_from_url(image_url)
        else:
            st.sidebar.warning("Please enter a valid URL.")

# Load images if any new images are found in the directory
if source_radio == "Load from Uploaded Images":
    if st.button("Check for new images"):
        latest_images = get_latest_images(UPLOAD_DIR)
        if latest_images:
            for image_path in latest_images:
                st.write(f"Displaying image: {image_path.name}")
                load_and_display_image(image_path)
        else:
            st.warning("No new images found.")

# Running the app with a simple server to handle uploads
if __name__ == "__main__":
    from flask import Flask, request, jsonify
    import os

    app = Flask(__name__)

    @app.route('/upload', methods=['POST'])
    def upload_file():
        if 'file' not in request.files:
            return jsonify({'detail': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'detail': 'No selected file'}), 400
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_DIR, filename))
            return jsonify({'detail': 'File uploaded successfully'}), 200

    app.run(host="0.0.0.0", port=8501)
