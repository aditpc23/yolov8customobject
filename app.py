# Import necessary libraries
from pathlib import Path
import time
import PIL.Image
import streamlit as st
import telegram

# External packages
import settings
import helper

# Directory where Telegram bot saves images and results
UPLOAD_DIR = 'uploads'
RESULT_DIR = 'results'

# Load your bot token
TELEGRAM_BOT_TOKEN = '7461035655:AAGWX7UlpESuItLmS7pLxWNYvXswouktnss'
bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)

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
source_radio = st.sidebar.radio("Select Source", ["Upload Image", "Telegram"])

def display_image(image, caption, use_column_width=True):
    """Helper function to display an image in Streamlit."""
    st.image(image, caption=caption, use_column_width=use_column_width)

def get_highest_confidence_box(boxes):
    """Helper function to get the highest confidence box from predictions."""
    highest_conf_box = max(boxes, key=lambda box: box.conf.item())
    return highest_conf_box

def load_and_display_image(image_path, save_result=False):
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

            if save_result:
                result_path = os.path.join(RESULT_DIR, f"{Path(image_path).stem}_result.jpg")
                PIL.Image.fromarray(highest_conf_plot).save(result_path)
                send_result_to_telegram(image_path, result_path)

            with st.expander("Detection Results"):
                st.write("Highest Confidence Box:")
                st.write(highest_conf_box.data)
        else:
            st.warning("No objects detected.")
    except Exception as ex:
        st.error("Error occurred during detection.")
        st.error(ex)

def send_result_to_telegram(image_path, result_path):
    """Function to send the detection result to the user via Telegram."""
    try:
        with open(f"{image_path}.chat_id", 'r') as f:
            chat_id = f.read().strip()
        
        bot.send_message(chat_id=chat_id, text="Here is the detection result:")
        bot.send_photo(chat_id=chat_id, photo=open(result_path, 'rb'))
    except Exception as ex:
        st.error("Error occurred while sending the result to Telegram.")
        st.error(ex)

if source_radio == "Upload Image":
    source_img = st.sidebar.file_uploader("Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    if source_img:
        load_and_display_image(source_img)

elif source_radio == "Telegram":
    latest_image = None
    st.sidebar.write("Upload an image via the Telegram bot and wait for it to process.")
    
    if st.sidebar.button("Check for new images"):
        uploaded_images = list(Path(UPLOAD_DIR).glob("*.jpg"))
        if uploaded_images:
            latest_image = max(uploaded_images, key=lambda p: p.stat().st_mtime)
            st.sidebar.success(f"Found new image: {latest_image.name}")
        else:
            st.sidebar.warning("No new images found.")

    if latest_image:
        load_and_display_image(latest_image, save_result=True)

else:
    st.error("Please select a valid source type!")
