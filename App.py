import os
import warnings
import base64
import random
import json
from urllib.parse import quote

import streamlit as st
from ultralytics import YOLO
from PIL import Image
from io import BytesIO

# Suppress deprecation warnings
warnings.filterwarnings("ignore")

# Constants
MODEL_PATH = "runs/classify/train/weights/best.pt"
DESCRIPTIONS_PATH = "Data.json"

def initialize_session_state():
    """Initialize all session state variables."""
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'class_descriptions' not in st.session_state:
        st.session_state.class_descriptions = None
    if 'model_classes' not in st.session_state:
        st.session_state.model_classes = None

    # Initialize or reset session state variables for the game
    if 'game_score' not in st.session_state:
        st.session_state.game_score = 0
    if 'game_level' not in st.session_state:
        st.session_state.game_level = 1
    if 'target_class' not in st.session_state:
        st.session_state.target_class = None

def image_to_base64(image):
    """Convert an image to a base64-encoded string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_as_text = base64.b64encode(buffered.getvalue()).decode('utf-8')

def load_model_and_descriptions():
    """Load the YOLO model and class descriptions, caching the result"""
    if st.session_state.model is None or st.session_state.class_descriptions is None:
        with st.spinner('Loading model and descriptions...'):
            # Check if the model file exists
            if not os.path.exists(MODEL_PATH):
                st.error(f"Model file not found at {MODEL_PATH}. Please check the path.")
                return
            
            model = YOLO(MODEL_PATH)

            # Load class descriptions from JSON
            if not os.path.exists(DESCRIPTIONS_PATH):
                st.error(f"Descriptions file not found at {DESCRIPTIONS_PATH}. Please check the path.")
                return

            with open(DESCRIPTIONS_PATH, 'r', encoding='utf-8') as f:
                class_descriptions = json.load(f)
            st.session_state.model = model
            st.session_state.class_descriptions = class_descriptions
            st.session_state.model_classes = list(model.names.values())

def predict_image(image_path):
    """Predict the class of an image using the YOLO model"""
    model = st.session_state.model
    class_descriptions = st.session_state.class_descriptions

    results = model.predict(image_path)
    result = results[0]

    probs = result.probs
    predicted_class_index = probs.top1
    predicted_class = result.names[predicted_class_index]
    confidence = probs.top1conf

    # Get the description for the predicted class
    description = class_descriptions.get(predicted_class, "No description available.")
    # Image processing for display
    image = Image.open(image_path).convert("RGB")
    return  image, predicted_class, confidence, description


def display_prediction_results(image, predicted_class, confidence, description):
    """
    Display the prediction results in a structured format.
    """
    image_col, info_col = st.columns([3, 2])
    with image_col:
        st.image(image, caption=f"Predicted: {predicted_class} ({confidence*100:.2f}%)", use_container_width =True)

    with info_col:
        st.subheader("Prediction Results")
        st.markdown(f"**Prediction:** `{predicted_class}`")
        st.markdown(f"**Confidence:** `{confidence*100:.2f}%`")

        # Display the description
        st.markdown("---")
        st.markdown(f"### About {predicted_class}")
        description_html = f'''
            <div style="
                background-color: #f9f9f9;
                padding: 15px;
                border-radius: 5px;
                border-left: 5px solid #6c6ce9;
                ">
                <p style="font-size: 18px; line-height: 1.5; color: #333333; font-weight: bold;">
                    {description}
                </p>
            </div>
        '''
        st.markdown(description_html, unsafe_allow_html=True)

        # Create two columns for buttons
        col_download, col_share = st.columns(2)
        with col_download:
            # Convert image to base64 for download
            base64_img = image_to_base64(image)
            download_button = f'''
                <a href="data:image/jpg;base64,{base64_img}" download="result.jpg">
                    <button style="
                        background-color: #4CAF50;
                        color: white;
                        padding: 8px 20px;
                        border: none;
                        border-radius: 5px;
                        cursor: pointer;
                        width: 100%;
                    ">
                        📥Download
                    </button>
                </a>
            '''
            st.markdown(download_button, unsafe_allow_html=True)

        with col_share:
            # Prepare Twitter share URL
            twitter_text = f"Check out this art piece that was identified as {predicted_class}!"
            encoded_twitter_text = quote(twitter_text)
            tweet_url = f"https://twitter.com/intent/tweet?text={encoded_twitter_text}"
            
            share_button = f'''
                <a href="{tweet_url}" target="_blank">
                    <button style="
                        background-color: #0d95e8;
                        color: white;
                        padding: 8px 20px;
                        border: none;
                        border-radius: 5px;
                        cursor: pointer;
                        width: 100%;
                    ">
                        <img src="https://cdn-icons-png.flaticon.com/512/733/733579.png" 
                             style="width: 20px; height: 20px;">
                        Share
                    </button>
                </a>
            '''
            st.markdown(share_button, unsafe_allow_html=True)

        # Add hover effects with CSS
        st.markdown(""" 
            <style>
                button:hover {
                    opacity: 0.8;
                    transition: opacity 0.3s;
                }
            </style>
        """, unsafe_allow_html=True)

def culture_detection_tab():
    """Content for the Artifacts scanner tab."""
    st.header(" Artifact Recogination")
    st.markdown("Upload an image or use your camera to detect the art piece.")

    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Upload Image", "Use Camera"],
        format_func=lambda x: f"📁 {x}" if x == "Upload Image" else f"📷 {x}"
    )

    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Save the uploaded file temporarily
            temp_image_path = "temp.jpg"
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Predict the image
            with st.spinner('Processing image...'):
                image, predicted_class, confidence, description = predict_image(temp_image_path)

            # Display results
            display_prediction_results(image ,predicted_class, confidence, description)
        else:
            st.info("Please upload an image to get started.")

    elif input_method == "Use Camera":
        # Capture image from camera
        image_file = st.camera_input("📸 Capture an image")
        if image_file:
            # Save the image temporarily
            temp_image_path = "temp.jpg"
            with open(temp_image_path, "wb") as f:
                f.write(image_file.getbuffer())

            # Predict the image
            with st.spinner('Processing image...'):
                image, predicted_class, confidence, description = predict_image(temp_image_path)

            # Display results
            display_prediction_results(image, predicted_class, confidence, description)

def add_footer():
    """
    Add a footer to the Streamlit app.
    """
    st.markdown("""
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            text-align: center;
            padding: 10px;
            font-size:12px;
        }
        </style>
        <div class="footer">
            Developed by Farida Adham 😎
        </div>
        """, unsafe_allow_html=True)

# The rest of the tabs remain unchanged...

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="MuseSnap",
        page_icon="📷",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state
    initialize_session_state()

    # Custom header
    st.markdown("""
        <style>
        .header {
            font-size:40px;
            font-weight:bold;
            text-align:center;
            padding:10px;
        }
        </style>
        <div class="header">MuseSnap📷</div>
        """, unsafe_allow_html=True)

    # Load the model and descriptions
    load_model_and_descriptions()

    tab_options = ["Artifacts scanner", "Artifacts hunting", "Quiz", "About"]
    selected_tab = st.sidebar.radio(
            "Navigation",
            tab_options,
            format_func=lambda x: {
                "Artifacts scanner": "🔍 Artifacts scanner",
                "Artifacts hunting": "🎯 Artifacts hunting",
                "Quiz": "📝 Quiz",
                "About": "ℹ️ About"
            }[x]
    )
    # Render the selected tab
    if selected_tab == "Artifacts scanner":
        culture_detection_tab()
    elif selected_tab == "Artifacts hunting":
        game_tab()
    elif selected_tab == "Quiz":
        quiz_tab()
    elif selected_tab == "About":
        about_tab()

    # Add the footer
    add_footer()

if __name__ == "__main__":
    main()
