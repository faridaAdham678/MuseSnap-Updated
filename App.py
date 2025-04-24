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
            model = YOLO(MODEL_PATH)
            # Load class descriptions from JSON
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
        else:
            st.info("Please capture an image to get started.")

def game_tab():
    """Content for the Artifacts hunting tab."""
    st.header("Hunt the Artifacts Game 🎯")
    st.markdown("Try to capture an image of the given art piece!")

    # Display game stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Level", st.session_state.game_level)
    with col2: 
        st.metric("Score", st.session_state.game_score)

    # Get the list of class names from the model
    if st.session_state.model_classes is None:
        st.session_state.model_classes = list(st.session_state.model.names.values())

    # Randomly select a target class if not already set
    if st.session_state.target_class is None:
        st.session_state.target_class = random.choice(st.session_state.model_classes)

    st.info(f"🎯 Your target: **{st.session_state.target_class}**")

    # Button to get a new target class
    if st.button("🎯 New Target"):
        st.session_state.target_class = random.choice(st.session_state.model_classes)

    # Capture image from camera
    image_file = st.camera_input("📸 Capture the target art piece")
    if image_file:
        with st.spinner("Checking your capture..."):
            temp_image_path = "temp.jpg"
            with open(temp_image_path, "wb") as f:
                f.write(image_file.getbuffer())

            image, predicted_class, confidence, description = predict_image(temp_image_path)

            # Display results
            st.image(image, caption=f"Predicted: {predicted_class} ({confidence*100:.2f}%)", use_container_width =True)

        # Check if the predicted class matches the target class
        if predicted_class == st.session_state.target_class:
            st.success(f"🎉 Congratulations! You captured the correct art piece!")
            st.balloons()
            st.session_state.game_score += 1
            if st.session_state.game_score % 3 == 0:  # Level up every 3 correct answers
                    st.session_state.game_level += 1
                    st.session_state.game_score = 0 # Reset score for new level
            st.session_state.target_class = None  # Reset target for next round
        else:
            st.error(f"❌ Oops! That's **{predicted_class}**. Try again!")

def quiz_tab():
    """Content for the Quiz tab."""
    st.header("📝 Art Recognition Quiz")
    st.markdown("Test your knowledge by matching the description to the correct art piece!")

    # Initialize session state variables for the quiz
    if 'quiz_score' not in st.session_state:
        st.session_state.quiz_score = 0
    if 'quiz_question' not in st.session_state:
        st.session_state.quiz_question = None
    if 'quiz_options' not in st.session_state:
        st.session_state.quiz_options = None
    if 'quiz_correct_answer' not in st.session_state:
        st.session_state.quiz_correct_answer = None
    if 'quiz_feedback' not in st.session_state:
        st.session_state.quiz_feedback = None

    # Get the list of class names from the model
    if st.session_state.model_classes is None:
        st.session_state.model_classes = list(st.session_state.model.names.values())

    # Initialize a new quiz question if needed
    if st.session_state.quiz_question is None:
        # Randomly select a class
        correct_class = random.choice(st.session_state.model_classes)
        correct_description = st.session_state.class_descriptions.get(correct_class, "No description available.")

        # Generate multiple-choice options
        other_classes = [cls for cls in st.session_state.model_classes if cls != correct_class]
        options = random.sample(other_classes, min(3, len(other_classes)))
        options.append(correct_class)
        random.shuffle(options)

        # Store in session state
        st.session_state.quiz_question = correct_description
        st.session_state.quiz_options = options
        st.session_state.quiz_correct_answer = correct_class
        st.session_state.quiz_feedback = None  # Reset feedback

    # Display the description
    st.markdown(f"### Description:")
    description_html = f'''
        <div style="
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            border-left: 5px solid #6c6ce9;
            ">
            <p style="font-size: 18px; line-height: 1.5; color: #333333; font-weight: bold;">
                {st.session_state.quiz_question}
            </p>
        </div>
    '''
    st.markdown(description_html, unsafe_allow_html=True)

    # Display options as a radio button list
    user_answer = st.radio("Select the art piece that matches the above description:", st.session_state.quiz_options)

    # Check the user's answer when they click the "Submit Answer" button
    if st.button("Submit Answer"):
        if user_answer == st.session_state.quiz_correct_answer:
            st.success("🎉 Correct! Great job!")
            st.session_state.quiz_score += 1
        else:
            st.error(f"❌ Incorrect. The correct answer was **{st.session_state.quiz_correct_answer}**.")
        st.session_state.quiz_feedback = True

    if st.session_state.quiz_feedback:
        st.write(f"**Your Score:** {st.session_state.quiz_score}")
        if st.button("Next Question"):
            # Reset quiz question to generate a new one
            st.session_state.quiz_question = None
            st.session_state.quiz_options = None
            st.session_state.quiz_correct_answer = None
            st.session_state.quiz_feedback = None
    else:
        st.write(f"**Your Score:** {st.session_state.quiz_score}")

def about_tab():
    """Content for the About tab with minimal CSS."""
    # App Introduction
    st.title("📷 MuseSnap")
    
    st.info("""
        Welcome to an innovative Website that bridges technology and cultural heritage! 
        Our website uses advanced AI to help you explore and learn about various cultural artifacts and artworks.
    """)

    # Key Features Section
    st.header('✨ Key Features')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("🔍 **Image Recognition**\n- Advanced AI-powered detection of cultural artifacts")
        st.success("📊 **Smart Analysis**\n- Accurate predictions with confidence scores")
        st.success("📱 **Camera Integration**\n- Real-time detection using your device's camera")
    
    with col2:
        st.success("🎮 **Interactive Learning**\n- Learn through games and quizzes")
        st.success("💾 **Save & Share**\n- Download results and share on social media")
        st.success("📚 **Rich Information**\n- Detailed descriptions of each artifact")

    # How It Works Section
    st.header('🔧 How It Works')
    
    st.info("""
        1. Upload an image or use your camera to capture an art piece
        2. Our AI model analyzes the image using YOLO technology
        3. Receive instant identification with confidence scores
        4. Learn about the artifact through fun and lighthearted detailed descriptions
        5. Save or share your discoveries
    """)

    # Additional Information
    st.header('📌 Additional Information')
    st.success("""
        This application uses YOLO (You Only Look Once) technology to provide:
        - Real-time object detection
        - High accuracy predictions
        - Fast processing speed
        - User-friendly interface
    """)
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