import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import gdown  # To download files from Google Drive

# Function to download the model from Google Drive
def download_model():
    # The Google Drive file ID from my shareable link
    file_id = '1GMYi_Mj8qyDmnXFfXZ-14amaWvKmYb33'  
    url = f'https://drive.google.com/uc?id={file_id}'
    
    # Download the model to the local directory
    gdown.download(url, 'CNN_plantdiseases_model.keras', quiet=False)

# Function to load the model and make predictions
def model_predict(image_path):
    print(f"Predicting for image: {image_path}")
    
    # Load the model
    model = tf.keras.models.load_model('CNN_plantdiseases_model.keras')
    
    # Load and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image could not be loaded. Check if the file exists: {image_path}")
    
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype="float32") / 255.0
    img = img.reshape(1, 224, 224, 3)

    # Make the prediction
    prediction = np.argmax(model.predict(img), axis=-1)[0]
    return prediction

# Sidebar
st.sidebar.title("Plant Disease Detection System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Import Image from pillow to open images
from PIL import Image

# Add a background image
def add_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({image_url});
            background-size: cover;
            background-position: center;
            color: white;  /* Change text color to white for better visibility */
        }}
        .popup {{
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 300px;
            background-color: rgba(0, 0, 0, 0.9); /* Black background */
            border-radius: 10px;
            padding: 20px;
            z-index: 1000;
            color: white; /* White text */
        }}
        .popup h4 {{
            margin-top: 0;
        }}
        .button-container {{
            position: absolute;
            top: 10px;
            right: 10px;
        }}
        .button {{
            background-color: rgba(255, 255, 255, 0.7); /* Semi-transparent button */
            border: none;
            border-radius: 5px;
            padding: 10px 15px;
            cursor: pointer;
            margin-left: 5px; /* Space between buttons */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Replace with your new static image URL
add_background("https://plus.unsplash.com/premium_photo-1663962158789-0ab624c4f17d?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MXx8cGxhbnRzfGVufDB8fDB8fHww")

# Main Page
if app_mode == "HOME":
    st.markdown(
        "<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>",
        unsafe_allow_html=True,
    )

# Prediction Page
elif app_mode == "DISEASE RECOGNITION":
    st.header("Plant Disease Detection System for Sustainable Agriculture")
    test_image = st.file_uploader("Choose an Image:")

    if test_image is not None:
        # Define the save path
        save_path = os.path.join(os.getcwd(), test_image.name)
        
        # Save the file to the working directory
        with open(save_path, "wb") as f:
            f.write(test_image.getbuffer())

        # Display the uploaded image
        if st.button("Show Image"):
            st.image(test_image, width=4, use_container_width=True)

        # Predict button
        if st.button("Predict"):
            st.snow()
            st.write("Our Prediction")
            
            # Download the model
            download_model()

            # Make prediction using the downloaded model
            result_index = model_predict(save_path)

            # Define the class names (same as before)
            class_name = [
                "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
                "Apple___healthy", "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew",
                "Cherry_(including_sour)___healthy", "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot",
                "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight",
                "Corn_(maize)___healthy", "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
                "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
                "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
                "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
                "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
                "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
                "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
                "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites_Two-spotted_spider_mite", 
                "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", 
                "Tomato___healthy"
            ]

            # Logic for healthy vs infected
            if result_index == 3 or result_index == 4 or result_index == 6 or result_index == 8 or result_index == 12 or result_index == 13 or result_index == 15 or result_index == 17 or result_index == 19 or result_index == 21 or result_index == 23 or result_index == 25 or result_index == 27 or result_index == 29 or result_index == 31:  # Healthy plant indexes
                st.success(f"Congratulations! Your plant does not have any disease!!")
            else:  # Infected plant
                st.error(f"Uh-oh!! Your plant is infected, its disease is: {class_name[result_index]}")
