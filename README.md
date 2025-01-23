# Plant Disease Detection System

## Overview
The Plant Disease Detection System is a web application built using Streamlit and TensorFlow that allows users to upload images of plants and receive predictions about potential diseases affecting them. This tool aims to support sustainable agriculture by enabling farmers and enthusiasts to identify plant diseases quickly and accurately.

## Features
- **Image Upload**: Users can upload images of plants in various formats (JPEG, PNG).
- **Disease Prediction**: The application uses a pre-trained Convolutional Neural Network (CNN) model to predict plant diseases.
- **User-Friendly Interface**: A simple and intuitive interface built with Streamlit.

## Technologies Used
- **Python**: The programming language used for the application.
- **Streamlit**: A framework for building web applications quickly.
- **TensorFlow**: An open-source library for machine learning used to build the CNN model.
- **OpenCV**: A library for image processing.
- **NumPy**: A library for numerical computations in Python.
- **Pillow (PIL)**: A library for image handling.

## Installation

### Prerequisites
Make sure you have Python 3.7 or higher installed on your machine. You will also need to install the required libraries.

### Step 1: Clone the Repository
git clone https://github.com/Akshint0407/plant-disease-detection.git
cd plant-disease-detection

### Step 2: Install Required Packages
You can install the necessary packages using pip. It’s recommended to create a virtual environment first.
pip install -r requirements.txt

### Step 3: Run the Application
To start the Streamlit application, run the following command:
streamlit run app.py

## Usage
1. Navigate to the application in your web browser (usually at `http://localhost:8501`).
2. Select "DISEASE RECOGNITION" from the sidebar.
3. Upload an image of a plant.
4. Click "Show Image" to view the uploaded image.
5. Click "Predict" to get the disease prediction.

## Model Information
The model used in this application is a Convolutional Neural Network (CNN) trained on a dataset of various plant diseases. The model is capable of identifying multiple diseases across different types of plants.

## Contributing
Contributions are welcome! If you have suggestions or improvements, please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE] file for details.



