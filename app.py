import streamlit as st
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from PIL import Image
from ultralytics import YOLO
import joblib

# Load your models
model_16_features = joblib.load('16_catboost_model.pkl')
model_28_features = joblib.load('catboost_model.pkl')
image_model = YOLO('pistachio_image_classification.pt')  # Load YOLOv5 or YOLOv8 model (replace with your model path if needed)

# Feature lists
minmax_features_16 = ['ECCENTRICITY', 'EXTENT', 'SOLIDITY', 'ROUNDNESS', 'SHAPEFACTOR_1', 'SHAPEFACTOR_2', 'SHAPEFACTOR_4']
standard_features_16 = ['AREA', 'PERIMETER', 'MAJOR_AXIS', 'MINOR_AXIS', 'EQDIASQ', 'CONVEX_AREA', 'ASPECT_RATIO', 'COMPACTNESS', 'SHAPEFACTOR_3']

minmax_features_28 = ['Area', 'Perimeter', 'Major_Axis', 'Minor_Axis', 'Convex_Area', 'Solidity', 'Roundness', 'Compactness', 'Shapefactor_1', 
                      'Shapefactor_2', 'Shapefactor_3', 'Shapefactor_4', 'Mean_RR', 'Mean_RG', 'Mean_RB', 'StdDev_RR', 'StdDev_RG', 
                      'StdDev_RB', 'Skew_RR', 'Skew_RG', 'Skew_RB', 'Kurtosis_RR', 'Kurtosis_RG', 'Kurtosis_RB']
standard_features_28 = ['Eccentricity', 'Extent', 'Aspect_Ratio']

# Scaling
minmax_scaler_16 = MinMaxScaler()
standard_scaler_16 = StandardScaler()

minmax_scaler_28 = MinMaxScaler()
standard_scaler_28 = StandardScaler()

# Streamlit UI
st.title("Pistachio Image Dataset Predictor")

# Selection of options
option = st.sidebar.selectbox("Choose prediction type", ["16 Features", "28 Features", "Image Classification"])

if option == "16 Features":
    st.subheader("Enter values for 16 features")

    # Input sliders for the 16 features
    inputs_minmax = [st.slider(f"{feature}", 0.0, 1.0, 0.5) for feature in minmax_features_16]
    inputs_standard = [st.slider(f"{feature}", 0.0, 1000.0, 500.0) for feature in standard_features_16]

    # Scaling input values
    scaled_minmax = minmax_scaler_16.transform([inputs_minmax])
    scaled_standard = standard_scaler_16.transform([inputs_standard])

    # Concatenate scaled features
    combined_inputs_16 = np.concatenate([scaled_minmax, scaled_standard], axis=1)

    # Predict
    if st.button("Predict"):
        prediction_16 = model_16_features.predict(combined_inputs_16)
        st.write(f"Prediction: {prediction_16[0]}")

elif option == "28 Features":
    st.subheader("Enter values for 28 features")

    # Input sliders for the 28 features
    inputs_minmax = [st.slider(f"{feature}", 0.0, 1.0, 0.5) for feature in minmax_features_28]
    inputs_standard = [st.slider(f"{feature}", 0.0, 1000.0, 500.0) for feature in standard_features_28]

    # Scaling input values
    scaled_minmax = minmax_scaler_28.transform([inputs_minmax])
    scaled_standard = standard_scaler_28.transform([inputs_standard])

    # Concatenate scaled features
    combined_inputs_28 = np.concatenate([scaled_minmax, scaled_standard], axis=1)

    # Predict
    if st.button("Predict"):
        prediction_28 = model_28_features.predict(combined_inputs_28)
        st.write(f"Prediction: {prediction_28[0]}")

elif option == "Image Classification":
    st.subheader("Upload an image for classification")
    
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image as required by YOLO (resizing, conversion to numpy array)
        image = np.array(image)

        # YOLO model expects an image path, so save the image temporarily
        img_path = "temp_image.jpg"
        Image.fromarray(image).save(img_path)

        if st.button("Classify"):
            # Perform classification using YOLOv5
            results = image_model(img_path)  # This runs inference

            # Extract predicted class and confidence score
            if results:
                result = results[0]  # Take the first prediction result
                st.write(f"Class: {result.names[result.pred[0, -1]]}")  # Display class
                st.write(f"Confidence: {result.pred[0, -2]}")  # Display confidence score
            else:
                st.write("No objects detected in the image.")
