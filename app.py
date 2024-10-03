import streamlit as st
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from PIL import Image
from ultralytics import YOLO
import joblib
import catboost

# Load your models
try:
    model_16_features = joblib.load('16_catboost_model.pkl')
    model_28_features = joblib.load('catboost_model.pkl')
    image_model = YOLO('pistachio_image_classification.pt')  # Load YOLOv8 model
except FileNotFoundError as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Feature lists for MinMax Scaling (16 and 28 features)
minmax_features_16 = ['ECCENTRICITY', 'EXTENT', 'SOLIDITY', 'ROUNDNESS', 'SHAPEFACTOR_1', 'SHAPEFACTOR_2', 'SHAPEFACTOR_4']
standard_features_16 = ['AREA', 'PERIMETER', 'MAJOR_AXIS', 'MINOR_AXIS', 'EQDIASQ', 'CONVEX_AREA', 'ASPECT_RATIO', 'COMPACTNESS', 'SHAPEFACTOR_3']

minmax_features_28 = [
    'Area', 'Perimeter', 'Major_Axis', 'Minor_Axis', 'Convex_Area', 'Solidity', 'Roundness', 'Compactness', 'Shapefactor_1', 
    'Shapefactor_2', 'Shapefactor_3', 'Shapefactor_4', 'Mean_RR', 'Mean_RG', 'Mean_RB', 'StdDev_RR', 'StdDev_RG', 'StdDev_RB', 
    'Skew_RR', 'Skew_RG', 'Skew_RB', 'Kurtosis_RR', 'Kurtosis_RG', 'Kurtosis_RB'  # Kurtosis_RB is present here
]
standard_features_28 = ['Eccentricity', 'Extent', 'Aspect_Ratio']

# Sample data to fit scalers (replace this with your actual data)
sample_data_16_minmax = np.random.rand(100, len(minmax_features_16))
sample_data_16_standard = np.random.rand(100, len(standard_features_16))

sample_data_28_minmax = np.random.rand(100, len(minmax_features_28))
sample_data_28_standard = np.random.rand(100, len(standard_features_28))

# Scaling
minmax_scaler_16 = MinMaxScaler().fit(sample_data_16_minmax)
standard_scaler_16 = StandardScaler().fit(sample_data_16_standard)

minmax_scaler_28 = MinMaxScaler().fit(sample_data_28_minmax)
standard_scaler_28 = StandardScaler().fit(sample_data_28_standard)

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
        try:
            prediction_16 = model_16_features.predict(combined_inputs_16)
            st.write(f"Prediction: {prediction_16[0]}")
        except catboost.CatBoostError as e:
            st.error(f"An error occurred during prediction: {e}")
            st.write("Please check if the input features are correct and try again.")

elif option == "28 Features":
    st.subheader("Enter values for 28 features")

    # Input sliders for the 28 features (including Kurtosis_RB)
    inputs_minmax = [st.slider(f"{feature}", 0.0, 1.0, 0.5) for feature in minmax_features_28]
    inputs_standard = [st.slider(f"{feature}", 0.0, 1000.0, 500.0) for feature in standard_features_28]

    # Fixed value for Kurtosis_RB (change this as necessary)
    fixed_kurtosis_rb_value = 0.0  # Change this to your desired constant value

    # Ensure we are appending the fixed value correctly
    # Kurtosis_RB should already be in your input list as it's part of minmax_features_28
    # If you want to exclude it, make sure it's properly adjusted in the lists used for scalers
    inputs_minmax.append(fixed_kurtosis_rb_value)  # Append for minmax input
    inputs_standard.append(fixed_kurtosis_rb_value)  # Append for standard input

    # Debugging: Print input lengths
    st.write("MinMax Inputs:", inputs_minmax)
    st.write("Standard Inputs:", inputs_standard)

    # Check if the lengths match expected counts
    if len(inputs_minmax) != len(minmax_features_28) or len(inputs_standard) != len(standard_features_28):
        st.error("Mismatch in the number of features. Please ensure all features are correctly set.")
    else:
        # Scaling input values
        try:
            # Transform using the scalers
            scaled_minmax = minmax_scaler_28.transform([inputs_minmax[:-1]])  # exclude Kurtosis_RB for minmax
            scaled_standard = standard_scaler_28.transform([inputs_standard[:-1]])  # exclude Kurtosis_RB for standard

            # Concatenate scaled features
            combined_inputs_28 = np.concatenate([scaled_minmax, scaled_standard], axis=1)

            # Predict
            if st.button("Predict"):
                try:
                    prediction_28 = model_28_features.predict(combined_inputs_28)
                    st.write(f"Prediction: {prediction_28[0]}")
                except catboost.CatBoostError as e:
                    st.error(f"An error occurred during prediction: {e}")
                    st.write("Please check if the input features are correct and try again.")
        except ValueError as e:
            st.error(f"Error in scaling: {e}")

            
elif option == "Image Classification":
    st.subheader("Upload an image for classification")
    st.write("Upload an image to classify the pistachio type")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Load the uploaded image
            image = Image.open(uploaded_file)
            # Display the image
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            st.write("")
            st.write("Results")
            
            # Perform classification
            results = image_model.predict(image)
            
            # Get the predicted class
            probs = results[0].probs
            predicted_class_index = probs.top1
            predicted_class_confidence = probs.top1conf
            
            # Retrieve class names from the model's attribute
            class_names = image_model.names
            
            # Get the predicted class name
            predicted_class_name = class_names[predicted_class_index]
            
            st.write(f"Predicted class: {predicted_class_name}")
            st.write(f"Confidence: {predicted_class_confidence:.2f}")
        except Exception as e:
            st.error(f"An error occurred during image classification: {e}")
            st.write("Please check if the uploaded image is valid and try again.")

# Add a section to display model information
st.sidebar.markdown("---")
st.sidebar.subheader("Model Information")
st.sidebar.write(f"16 Features Model: {type(model_16_features).__name__}")
st.sidebar.write(f"28 Features Model: {type(model_28_features).__name__}")
st.sidebar.write(f"Image Model: {type(image_model).__name__}")
