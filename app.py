import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model import train_model, load_or_train_model
from utils import preprocess_image
import cv2

# Set page configuration
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✏️",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'model' not in st.session_state:
    st.session_state.model = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = None
if 'test_accuracy' not in st.session_state:
    st.session_state.test_accuracy = None
if 'test_loss' not in st.session_state:
    st.session_state.test_loss = None
if 'confusion_matrix' not in st.session_state:
    st.session_state.confusion_matrix = None

# Title and app description
st.title("✏️ Handwritten Digit Recognition")
st.markdown("""
This application uses a Random Forest model trained on the MNIST dataset 
to recognize handwritten digits. You can either use the pre-trained model or train a new model.
""")

# Sidebar for model options
st.sidebar.title("Model Options")

# Train model option
if st.sidebar.button("Train New Model"):
    with st.spinner("Training model... This may take a few minutes."):
        model, history, test_accuracy, test_loss, confusion_matrix = train_model()
        st.session_state.model = model
        st.session_state.training_history = history
        st.session_state.test_accuracy = test_accuracy
        st.session_state.test_loss = test_loss
        st.session_state.confusion_matrix = confusion_matrix
    st.sidebar.success("Model trained successfully!")

# Load pre-trained model
if st.session_state.model is None:
    st.sidebar.markdown("**Note:** Training a new model takes time. For quick testing, the app will use a pre-trained model if available.")
    with st.spinner("Loading or training model..."):
        # Load a pre-trained model when the app starts for better user experience
        try:
            model, history, test_accuracy, test_loss, confusion_matrix = load_or_train_model()
            st.session_state.model = model
            st.session_state.training_history = history
            st.session_state.test_accuracy = test_accuracy
            st.session_state.test_loss = test_loss
            st.session_state.confusion_matrix = confusion_matrix
            
            if history is None:
                st.sidebar.success("Pre-trained model loaded!")
            else:
                st.sidebar.success("Model trained and loaded!")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {e}")

# Main content section
st.markdown("## Upload a Handwritten Digit Image")
st.markdown("""
Please upload an image of a handwritten digit (0-9). For best results:
- Use a black digit on a white background
- Ensure the digit is centered in the image
- The image should be clear and well-lit
- Supported formats: JPG, JPEG, PNG
""")

# Image upload section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the uploaded image
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Process the image and make predictions
    with col2:
        st.markdown("### Processed Image")
        
        try:
            # Convert to numpy array and preprocess
            image_array = np.array(image)
            processed_image, display_image = preprocess_image(image_array)
            
            # Display processed image
            st.image(display_image, caption="Processed Image", use_column_width=True)
            
            # Make prediction
            if st.session_state.model is not None:
                # With Random Forest, we get class probabilities
                prediction_proba = st.session_state.model.predict_proba(processed_image)
                predicted_class = st.session_state.model.predict(processed_image)[0]
                confidence = prediction_proba[0][predicted_class] * 100
                
                # Display prediction result
                st.markdown(f"### Prediction: {predicted_class}")
                st.markdown(f"**Confidence: {confidence:.2f}%**")
                
                # Display confidence bars for all digits
                st.markdown("### Confidence Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(range(10), prediction_proba[0] * 100)
                
                # Highlight the predicted digit
                bars[predicted_class].set_color('red')
                
                ax.set_xticks(range(10))
                ax.set_xticklabels(range(10))
                ax.set_ylim(0, 100)
                ax.set_xlabel('Digit')
                ax.set_ylabel('Confidence (%)')
                ax.set_title('Prediction Confidence for Each Digit')
                
                for i, v in enumerate(prediction_proba[0] * 100):
                    ax.text(i, v + 1, f"{v:.1f}%", ha='center', va='bottom', fontsize=8)
                
                st.pyplot(fig)
            else:
                st.error("Model not loaded. Please train a new model or wait for the pre-trained model to load.")
                
        except Exception as e:
            st.error(f"Error processing image: {e}")
            st.markdown("Please ensure you've uploaded a valid image file.")

# Model performance section
if st.session_state.model is not None and st.session_state.training_history is not None:
    st.markdown("## Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Plot training & validation accuracy
        st.markdown("### Training and Validation Accuracy")
        fig, ax = plt.subplots(figsize=(10, 6))
        # For Random Forest, we only have single point accuracy, not epoch-by-epoch
        if len(st.session_state.training_history.history['accuracy']) == 1:
            # Single point plot for Random Forest
            ax.bar(['Training', 'Validation'], 
                   [st.session_state.training_history.history['accuracy'][0], 
                    st.session_state.training_history.history['val_accuracy'][0]],
                   color=['blue', 'orange'])
            ax.set_ylim(0, 100)
            ax.set_ylabel('Accuracy (%)')
            ax.set_title('Model Accuracy')
            
            # Add text labels
            ax.text(0, st.session_state.training_history.history['accuracy'][0] + 1, 
                    f"{st.session_state.training_history.history['accuracy'][0]:.2f}%", 
                    ha='center')
            ax.text(1, st.session_state.training_history.history['val_accuracy'][0] + 1, 
                    f"{st.session_state.training_history.history['val_accuracy'][0]:.2f}%", 
                    ha='center')
        else:
            # Multi-epoch plot for other models
            ax.plot(st.session_state.training_history.history['accuracy'], label='Training Accuracy')
            ax.plot(st.session_state.training_history.history['val_accuracy'], label='Validation Accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Model Accuracy')
            ax.legend()
        st.pyplot(fig)
        
        # Display test accuracy
        if st.session_state.test_accuracy is not None:
            st.markdown(f"**Test Accuracy: {st.session_state.test_accuracy:.2f}%**")
    
    with col2:
        # For Random Forest, we don't have loss values, so show feature importance instead
        if st.session_state.model is not None and hasattr(st.session_state.model, 'feature_importances_'):
            st.markdown("### Feature Importance")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get the top 20 most important features
            feature_importances = st.session_state.model.feature_importances_
            top_n = 20
            indices = np.argsort(feature_importances)[-top_n:]
            
            ax.barh(range(top_n), feature_importances[indices])
            ax.set_yticks(range(top_n))
            ax.set_yticklabels([f'Feature {i}' for i in indices])
            ax.set_xlabel('Relative Importance')
            ax.set_title('Top 20 Feature Importance')
            
            st.pyplot(fig)
        else:
            # Plot training & validation loss if available
            st.markdown("### Training and Validation Loss")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(st.session_state.training_history.history['loss'], label='Training Loss')
            ax.plot(st.session_state.training_history.history['val_loss'], label='Validation Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Model Loss')
            ax.legend()
            st.pyplot(fig)
            
            # Display test loss
            if st.session_state.test_loss is not None:
                st.markdown(f"**Test Loss: {st.session_state.test_loss:.4f}**")
    
    # Confusion Matrix
    if st.session_state.confusion_matrix is not None:
        st.markdown("### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.matshow(st.session_state.confusion_matrix, cmap='Blues')
        fig.colorbar(cax)
        
        # Set labels for axes
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        
        # Add text annotations
        for i in range(10):
            for j in range(10):
                ax.text(j, i, str(st.session_state.confusion_matrix[i, j]), 
                        ha='center', va='center', 
                        color='white' if st.session_state.confusion_matrix[i, j] > st.session_state.confusion_matrix.max() / 2 else 'black')
        
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticklabels(range(10))
        ax.set_yticklabels(range(10))
        
        st.pyplot(fig)

# Instructions and about section
with st.expander("About this Application"):
    st.markdown("""
    ### Handwritten Digit Recognition with Machine Learning
    
    This application uses a Random Forest model trained on the MNIST dataset to recognize handwritten digits.
    
    **Technologies Used:**
    - Scikit-learn for the machine learning model
    - Streamlit for the web interface
    - OpenCV and PIL for image processing
    - NumPy for numerical operations
    - Matplotlib for visualization
    
    **How it Works:**
    1. The model is trained on the MNIST dataset of 70,000 handwritten digits
    2. When you upload an image, it is preprocessed to match the format of the MNIST dataset
    3. The model then predicts which digit is in the image
    4. The prediction and confidence scores are displayed
    
    **Model Architecture:**
    - Random Forest classifier with multiple decision trees
    - Each tree in the forest is trained on a random subset of the data
    - The final prediction is made by aggregating the predictions of all trees
    - This approach is highly effective for digit recognition tasks
    """)
