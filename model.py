import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import streamlit as st
import os
import pickle
import joblib
import time

from pathlib import Path

MODEL_PATH = Path("random_forest_mnist.joblib")

class TrainingHistory:
    """Simple class to mimic the history attribute of TensorFlow models"""
    def __init__(self):
        self.history = {
            'accuracy': [],
            'val_accuracy': [],
            'loss': [],
            'val_loss': []
        }

def train_model(n_estimators=100, max_depth=30, save_model=True):
    """
    Train a Random Forest model on the MNIST dataset
    
    Args:
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of the trees
        save_model: Whether to save the trained model to disk
        
    Returns:
        model: Trained model
        history: Training history
        test_accuracy: Accuracy on test set
        test_loss: None (not applicable for Random Forest)
        confusion_matrix: Confusion matrix for test predictions
    """
    # Load MNIST dataset from OpenML
    st.text("Loading MNIST dataset...")
    try:
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    except Exception as e:
        st.error(f"Error fetching MNIST data: {e}")
        # Use a smaller dataset as fallback if OpenML is not available
        X = np.random.rand(7000, 784)  # 7000 samples, 784 features (28x28)
        y = np.random.randint(0, 10, 7000).astype(str)  # 10 classes (0-9)
    
    # Normalize pixel values to [0, 1]
    X = X / 255.0
    
    # Convert labels to integers
    y = y.astype(int)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Further split training data to create a validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    # Create a simple history object to track progress
    history = TrainingHistory()
    
    # Train the model
    st.text("Training Random Forest model...")
    start_time = time.time()
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate on training, validation, and test sets
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, train_preds) * 100
    val_accuracy = accuracy_score(y_val, val_preds) * 100
    test_accuracy = accuracy_score(y_test, test_preds) * 100
    
    # Populate history object
    history.history['accuracy'] = [train_accuracy]
    history.history['val_accuracy'] = [val_accuracy]
    history.history['loss'] = [0]  # Not applicable for Random Forest
    history.history['val_loss'] = [0]  # Not applicable for Random Forest
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, test_preds)
    
    # Save the model if requested
    if save_model:
        joblib.dump(model, 'mnist_model.joblib')
    
    training_time = time.time() - start_time
    st.text(f"Training completed in {training_time:.2f} seconds")
    
    return model, history, test_accuracy, None, cm

def load_or_train_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    else:
        model = train_model()
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        return model
