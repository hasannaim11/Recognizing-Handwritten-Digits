import numpy as np
import cv2
from PIL import Image, ImageOps
import io

def preprocess_image(image_array):
    """
    Preprocess image for prediction with the MNIST model
    
    Args:
        image_array: Numpy array of the input image
    
    Returns:
        processed_image: Processed image ready for model prediction
        display_image: Image for display purposes
    """
    # Check if image is grayscale or has color channels
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        # Convert RGB to grayscale
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        # Already grayscale
        gray = image_array
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply thresholding to create a binary image
    # We invert the image (white digit on black background to black digit on white background)
    _, binary = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours to identify the digit
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours found, return the original binary image
    if not contours:
        # Resize to 28x28 (MNIST format)
        resized = cv2.resize(binary, (28, 28))
        # Normalize to [0, 1]
        normalized = resized / 255.0
        # Reshape for scikit-learn model (flatten to 1D array)
        processed = normalized.reshape(1, 784)
        return processed, binary
    
    # Find the largest contour (assumes this is the digit)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding box for the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add padding around the digit
    padding = int(max(w, h) * 0.2)
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(binary.shape[1], x + w + padding)
    y_end = min(binary.shape[0], y + h + padding)
    
    # Extract the digit with padding
    digit = binary[y_start:y_end, x_start:x_end]
    
    # Create a blank square image with the larger dimension of width or height
    max_dim = max(digit.shape[0], digit.shape[1])
    square_img = np.zeros((max_dim, max_dim), dtype=np.uint8)
    
    # Center the digit in the square image
    y_offset = (max_dim - digit.shape[0]) // 2
    x_offset = (max_dim - digit.shape[1]) // 2
    square_img[y_offset:y_offset + digit.shape[0], x_offset:x_offset + digit.shape[1]] = digit
    
    # Resize to 20x20 and center in 28x28 (MNIST standard)
    resized_digit = cv2.resize(square_img, (20, 20))
    mnist_image = np.zeros((28, 28), dtype=np.uint8)
    mnist_image[4:24, 4:24] = resized_digit
    
    # Create a display image (upsized for better visibility)
    display_image = cv2.resize(mnist_image, (140, 140), interpolation=cv2.INTER_NEAREST)
    
    # Normalize to [0, 1] for the model
    normalized = mnist_image / 255.0
    
    # Flatten for scikit-learn model (from 28x28 to 784x1)
    processed = normalized.reshape(1, 784)
    
    return processed, display_image
