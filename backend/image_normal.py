import cv2
import numpy as np

def normalize_image(image, target_min=0.0, target_max=1.0):
    """idk how to normalize this shit"""

def resize_image(image, size=(256, 256)):
    """
    Resizes the image to the given size.
    
    Parameters:
        image (np.ndarray): Input image array.
        size (tuple): Desired dimensions (width, height).
    
    Returns:
        np.ndarray: Resized image.
    """
    return cv2.resize(image, size)

# Example usage:
if __name__ == "__main__":
    # Load an image using OpenCV (make sure you have an image file available)
    # For example: image = cv2.imread("path_to_your_image.jpg")
    # Here, we'll create a dummy image for demonstration:
    dummy_image = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
    image = cv2.imread("uploads/")  # Replace with your image path
    # Resize the image to 256x256 pixels
    resized_image = resize_image(dummy_image, size=(256, 256))
    
    # Normalize the resized image
    normalized_image = normalize_image(resized_image)
    
    print("Resized image shape:", resized_image.shape)
    print("Normalized image range:", normalized_image.min(), "-", normalized_image.max())