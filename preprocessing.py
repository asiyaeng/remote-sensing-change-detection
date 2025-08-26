import cv2
import numpy as np
from PIL import Image
import io

def preprocess_image(img, apply_gaussian=True, kernel_size=5):
    """
    Preprocess an image for change detection.
    
    Args:
        img (numpy.ndarray): Input image as a NumPy array
        apply_gaussian (bool): Whether to apply Gaussian blur
        kernel_size (int): Size of the Gaussian kernel (must be odd)
    
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Convert to grayscale if RGB
    if len(img.shape) == 3 and img.shape[2] == 3:
        # Keep original for later use
        img_processed = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_processed = img.copy()
    
    # Apply Gaussian blur if requested
    if apply_gaussian:
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        img_processed = cv2.GaussianBlur(img_processed, (kernel_size, kernel_size), 0)
    
    # Normalize to range 0-1 for better algorithm performance
    img_processed = img_processed.astype(np.float32) / 255.0
    
    return img_processed

def resize_image(pil_img, target_size_kb=50):
    """
    Resize a PIL image to approximately the target file size in KB.
    
    Args:
        pil_img (PIL.Image): Input image as a PIL Image object
        target_size_kb (int): Target size in kilobytes
    
    Returns:
        PIL.Image: Resized image
    """
    # Get current size to use as starting point
    width, height = pil_img.size
    
    # Skip if image is already small
    img_byte = io.BytesIO()
    pil_img.save(img_byte, format='JPEG', quality=95)
    current_size = len(img_byte.getvalue()) / 1024
    
    if current_size <= target_size_kb:
        return pil_img
    
    # Calculate target dimensions
    ratio = np.sqrt(target_size_kb / current_size)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    
    # Resize the image
    resized_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
    
    # Check size and adjust if needed
    # This is a simple approach - in production, a binary search would be more efficient
    img_byte = io.BytesIO()
    quality = 95
    resized_img.save(img_byte, format='JPEG', quality=quality)
    new_size = len(img_byte.getvalue()) / 1024
    
    # If still too big, reduce quality
    while new_size > target_size_kb and quality > 30:
        quality -= 5
        img_byte = io.BytesIO()
        resized_img.save(img_byte, format='JPEG', quality=quality)
        new_size = len(img_byte.getvalue()) / 1024
    
    # Return the resized image as a PIL Image
    if quality < 95:
        # If we had to reduce quality, reload from the compressed bytes
        img_byte.seek(0)
        return Image.open(img_byte)
    else:
        return resized_img
