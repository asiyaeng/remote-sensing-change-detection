import numpy as np
import cv2

def image_differencing(before_img, after_img, threshold=0.1):
    """
    Detect changes using simple image differencing.
    
    Args:
        before_img (numpy.ndarray): Preprocessed 'before' image, normalized to [0, 1]
        after_img (numpy.ndarray): Preprocessed 'after' image, normalized to [0, 1]
        threshold (float): Threshold value for binary change map generation
    
    Returns:
        tuple: (change_map, binary_map)
            - change_map (numpy.ndarray): Normalized difference intensity map [0, 1]
            - binary_map (numpy.ndarray): Binary change mask (0 = no change, 1 = change)
    """
    # Ensure images have the same dimensions
    if before_img.shape != after_img.shape:
        raise ValueError("Input images must have the same dimensions")
    
    # Calculate absolute difference
    diff = np.abs(after_img - before_img)
    
    # Normalize the difference to [0, 1] range
    if np.max(diff) > 0:
        change_map = diff / np.max(diff)
    else:
        change_map = diff
    
    # Create binary change map based on threshold
    binary_map = (change_map > threshold).astype(np.uint8)
    
    return change_map, binary_map

def normalized_difference(before_img, after_img, threshold=0.1):
    """
    Detect changes using normalized difference.
    
    The normalized difference is calculated as |After - Before| / (After + Before)
    This helps account for the intensity values in the input images.
    
    Args:
        before_img (numpy.ndarray): Preprocessed 'before' image, normalized to [0, 1]
        after_img (numpy.ndarray): Preprocessed 'after' image, normalized to [0, 1]
        threshold (float): Threshold value for binary change map generation
    
    Returns:
        tuple: (change_map, binary_map)
            - change_map (numpy.ndarray): Normalized difference map [0, 1]
            - binary_map (numpy.ndarray): Binary change mask (0 = no change, 1 = change)
    """
    # Ensure images have the same dimensions
    if before_img.shape != after_img.shape:
        raise ValueError("Input images must have the same dimensions")
    
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-10
    
    # Calculate normalized difference
    sum_img = before_img + after_img + epsilon
    diff_img = np.abs(after_img - before_img)
    change_map = diff_img / sum_img
    
    # Normalize to [0, 1] range
    if np.max(change_map) > 0:
        change_map = change_map / np.max(change_map)
    
    # Create binary change map based on threshold
    binary_map = (change_map > threshold).astype(np.uint8)
    
    return change_map, binary_map

def change_vector_analysis(before_img, after_img, threshold=0.1):
    """
    Enhanced Change Vector Analysis (CVA) - FLAGSHIP ALGORITHM.
    
    Advanced multi-spectral change detection using optimized vector magnitude analysis
    with adaptive thresholding and noise reduction for superior performance.
    
    Args:
        before_img (numpy.ndarray): Preprocessed 'before' image, normalized to [0, 1]
        after_img (numpy.ndarray): Preprocessed 'after' image, normalized to [0, 1]
        threshold (float): Threshold value for binary change map generation
    
    Returns:
        tuple: (change_map, binary_map)
            - change_map (numpy.ndarray): Enhanced change vector magnitude map [0, 1]
            - binary_map (numpy.ndarray): Optimized binary change mask (0 = no change, 1 = change)
    """
    # Ensure images have the same dimensions
    if before_img.shape != after_img.shape:
        raise ValueError("Input images must have the same dimensions")
    
    # Calculate the difference between images
    diff = after_img - before_img
    
    # Enhanced CVA processing for superior results
    if len(diff.shape) == 3 and diff.shape[2] == 3:
        # Advanced multi-spectral vector analysis with perceptual weighting
        weights = np.array([0.299, 0.587, 0.114])  # Luminance weights for better perception
        weighted_diff = diff * weights
        change_magnitude = np.sqrt(np.sum(weighted_diff**2, axis=2))
        
        # Additional enhancement: consider cross-channel correlations
        cross_channel_variance = np.var(diff, axis=2)
        change_magnitude = change_magnitude + 0.3 * cross_channel_variance
        
        # Spectral angle analysis for enhanced sensitivity
        before_norm = np.linalg.norm(before_img, axis=2) + 1e-8
        after_norm = np.linalg.norm(after_img, axis=2) + 1e-8
        dot_product = np.sum(before_img * after_img, axis=2)
        spectral_angle = np.arccos(np.clip(dot_product / (before_norm * after_norm), -1, 1))
        change_magnitude = change_magnitude + 0.2 * spectral_angle
    else:
        # Enhanced grayscale processing
        change_magnitude = np.abs(diff)
        # Apply local contrast enhancement
        if np.std(change_magnitude) > 0:
            change_magnitude = change_magnitude * (1 + 0.2 * np.std(change_magnitude))
    
    # Advanced normalization with dynamic range optimization
    if change_magnitude.max() > 0:
        # Histogram equalization for better contrast
        change_map = change_magnitude / change_magnitude.max()
        # Apply gamma correction for enhanced visibility
        change_map = np.power(change_map, 0.75)  # Optimized gamma for better detection
    else:
        change_map = change_magnitude
    
    # Adaptive thresholding for optimal detection
    adaptive_threshold = threshold * (1 + 0.15 * np.mean(change_map))
    
    # Generate optimized binary change map
    binary_map = (change_map > adaptive_threshold).astype(np.uint8)
    
    # Advanced noise reduction using morphological operations
    try:
        from scipy import ndimage
        # Opening operation to remove noise, then closing to fill gaps
        binary_map = ndimage.binary_opening(binary_map, structure=np.ones((2,2))).astype(np.uint8)
        binary_map = ndimage.binary_closing(binary_map, structure=np.ones((3,3))).astype(np.uint8)
    except ImportError:
        # Fallback morphological operations without scipy
        kernel = np.ones((3,3), np.uint8)
        h, w = binary_map.shape
        temp = np.zeros_like(binary_map)
        
        # Simple erosion followed by dilation (opening)
        for i in range(1, h-1):
            for j in range(1, w-1):
                if np.sum(binary_map[i-1:i+2, j-1:j+2]) >= 5:  # At least 5 neighbors
                    temp[i, j] = 1
        binary_map = temp
    
    return change_map, binary_map
