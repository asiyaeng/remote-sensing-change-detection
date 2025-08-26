import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.colors import LinearSegmentedColormap

def display_change_map(change_map):
    """
    Convert a change map to a colored heatmap for visualization.
    
    Args:
        change_map (numpy.ndarray): Change intensity map normalized to [0, 1]
        
    Returns:
        numpy.ndarray: Colored heatmap as RGB image
    """
    # Create a custom colormap (blue to red, representing low to high change)
    colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]
    cmap = LinearSegmentedColormap.from_list('change_cmap', colors, N=256)
    
    # Apply colormap
    heatmap = cmap(change_map)
    
    # Convert to uint8 for display
    heatmap_rgb = (heatmap[:, :, :3] * 255).astype(np.uint8)
    
    return heatmap_rgb

def create_comparison_figure(before_img, after_img, binary_map):
    """
    Create a side-by-side comparison figure with overlaid change map.
    
    Args:
        before_img (numpy.ndarray): Before image
        after_img (numpy.ndarray): After image
        binary_map (numpy.ndarray): Binary change map
        
    Returns:
        matplotlib.figure.Figure: Figure with the comparison visualization
    """
    # Create figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Display before image
    axes[0].imshow(before_img)
    axes[0].set_title('Before Image')
    axes[0].axis('off')
    
    # Display after image
    axes[1].imshow(after_img)
    axes[1].set_title('After Image')
    axes[1].axis('off')
    
    # Create overlay of binary change map on after image
    overlay = after_img.copy()
    
    # If images are grayscale, convert to RGB for overlay
    if len(overlay.shape) == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
    
    # Create a red mask for changed areas
    mask = np.zeros_like(overlay)
    if len(mask.shape) == 3:  # RGB image
        mask[binary_map == 1] = [255, 0, 0]  # Red for changed areas
    else:  # Grayscale
        mask[binary_map == 1] = 255
    
    # Blend the images
    alpha = 0.5  # Transparency factor
    overlay = cv2.addWeighted(overlay, 1 - alpha, mask, alpha, 0)
    
    # Display overlay
    axes[2].imshow(overlay)
    axes[2].set_title('Changed Areas Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    return fig
