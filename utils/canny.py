import cv2
import numpy as np


def canny_edges(rgb_img, low=50, high=150):
    """Canny edge map of an RGB image (H, W, 3) or (3, H, W), returns (H, W) uint8/float."""
    if rgb_img.ndim == 3 and rgb_img.shape[0] in (1, 3) and rgb_img.shape[0] < rgb_img.shape[-1]:
        rgb_img = rgb_img.transpose(1, 2, 0)
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    if gray.max() <= 1.0:
        gray = (gray * 255).astype(np.uint8)
    edge = cv2.Canny(gray, low, high)
    return (edge.astype(np.float32) / 255.0)
