import cv2
import numpy as np


def quantize_colors(img, k=12):
    """Quantize image colors using K-Means clustering."""
    pixels = img.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, kmeans_labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)
    quantized = centers[kmeans_labels.flatten()].reshape(img.shape)
    return quantized


def colorize_regions(img, labels, num_labels, quantized):   
    """Assigns each segmented region its median quantized color to produce flat color blocks."""
    output = np.zeros_like(img)

    for region_id in range(1, num_labels):
        mask = (labels == region_id)
        if not mask.any():
            continue
        region_color = np.median(quantized[mask], axis=0).astype(np.uint8)
        output[mask] = region_color

    return output


def overlay_edges(output, closed_edges, edge_color=(15, 15, 15)):
    """Overlay thinned edge lines on the colored output"""
    # Thin the edges slightly with erosion
    thin_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thin_edges = cv2.erode(closed_edges, thin_kernel, iterations=1)

    result = output.copy()
    result[thin_edges == 255] = edge_color
    return result
