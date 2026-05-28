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
    # output = quantized.copy()
    # output = np.ones_like(quantized) * 255

    for region_id in range(1, num_labels):
        mask = (labels == region_id)
        if not mask.any():
            continue
        region_color = np.median(quantized[mask], axis=0).astype(np.uint8)
        output[mask] = region_color

    return output


# def overlay_edges(output, closed_edges, edge_color=(15, 15, 15)):
#     """Overlay thinned edge lines on the colored output"""
#     # Thin the edges slightly with erosion
#     thin_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
#     thin_edges = cv2.erode(closed_edges, thin_kernel, iterations=1)

#     result = output.copy()
#     result[thin_edges == 255] = edge_color
#     return result

def overlay_edges(output, closed_edges):
    """Fill edge pixels with the nearest neighbor color from colored regions."""
    result = output.copy()
    
    # Create edge mask
    edge_mask = closed_edges == 255
    
    if not edge_mask.any():
        return result
    
    # For each edge pixel, find the nearest colored pixel
    edge_coords = np.where(edge_mask)
    
    for y, x in zip(edge_coords[0], edge_coords[1]):
        # Search in expanding square neighborhoods
        found = False
        for radius in range(1, max(result.shape)):
            y_min, y_max = max(0, y - radius), min(result.shape[0], y + radius + 1)
            x_min, x_max = max(0, x - radius), min(result.shape[1], x + radius + 1)
            
            # Get neighborhood
            neighborhood = result[y_min:y_max, x_min:x_max]
            non_edge_mask = ~edge_mask[y_min:y_max, x_min:x_max]
            
            # Find colored pixels in neighborhood
            colored_pixels = neighborhood[non_edge_mask]
            if len(colored_pixels) > 0:
                # Use median color from neighborhood
                result[y, x] = np.median(colored_pixels, axis=0).astype(np.uint8)
                found = True
                break
        
        if not found:
            # Fallback: use output color if no neighbor found
            result[y, x] = output[y, x]
    
    return result
