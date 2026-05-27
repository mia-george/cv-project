import cv2
import numpy as np


def segment_regions(closed_edges, min_area=80):
    """Segment image into regions using connected components on the inverse edge map"""

    # Invert regions are where there are no edges
    open_pixels = np.uint8(closed_edges == 0) * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        open_pixels, connectivity=4
    )

    # Filter out small regions
    for region_id in range(1, num_labels):
        if stats[region_id, cv2.CC_STAT_AREA] < min_area:
            labels[labels == region_id] = 0

    return labels, num_labels
