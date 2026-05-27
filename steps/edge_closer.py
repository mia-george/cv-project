import cv2
import numpy as np


def _find_endpoints(edge_map):
    """Find edge pixels with exactly 1 neighbor (endpoints of edge segments)."""
    binary = edge_map // 255

    # Count 8-connected neighbors for every edge pixel
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0  # don't count the pixel itself

    neighbor_count = cv2.filter2D(binary.astype(np.float32), -1, kernel.astype(np.float32))
    neighbor_count = (neighbor_count * binary).astype(np.uint8)  # only count on edge pixels

    # Endpoint = edge pixel with exactly 1 neighbor
    endpoint_mask = (binary == 1) & (neighbor_count == 1)
    return np.argwhere(endpoint_mask)


def _close_gaps(edge_map, endpoint_coords, max_gap=50000):
    closed = edge_map.copy()
    used = set()

    # Build and sort all pairs by distance, shortest first
    pairs = []
    for i in range(len(endpoint_coords)):
        for j in range(i + 1, len(endpoint_coords)):
            y1, x1 = endpoint_coords[i]
            y2, x2 = endpoint_coords[j]
            dist = np.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2)
            pairs.append((dist, i, j))
    pairs.sort()

    for dist, i, j in pairs:
        if dist > max_gap:
            break
        if i in used or j in used:
            continue
        y1, x1 = endpoint_coords[i]
        y2, x2 = endpoint_coords[j]
        cv2.line(closed, (x1, y1), (x2, y2), 255, thickness=1)
        used.add(i)
        used.add(j)

    return closed


def close_edges(edges, max_gap=80):
    closed = edges.copy()
    h, w = closed.shape

    # Add a 1-pixel border to edges to seal open boundaries against the borders
    closed[0, :] = 255
    closed[-1, :] = 255
    closed[:, 0] = 255
    closed[:, -1] = 255

    # First pass of endpoint stitching
    endpoints = _find_endpoints(closed)
    closed = _close_gaps(closed, endpoints, max_gap)

    # Iterative passes of endpoint stitching until no progress is made
    remaining = _find_endpoints(closed)
    while len(remaining) > 0:
        prev_count = len(remaining)
        closed = _close_gaps(closed, remaining, max_gap)
        remaining = _find_endpoints(closed)
        if len(remaining) >= prev_count:
            break

    # Connect any loose endpoints close to the image borders directly to the borders
    endpoints = _find_endpoints(closed)
    for y, x in endpoints:
        if x < max_gap:
            cv2.line(closed, (x, y), (0, y), 255, thickness=1)
        if (w - 1 - x) < max_gap:
            cv2.line(closed, (x, y), (w - 1, y), 255, thickness=1)
        if y < max_gap:
            cv2.line(closed, (x, y), (x, 0), 255, thickness=1)
        if (h - 1 - y) < max_gap:
            cv2.line(closed, (x, y), (x, h - 1), 255, thickness=1)

    # Make sure the 1-pixel border is fully intact
    closed[0, :] = 255
    closed[-1, :] = 255
    closed[:, 0] = 255
    closed[:, -1] = 255

    return closed
