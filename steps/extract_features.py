import matplotlib.pyplot as plt
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request

# Download the model file (only needed once)
# urllib.request.urlretrieve(
#     "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
#     "face_landmarker.task"
# )

# Feature landmark indices for eyes and lips
FACEMESH_LEFT_EYE = frozenset([
    (263,249),(249,390),(390,373),(373,374),(374,380),(380,381),(381,382),
    (382,362),(362,398),(398,384),(384,385),(385,386),(386,387),(387,388),
    (388,466),(466,263)
])
FACEMESH_RIGHT_EYE = frozenset([
    (33,7),(7,163),(163,144),(144,145),(145,153),(153,154),(154,155),
    (155,133),(133,173),(173,157),(157,158),(158,159),(159,160),(160,161),
    (161,246),(246,33)
])
# Lips outer contour (outer perimeter)
FACEMESH_LIPS_OUTER = frozenset([
    (61,146),(146,91),(91,181),(181,84),(84,17),(17,314),(314,405),(405,321),
    (321,375),(375,291),(291,409),(409,270),(270,269),(269,267),(267,0),
    (0,37),(37,39),(39,40),(40,185),(185,61)
])
# Lips inner contour (mouth interior)
FACEMESH_LIPS_INNER = frozenset([
    (78,95),(95,88),(88,178),(178,87),(87,14),(14,317),(317,402),(402,318),
    (318,324),(324,308),(308,415),(415,310),(310,311),(311,312),(312,13),
    (13,82),(82,81),(81,80),(80,191),(191,78)
])

def order_points_from_connections(connections):
    """Walk the edge graph to produce an ordered list of landmark indices."""
    adj = {}
    for a, b in connections:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    start = next(iter(adj))
    ordered = [start]
    visited = {start}
    current = start
    while True:
        neighbors = [n for n in adj[current] if n not in visited]
        if not neighbors:
            break
        current = neighbors[0]
        ordered.append(current)
        visited.add(current)
    return ordered

def draw_filled_feature(canvas, landmarks, connections, h, w):
    """Draw filled polygon for a feature on the canvas."""
    ordered_indices = order_points_from_connections(connections)
    pts = []
    for i in ordered_indices:
        lm = landmarks[i]
        pts.append((int(lm.x * w), int(lm.y * h)))
    pts = np.array(pts, dtype=np.int32)
    cv2.fillPoly(canvas, [pts], 255)

def create_feature_mask(img):
    """Create a binary mask for eyes and lips using MediaPipe Face Landmarker."""
    feature_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    h, w = img.shape[:2]

    try:
        base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
        options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1, min_face_detection_confidence=0.5)
        
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            result = landmarker.detect(mp_image)
            
            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                draw_filled_feature(feature_mask, landmarks, FACEMESH_LEFT_EYE, h, w)
                draw_filled_feature(feature_mask, landmarks, FACEMESH_RIGHT_EYE, h, w)
                # Draw both outer and inner lip contours for complete outline
                draw_filled_feature(feature_mask, landmarks, FACEMESH_LIPS_OUTER, h, w)
                draw_filled_feature(feature_mask, landmarks, FACEMESH_LIPS_INNER, h, w)
                print(f"Successfully extracted eye and lips masks")
    except Exception as e:
        print(f"Could not extract facial landmarks (continuing without selective filtering): {e}")
        feature_mask = np.zeros_like(img[:, :, 0])

    return feature_mask