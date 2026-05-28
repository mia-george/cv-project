import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def remove_background(img, model_asset_path="selfie_segmenter.tflite", background_color=(255, 255, 255)):
    """
    Remove background from image using MediaPipe Selfie Segmentation.
    Keeps the person with specified background color.
    
    Args:
        img: Input image (BGR)
        model_asset_path: Path to selfie segmentation model
        background_color: RGB color for background (default white)
    
    Returns:
        Image with background removed, mask
    """
    try:
        base_options = python.BaseOptions(model_asset_path=model_asset_path)
        options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)
        
        with vision.ImageSegmenter.create_from_options(options) as segmenter:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
            segmentation_result = segmenter.segment(mp_image)
            
            # Get the mask (0 = background, > 0 = foreground)
            category_mask = segmentation_result.category_mask.numpy_view()
            # Ensure mask is 2D
            if category_mask.ndim > 2:
                category_mask = category_mask.squeeze()
            
            mask = (category_mask <= 0.5).astype(np.uint8) * 255
            
            # Apply mask to image
            result = img.copy()
            bg_color_bgr = (background_color[2], background_color[1], background_color[0])
            
            # Create background image
            background = np.full_like(img, bg_color_bgr, dtype=np.uint8)
            
            # Use mask to blend: foreground where mask is 255, background where mask is 0
            mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
            result = (img.astype(np.float32) * mask_3d + background.astype(np.float32) * (1 - mask_3d)).astype(np.uint8)
            
            return result, mask
    
    except Exception as e:
        print(f"Could not remove background: {e}")
        return img, np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255