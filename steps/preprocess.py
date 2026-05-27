import cv2


BILATERAL_PASSES = 3

def to_gray(img):
    # for edge detection
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img

def bilateral_filter(img):
    # bilateral filter smooths textures but preserves edges
    for _ in range(max(1, BILATERAL_PASSES)):
        filtered_img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    
    return filtered_img


def preprocess(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at path: '{img_path}'. Please check that the file path is correct.")

    gray_img = to_gray(img)
    bf_img = bilateral_filter(img)
    gray_bf = cv2.cvtColor(bf_img, cv2.COLOR_BGR2GRAY)

    return img, gray_img, bf_img, gray_bf