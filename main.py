import sys
import os
import steps.preprocess as preproc
import steps.edge_detector as edge_detector
import argparse
import cv2
import numpy as np

def main():
    # parser = argparse.ArgumentParser(description="Color Block Portrait Generator")
    # parser.add_argument("--image", type=str, required=True, help="Path to the image")
    # args = parser.parse_args()

    # image = preprocess.preprocess(args.image)
    
    gray_img, bf_img = preproc.preprocess("sample_images/boy_portrait.jpeg") # hardcoded for testing
    
    edges = edge_detector.canny_edge_detector(gray_img)
    edge_img = (edges.astype(np.uint8)) * 255

    cv2.imshow("Gray Image", gray_img)
    cv2.imshow("Bilateral Filtered Image", bf_img)
    cv2.imshow("Canny Edges", edge_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()