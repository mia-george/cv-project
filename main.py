import sys
import os
import argparse
import cv2
import steps.preprocess as preproc
import steps.edge_detector as edge_detector
import steps.edge_closer as edge_closer
import steps.region_segmentor as region_segmentor
import steps.color_quantizer as color_quantizer


def main():
    parser = argparse.ArgumentParser(description="Color Block Portrait Generator")
    parser.add_argument("--image", type=str, default='sample_images/portrait1.jpg', help="Path to the input image")
    parser.add_argument("--colors", type=int, default=6, help="Number of colors for quantization")
    args = parser.parse_args()

    image = args.image
    colors = args.colors

    # Step 1: Preprocess image 
    print("Step 1: Preprocessing")
    img, gray_img, bf_img, bf_gray = preproc.preprocess(image)

    # Step 2: Color Quantization - k-means on smoothed bf_img to avoid dither noise
    print("Step 2: Quantizing colors")
    quantized = color_quantizer.quantize_colors(bf_img, k=colors)

    # Step 3: Edge Detection - structural canny + k-means color boundaries
    print("Step 3: Detecting edges and color boundaries")
    structural_edges = edge_detector.canny_edge_detector(bf_gray, percentageOfNonEdge=0.85)
    
    # Extract clean boundaries from the flat quantized color blocks
    quantized_gray = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)
    color_edges = cv2.Canny(quantized_gray, 10, 100)
    
    # Combine structural outlines with color-delineated boundaries 
    combined_edges = cv2.bitwise_or(structural_edges, color_edges)

    # Step 4: Close Edge Gaps - iterative endpoint stitching + border sealing
    print("Step 4: Closing edge gaps")
    closed = edge_closer.close_edges(combined_edges)

    # Step 5: Region Segmentation - connected components
    print("Step 5: Segmenting regions")
    labels, num_labels = region_segmentor.segment_regions(closed, min_area=200)

    # Step 6: Colorize Regions - median quantized color per region
    print("Step 6: Colorizing regions")
    output = color_quantizer.colorize_regions(img, labels, num_labels, quantized)

    # Step 7: Overlay Edges
    print("Step 7: Overlaying edges")
    result = color_quantizer.overlay_edges(output, closed)

    # Save and display
    image_name = os.path.splitext(os.path.basename(image))[0]
    output_path = f"outputs/{image_name}_colorblock.png"
    cv2.imwrite(output_path, result)
    print(f"Output saved to {output_path}")

    cv2.imshow("Color Block Portrait", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()