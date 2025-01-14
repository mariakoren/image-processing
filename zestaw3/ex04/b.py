import numpy as np
import cv2

def otsu_three_class_thresholding(image, delta=2):
    """
    Perform three-class iterative thresholding using Otsu's method with the condition \u2206 < 2.

    :param image: Input grayscale image.
    :param delta: Convergence criterion for threshold difference.
    :return: Binary images for three classes.
    """
    if len(image.shape) != 2:
        raise ValueError("Input image must be grayscale.")

    # Compute the histogram of the image
    hist, bin_edges = np.histogram(image.ravel(), bins=256, range=(0, 256))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Initial thresholds (t1 and t2)
    t1, t2 = np.percentile(image, [33, 66])

    prev_t1, prev_t2 = 0, 0

    while abs(t1 - prev_t1) >= delta or abs(t2 - prev_t2) >= delta:
        prev_t1, prev_t2 = t1, t2

        # Classify pixels based on current thresholds
        class1 = hist[:int(t1)]
        class2 = hist[int(t1):int(t2)]
        class3 = hist[int(t2):]

        # Calculate means for each class
        mean1 = np.dot(class1, bin_centers[:int(t1)]) / (np.sum(class1) + 1e-8)
        mean2 = np.dot(class2, bin_centers[int(t1):int(t2)]) / (np.sum(class2) + 1e-8)
        mean3 = np.dot(class3, bin_centers[int(t2):]) / (np.sum(class3) + 1e-8)

        # Update thresholds based on class means
        t1 = (mean1 + mean2) / 2
        t2 = (mean2 + mean3) / 2

    # Generate binary masks for the three classes
    class1_mask = (image <= t1).astype(np.uint8)
    class2_mask = ((image > t1) & (image <= t2)).astype(np.uint8)
    class3_mask = (image > t2).astype(np.uint8)

    return class1_mask, class2_mask, class3_mask

# Example usage
if __name__ == "__main__":
    # Load a grayscale image
    image_path = "roze.png"  # Replace with your image path
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Perform three-class thresholding
    class1, class2, class3 = otsu_three_class_thresholding(image)

    # Save the results
    cv2.imwrite("class1_output.png", class1 * 255)  # Save as a binary image (0 or 255)
    cv2.imwrite("class2_output.png", class2 * 255)
    cv2.imwrite("class3_output.png", class3 * 255)

    # Display the results
    cv2.imshow("Class 1", class1 * 255)
    cv2.imshow("Class 2", class2 * 255)
    cv2.imshow("Class 3", class3 * 255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
