import cv2
import numpy as np

def midrange_filter(image, kernel_size):
    """
    Applies a midrange filter to the input image.

    Parameters:
        image (numpy.ndarray): Grayscale image to process.
        kernel_size (int): Size of the square kernel (e.g., 3 for a 3x3 kernel).

    Returns:
        numpy.ndarray: Image after applying the midrange filter.
    """
    pad_size = kernel_size // 2
    padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)

    filtered_image = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            kernel_region = padded_image[i:i+kernel_size, j:j+kernel_size]
            min_val = np.min(kernel_region)
            max_val = np.max(kernel_region)
            midrange_val = (min_val + max_val) / 2
            filtered_image[i, j] = midrange_val

    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
    return filtered_image

def trimmed_mean_filter(image, kernel_size, k):
    """
    Applies a k-trimmed mean filter to the input image.

    Parameters:
        image (numpy.ndarray): Grayscale image to process.
        kernel_size (int): Size of the square kernel.
        k (int): Number of smallest and largest values to trim.

    Returns:
        numpy.ndarray: Image after applying the trimmed mean filter.
    """
    pad_size = kernel_size // 2
    padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)

    filtered_image = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            kernel_region = padded_image[i:i+kernel_size, j:j+kernel_size].flatten()
            trimmed_region = np.sort(kernel_region)[k:-k]
            trimmed_mean = np.mean(trimmed_region)
            filtered_image[i, j] = trimmed_mean

    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
    return filtered_image

def knn_filter(image, kernel_size, k):
    """
    Applies a k-Nearest Neighbor filter to the input image.

    Parameters:
        image (numpy.ndarray): Grayscale image to process.
        kernel_size (int): Size of the square kernel.
        k (int): Number of nearest neighbors to consider.

    Returns:
        numpy.ndarray: Image after applying the k-NN filter.
    """
    pad_size = kernel_size // 2
    padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)

    filtered_image = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            kernel_region = padded_image[i:i+kernel_size, j:j+kernel_size].flatten()
            center_pixel = image[i, j]
            distances = np.abs(kernel_region - center_pixel)
            nearest_indices = np.argsort(distances)[:k]
            knn_mean = np.mean(kernel_region[nearest_indices])
            filtered_image[i, j] = knn_mean

    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
    return filtered_image

def symmetric_nearest_neighbor_filter(image, kernel_size):
    """
    Applies a Symmetric Nearest Neighbor filter to the input image.

    Parameters:
        image (numpy.ndarray): Grayscale image to process.
        kernel_size (int): Size of the square kernel.

    Returns:
        numpy.ndarray: Image after applying the symmetric nearest neighbor filter.
    """
    pad_size = kernel_size // 2
    padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)

    filtered_image = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            kernel_region = padded_image[i:i+kernel_size, j:j+kernel_size].flatten()
            sorted_values = np.sort(kernel_region)
            symmetric_values = sorted_values[1:-1]  # Remove one smallest and one largest
            symmetric_mean = np.mean(symmetric_values)
            filtered_image[i, j] = symmetric_mean

    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
    return filtered_image

if __name__ == "__main__":
    # Load the image
    input_image = cv2.imread("meduza.png", cv2.IMREAD_GRAYSCALE)

    if input_image is None:
        print("Error: Could not load image.")
        exit()

    # Apply filters
    kernel_size = 3

    midrange_filtered = midrange_filter(input_image, kernel_size=kernel_size)
    trimmed_mean_filtered = trimmed_mean_filter(input_image, kernel_size=kernel_size, k=2)
    knn_filtered = knn_filter(input_image, kernel_size=kernel_size, k=6)
    symmetric_filtered = symmetric_nearest_neighbor_filter(input_image, kernel_size=kernel_size)

    # Save and display results
    cv2.imwrite("meduza-b.png", midrange_filtered)
    cv2.imwrite("meduza-c.png", trimmed_mean_filtered)
    cv2.imwrite("meduza-d.png", knn_filtered)
    cv2.imwrite("meduza-e.png", symmetric_filtered)

    cv2.imshow("Original", input_image)
    cv2.imshow("Midrange Filtered", midrange_filtered)
    cv2.imshow("Trimmed Mean Filtered", trimmed_mean_filtered)
    cv2.imshow("k-NN Filtered", knn_filtered)
    cv2.imshow("Symmetric Filtered", symmetric_filtered)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
