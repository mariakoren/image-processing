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
    # Padding the image to handle border pixels
    pad_size = kernel_size // 2
    padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)

    # Output image
    filtered_image = np.zeros_like(image, dtype=np.float32)

    # Iterate over each pixel in the image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract the kernel region
            kernel_region = padded_image[i:i+kernel_size, j:j+kernel_size]

            # Calculate the midrange value
            min_val = np.min(kernel_region)
            max_val = np.max(kernel_region)
            midrange_val = (min_val + max_val) / 2

            # Assign the result to the output image
            filtered_image[i, j] = midrange_val

    # Normalize to 8-bit range if needed
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

    return filtered_image

# Example usage
if __name__ == "__main__":
    # Load a grayscale image
    input_image = cv2.imread("meduza.png", cv2.IMREAD_GRAYSCALE)
    
    if input_image is None:
        print("Error: Could not load image.")
        exit()

    # Apply the midrange filter
    kernel_size = 3  # Define the size of the kernel
    output_image = midrange_filter(input_image, kernel_size)

    # Save and display the result
    cv2.imwrite("meduza-b.png", output_image)
    cv2.imshow("Filtered Image", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
