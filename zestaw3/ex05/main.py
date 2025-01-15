import cv2
import numpy as np
import matplotlib.pyplot as plt

def hyperbolize_histogram(image_path, output_path, parameter):
    """
    Perform histogram hyperbolization on an image and save the result.

    :param image_path: Path to the input image.
    :param output_path: Path to save the output image.
    :param parameter: Parameter for hyperbolization (e.g., -1/3).
    """
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} could not be loaded.")

    # Normalize the image to the range [0, 1]
    normalized_image = image / 255.0

    # Apply hyperbolization formula
    hyperbolized_image = np.power(normalized_image, parameter)

    # Scale back to the range [0, 255]
    result_image = np.clip(hyperbolized_image * 255, 0, 255).astype(np.uint8)

    # Save the resulting image
    cv2.imwrite(output_path, result_image)

    # Display the input and output images for comparison
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Hyperbolized Image")
    plt.imshow(result_image, cmap='gray')
    plt.axis('off')

    plt.show()

# Example usage
image_path = "czaszka.png"  # Replace with the path to your image
output_path = "czaszka-b.png"
parameter = -0.33

hyperbolize_histogram(image_path, output_path, parameter)