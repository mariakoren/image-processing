import cv2
import numpy as np
from scipy.ndimage import generic_filter

# Wczytanie obrazów w odcieniach szarości
muchaA = cv2.imread('muchaA.png', cv2.IMREAD_GRAYSCALE)
muchaB = cv2.imread('muchaB.png', cv2.IMREAD_GRAYSCALE)
muchaC = cv2.imread('muchaC.png', cv2.IMREAD_GRAYSCALE)


def global_contrast(image):
    mean = np.mean(image)
    std_dev = np.std(image)
    return std_dev

contrast_A = global_contrast(muchaA)
contrast_B = global_contrast(muchaB)
contrast_C = global_contrast(muchaC)

print("Kontrast globalny:")
print("MuchaA:", contrast_A)
print("MuchaB:", contrast_B)
print("MuchaC:", contrast_C)


def local_contrast(image):
    def contrast_function(pixel_values):
        center_pixel = pixel_values[4]
        neighbors = np.delete(pixel_values, 4)
        return np.mean(np.abs(center_pixel - neighbors))

    # Oblicz kontrast lokalny przy użyciu sąsiedztwa 3x3
    local_contrast_map = generic_filter(image, contrast_function, size=(3, 3))
    return np.mean(local_contrast_map)

contrast_local_A = local_contrast(muchaA)
contrast_local_B = local_contrast(muchaB)
contrast_local_C = local_contrast(muchaC)

print("Kontrast lokalny:")
print("MuchaA:", contrast_local_A)
print("MuchaB:", contrast_local_B)
print("MuchaC:", contrast_local_C)
