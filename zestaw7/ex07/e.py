import cv2
import numpy as np
from skimage import filters

def progowanie_z_histereza(image):
    # Zamiana obrazu na skale szarości, jeśli jest kolorowy
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Obliczenie progu T2 metodą Otsu
    T2 = filters.threshold_otsu(gray_image)

    # T1 = 1/2 * T2
    T1 = T2 / 2

    # Zastosowanie progowania z histerezą
    result = np.zeros_like(gray_image)

    # Wartości powyżej T2 ustawiamy na 255 (czarne na białym obrazie)
    result[gray_image > T2] = 255

    # Wartości pomiędzy T1 a T2 ustawiamy na 255 tylko, jeśli mają sąsiedztwo z wartością > T2
    result[(gray_image > T1) & (gray_image <= T2)] = 255
    print(f"T1: {T1}")
    print(f"T2: {T2}")


    return result

# Wczytanie obrazu
image = cv2.imread('zyczenia-d.png', cv2.IMREAD_GRAYSCALE)

# Wykonanie progowania z histerezą
result_image = progowanie_z_histereza(image)

# Zapisanie wyniku
cv2.imwrite('zyczenia-e.png', result_image)
