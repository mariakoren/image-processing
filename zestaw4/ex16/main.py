import numpy as np
import cv2
from scipy.signal import convolve2d

# Funkcja do implementacji algorytmu Van Citterta
def van_cittert_deconvolution(blurred_image, kernel, iterations):
    # Zainicjalizuj obraz początkowy jako obraz rozmyty
    restored_image = blurred_image.astype(np.float32)

    # Tworzymy identyczny kernel do przywracania
    kernel = kernel.astype(np.float32)
    
    for _ in range(iterations):
        # Rekonstrukcja: r(n+1) = r(n) + (g - h * r(n))
        convolved = convolve2d(restored_image, kernel, mode='same', boundary='symm')
        error = blurred_image - convolved
        restored_image += error

    # Normalizacja obrazu do przedziału [0, 255]
    restored_image = np.clip(restored_image, 0, 255).astype(np.uint8)
    return restored_image

input_image_path = "jesien_filtered.png"
blurred_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

if blurred_image is None:
    raise FileNotFoundError("Nie znaleziono obrazu wejściowego.")

# Definicja filtru dolnoprzepustowego (współczynniki Newtona)
kernel = (1 / 256) * np.array([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1]
], dtype=np.float32)

# Liczba iteracji do zapisania
iterations_to_save = [2, 5, 15]

# Dekonwolucja i zapisywanie obrazów
for iterations in iterations_to_save:
    restored_image = van_cittert_deconvolution(blurred_image, kernel, iterations)
    output_path = f"{iterations}.png"
    cv2.imwrite(output_path, restored_image)
    print(f"Zapisano obraz po {iterations} iteracjach: {output_path}")