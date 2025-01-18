import numpy as np
import cv2
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

def compute_second_derivatives(image):
    # Definicja masek pierwszych pochodnych
    hx = np.array([1, 0, -1]).reshape(1, 3)  # Pionowa maska (h_x)
    hy = np.array([[1], [0], [-1]])          # Pozioma maska (h_y)

    # Obliczanie masek drugich pochodnych
    hxx = convolve2d(hx, hx, mode='full')
    hyy = convolve2d(hy, hy, mode='full')
    hxy = convolve2d(hx, hy, mode='full')

    # Obliczanie drugich pochodnych kierunkowych
    gxx = convolve2d(image, hxx, mode='same', boundary='symm')
    gyy = convolve2d(image, hyy, mode='same', boundary='symm')
    gxy = convolve2d(image, hxy, mode='same', boundary='symm')

    return gxx, gyy, gxy

def save_image(output_path, image):
    # Normalizacja obrazu do zakresu [0, 255] i zapis do pliku
    image_normalized = (255 * (image - np.min(image)) / (np.max(image) - np.min(image))).astype(np.uint8)
    cv2.imwrite(output_path, image_normalized)

def find_ridges(image_path, output_dir):
    # Wczytanie obrazu w odcieniach szarości
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Nie udało się wczytać obrazu. Sprawdź ścieżkę do pliku.")

    # Normalizacja obrazu do zakresu [0, 1]
    image = image.astype(np.float32) / 255.0

    # Obliczanie drugich pochodnych kierunkowych
    gxx, gyy, gxy = compute_second_derivatives(image)

    # Obliczanie wskaźnika grzbietu jako kombinacji pochodnych
    # ridge_measure = gxx * gyy - gxy**2

    # Zapisywanie wyników do plików
    save_image(f"original_image.png", image)
    save_image(f"gxx.png", gxx)
    save_image(f"gyy.png", gyy)
    save_image(f"gxy.png", gxy)

    print(f"Obrazy zapisano w katalogu: {output_dir}")

# Przykładowe użycie
image_path = 'zyczenia-a.png'  # Podaj ścieżkę do obrazu
output_dir = ''  # Podaj katalog wyjściowy
find_ridges(image_path, output_dir)
