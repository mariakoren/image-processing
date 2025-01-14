import numpy as np
import cv2
import matplotlib.pyplot as plt

def smooth_image(image):
    # Wymiary obrazu
    rows, cols = image.shape
    
    # Utworzenie pustego obrazu na wynik
    smoothed_image = np.zeros_like(image, dtype=np.float32)
    
    # Iteracja po każdym pikselu obrazu (pomijamy krawędzie, aby uniknąć błędów)
    for m in range(1, rows-1):
        for n in range(1, cols-1):
            # Zastosowanie formuły g'(m, n) = 1/9 * suma w oknie 3x3
            region = image[m-1:m+2, n-1:n+2]
            smoothed_image[m, n] = np.mean(region)
    
    # Zaokrąglamy wynik do wartości całkowitych
    smoothed_image = np.uint8(smoothed_image)
    
    return smoothed_image

# Wczytanie obrazu
image = cv2.imread('ptaki-a.png', cv2.IMREAD_GRAYSCALE)

# Jeśli obraz ma kolor, przekonwertuj go na skalę szarości
if len(image.shape) == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Zastosowanie funkcji wygładzającej
smoothed_image = smooth_image(image)

# Zapisanie wyniku do pliku
cv2.imwrite('ptaki-b.png', smoothed_image)

# Wyświetlenie obrazu wynikowego (opcjonalne)
plt.imshow(smoothed_image, cmap='gray')
plt.axis('off')
plt.show()
