import cv2
import numpy as np
import matplotlib.pyplot as plt

# Wczytanie obrazu
image = cv2.imread('TestFiltrowSlon.png', cv2.IMREAD_GRAYSCALE)

# Definicja jąder splotu
ha = np.array([[0, 0, 1, 0, 0],
               [0, 2, 2, 2, 0],
               [1, 2, 5, 2, 1],
               [0, 2, 2, 2, 0],
               [0, 0, 1, 0, 0]]) / 25.0  # Normalizacja

hb = np.array([[1, -2, 1],
               [-2, 5, -2],
               [1, -2, 1]]) / 1.0  # Normalizacja

hc = np.array([[0, 1, 1],
               [-1, 1, 1],
               [-1, -1, 0]]) / 1.0  # Normalizacja

hd = np.array([[1, -1, -1],
               [1, -2, -1],
               [1, 1, 1]]) / 1.0  # Normalizacja

# Zastosowanie filtrów
filtered_ha = cv2.filter2D(image, -1, ha)
filtered_hb = cv2.filter2D(image, -1, hb)
filtered_hc = cv2.filter2D(image, -1, hc)
filtered_hd = cv2.filter2D(image, -1, hd)

# Wyświetlenie wyników
plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.imshow(filtered_ha, cmap='gray')
plt.title('Filtr ha')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(filtered_hb, cmap='gray')
plt.title('Filtr hb')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(filtered_hc, cmap='gray')
plt.title('Filtr hc')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(filtered_hd, cmap='gray')
plt.title('Filtr hd')
plt.axis('off')

plt.tight_layout()
plt.savefig("a.png")
