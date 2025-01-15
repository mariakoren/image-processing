import cv2
import numpy as np
from scipy.signal import convolve2d
 
# Wczytaj obraz
image_path = 'images/smok.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
 
# Definiowanie filtrów
h1 = np.ones((3, 3)) / 9  # Filtr uśredniający 3x3
h2 = np.array([[-1], [0], [1]])  # Jednowymiarowy pionowy filtr gradientowy
 
# (a) Wyznaczenie obrazu g1 = g ∗ h1
g1 = convolve2d(image, h1, boundary='symm', mode='same')
g1 = np.clip(g1, 0, 255).astype(np.uint8)
cv2.imwrite('images/g1.png', g1)
 
# (b) Wyznaczenie obrazu g2 = g1 ∗ h2
g2 = convolve2d(g1, h2, boundary='symm', mode='same')
g2 = np.clip(g2, 0, 255).astype(np.uint8)
cv2.imwrite('images/g2.png', g2)
 
# (c) Połączenie filtrów h1 i h2 w jeden dwuwymiarowy filtr h3
h3 = convolve2d(h1, h2, mode='full')
 
# Wyznaczenie obrazu g3 = g ∗ h3
g3 = convolve2d(image, h3, boundary='symm', mode='same')
g3 = np.clip(g3, 0, 255).astype(np.uint8)
cv2.imwrite('images/g3.png', g3)
 
print("Obrazy g1, g2 i g3 zostały zapisane.")