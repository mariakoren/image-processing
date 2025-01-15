import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt

# Definicja obrazów g1, g2, g3
g1 = np.array([
    [255, 255, 255, 255, 0, 0, 0],
    [255, 255, 255, 255, 0, 0, 0],
    [255, 255, 255, 255, 0, 0, 0],
    [255, 255, 255, 255, 0, 0, 0],
    [255, 255, 255, 255, 0, 0, 0],
    [255, 255, 255, 255, 0, 0, 0],
    [255, 255, 255, 255, 0, 0, 0]
], dtype=float)

g2 = np.array([
    [0, 0, 0, 255, 0, 0, 0],
    [0, 0, 0, 255, 0, 0, 0],
    [0, 0, 0, 255, 0, 0, 0],
    [0, 0, 0, 255, 0, 0, 0],
    [0, 0, 0, 255, 0, 0, 0],
    [0, 0, 0, 255, 0, 0, 0],
    [0, 0, 0, 255, 0, 0, 0]
], dtype=float)

g3 = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 255, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
], dtype=float)

# Filtr gradientowy h_g (w kierunku poziomym)
h_g = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=float)

# Filtr Prewitta h_P (w kierunku poziomym)
h_P = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=float)

# Filtr Sobela h_S (w kierunku poziomym)
# h_S = np.array([
#     [-1, 0, 1],
#     [-2, 0, 2],
#     [-1, 0, 1]
# ], dtype=float)

h_S =np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
])

# Filtr Laplace'a h_L
h_L = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
], dtype=float)

# Funkcja do obliczenia splotu
def apply_filter(image, kernel):
    return convolve(image, kernel, mode='constant', cval=0.0)

# Funkcja do rozszerzania obrazu (dodanie wartości 0 poza jego obszarem)
def expand_image(image, border_size):
    return np.pad(image, pad_width=border_size, mode='constant', constant_values=0)

# Funkcja do progowania obrazu do wartości 0 lub 255
def threshold_image(image, threshold=127):
    return np.where(image > threshold, 255, 0)

# Rozszerz obrazy o 1 piksel
g1_expanded = expand_image(g1, 1)
g2_expanded = expand_image(g2, 1)
g3_expanded = expand_image(g3, 1)

# Obliczenie splotów dla rozszerzonych obrazów
g1_prime_g_expanded = apply_filter(g1_expanded, h_g)
g2_prime_g_expanded = apply_filter(g2_expanded, h_g)
g3_prime_g_expanded = apply_filter(g3_expanded, h_g)

g1_prime_P_expanded = apply_filter(g1_expanded, h_P)
g2_prime_P_expanded = apply_filter(g2_expanded, h_P)
g3_prime_P_expanded = apply_filter(g3_expanded, h_P)

g1_prime_S_expanded = apply_filter(g1_expanded, h_S)
g2_prime_S_expanded = apply_filter(g2_expanded, h_S)
g3_prime_S_expanded = apply_filter(g3_expanded, h_S)

g1_prime_L_expanded = apply_filter(g1_expanded, h_L)
g2_prime_L_expanded = apply_filter(g2_expanded, h_L)
g3_prime_L_expanded = apply_filter(g3_expanded, h_L)

# Progowanie wyników do wartości 0 lub 255
g1_prime_g_expanded = threshold_image(g1_prime_g_expanded)
g2_prime_g_expanded = threshold_image(g2_prime_g_expanded)
g3_prime_g_expanded = threshold_image(g3_prime_g_expanded)

g1_prime_P_expanded = threshold_image(g1_prime_P_expanded)
g2_prime_P_expanded = threshold_image(g2_prime_P_expanded)
g3_prime_P_expanded = threshold_image(g3_prime_P_expanded)

g1_prime_S_expanded = threshold_image(g1_prime_S_expanded)
g2_prime_S_expanded = threshold_image(g2_prime_S_expanded)
g3_prime_S_expanded = threshold_image(g3_prime_S_expanded)

g1_prime_L_expanded = threshold_image(g1_prime_L_expanded)
g2_prime_L_expanded = threshold_image(g2_prime_L_expanded)
g3_prime_L_expanded = threshold_image(g3_prime_L_expanded)

# Wyświetlenie nowych wyników w jednym rzędzie 3 obrazków po tej samej operacji
fig, axes = plt.subplots(4, 3, figsize=(12, 16))

# Układ obrazów w siatce 4x3
axes[0, 0].imshow(g1_prime_g_expanded, cmap='gray')
axes[0, 0].set_title('g1_prime_g_expanded')
axes[0, 0].axis('off')

axes[0, 1].imshow(g2_prime_g_expanded, cmap='gray')
axes[0, 1].set_title('g2_prime_g_expanded')
axes[0, 1].axis('off')

axes[0, 2].imshow(g3_prime_g_expanded, cmap='gray')
axes[0, 2].set_title('g3_prime_g_expanded')
axes[0, 2].axis('off')

axes[1, 0].imshow(g1_prime_P_expanded, cmap='gray')
axes[1, 0].set_title('g1_prime_P_expanded')
axes[1, 0].axis('off')

axes[1, 1].imshow(g2_prime_P_expanded, cmap='gray')
axes[1, 1].set_title('g2_prime_P_expanded')
axes[1, 1].axis('off')

axes[1, 2].imshow(g3_prime_P_expanded, cmap='gray')
axes[1, 2].set_title('g3_prime_P_expanded')
axes[1, 2].axis('off')

axes[2, 0].imshow(g1_prime_S_expanded, cmap='gray')
axes[2, 0].set_title('g1_prime_S_expanded')
axes[2, 0].axis('off')

axes[2, 1].imshow(g2_prime_S_expanded, cmap='gray')
axes[2, 1].set_title('g2_prime_S_expanded')
axes[2, 1].axis('off')

axes[2, 2].imshow(g3_prime_S_expanded, cmap='gray')
axes[2, 2].set_title('g3_prime_S_expanded')
axes[2, 2].axis('off')

axes[3, 0].imshow(g1_prime_L_expanded, cmap='gray')
axes[3, 0].set_title('g1_prime_L_expanded')
axes[3, 0].axis('off')

axes[3, 1].imshow(g2_prime_L_expanded, cmap='gray')
axes[3, 1].set_title('g2_prime_L_expanded')
axes[3, 1].axis('off')

axes[3, 2].imshow(g3_prime_L_expanded, cmap='gray')
axes[3, 2].set_title('g3_prime_L_expanded')
axes[3, 2].axis('off')

# Zapisanie wyników w zbiorczym obrazie
plt.tight_layout()
plt.savefig('expanded_images_combined_binary_separated.png')

print("Obrazy zostały zapisane, a wynikowy obraz w 'expanded_images_combined_binary_separated.png'.")
