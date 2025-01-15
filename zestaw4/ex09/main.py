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
h_S = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=float)

# Filtr Laplace'a h_L
h_L = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
], dtype=float)

# Funkcja do obliczenia splotu
def apply_filter(image, kernel):
    return convolve(image, kernel, mode='constant', cval=0.0)

# Obliczenie splotów dla każdego obrazu
g1_prime_g = apply_filter(g1, h_g)
g2_prime_g = apply_filter(g2, h_g)
g3_prime_g = apply_filter(g3, h_g)

g1_prime_P = apply_filter(g1, h_P)
g2_prime_P = apply_filter(g2, h_P)
g3_prime_P = apply_filter(g3, h_P)

g1_prime_S = apply_filter(g1, h_S)
g2_prime_S = apply_filter(g2, h_S)
g3_prime_S = apply_filter(g3, h_S)

g1_prime_L = apply_filter(g1, h_L)
g2_prime_L = apply_filter(g2, h_L)
g3_prime_L = apply_filter(g3, h_L)

# Funkcja do zapisywania obrazów do plików
def save_image(image, filename):
    plt.imsave(filename, image, cmap='gray')

# Zapisz obrazy g1, g2, g3
save_image(g1, 'g1.png')
save_image(g2, 'g2.png')
save_image(g3, 'g3.png')

# Zapisz obrazy wynikowe dla każdego filtru
save_image(g1_prime_g, 'g1_prime_g.png')
save_image(g2_prime_g, 'g2_prime_g.png')
save_image(g3_prime_g, 'g3_prime_g.png')

save_image(g1_prime_P, 'g1_prime_P.png')
save_image(g2_prime_P, 'g2_prime_P.png')
save_image(g3_prime_P, 'g3_prime_P.png')

save_image(g1_prime_S, 'g1_prime_S.png')
save_image(g2_prime_S, 'g2_prime_S.png')
save_image(g3_prime_S, 'g3_prime_S.png')

save_image(g1_prime_L, 'g1_prime_L.png')
save_image(g2_prime_L, 'g2_prime_L.png')
save_image(g3_prime_L, 'g3_prime_L.png')

# Tworzenie jednego obrazu zawierającego wszystkie obrazy
fig, axes = plt.subplots(3, 4, figsize=(16, 9))

# Układ obrazów w siatce 3x4
axes[0, 0].imshow(g1, cmap='gray')
axes[0, 0].set_title('g1')
axes[0, 0].axis('off')

axes[0, 1].imshow(g2, cmap='gray')
axes[0, 1].set_title('g2')
axes[0, 1].axis('off')

axes[0, 2].imshow(g3, cmap='gray')
axes[0, 2].set_title('g3')
axes[0, 2].axis('off')

axes[0, 3].imshow(g1_prime_g, cmap='gray')
axes[0, 3].set_title('g1_prime_g')
axes[0, 3].axis('off')

axes[1, 0].imshow(g1_prime_P, cmap='gray')
axes[1, 0].set_title('g1_prime_P')
axes[1, 0].axis('off')

axes[1, 1].imshow(g2_prime_P, cmap='gray')
axes[1, 1].set_title('g2_prime_P')
axes[1, 1].axis('off')

axes[1, 2].imshow(g3_prime_P, cmap='gray')
axes[1, 2].set_title('g3_prime_P')
axes[1, 2].axis('off')

axes[1, 3].imshow(g1_prime_S, cmap='gray')
axes[1, 3].set_title('g1_prime_S')
axes[1, 3].axis('off')

axes[2, 0].imshow(g2_prime_S, cmap='gray')
axes[2, 0].set_title('g2_prime_S')
axes[2, 0].axis('off')

axes[2, 1].imshow(g3_prime_S, cmap='gray')
axes[2, 1].set_title('g3_prime_S')
axes[2, 1].axis('off')

axes[2, 2].imshow(g1_prime_L, cmap='gray')
axes[2, 2].set_title('g1_prime_L')
axes[2, 2].axis('off')

axes[2, 3].imshow(g2_prime_L, cmap='gray')
axes[2, 3].set_title('g2_prime_L')
axes[2, 3].axis('off')

# Usuwamy ostatnią pustą komórkę
axes[2, 3].axis('off')

# Zapisanie zbiorczego obrazu
plt.tight_layout()
plt.savefig('all_images_combined.png')

print("Obrazy zostały zapisane do plików oraz połączony obraz w 'all_images_combined.png'.")
