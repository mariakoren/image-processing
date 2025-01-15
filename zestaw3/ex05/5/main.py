import cv2
import numpy as np
import matplotlib.pyplot as plt

# Wczytanie obrazu w skali szarości
image = cv2.imread('czaszka.png', cv2.IMREAD_GRAYSCALE)

# Funkcja do normalizacji histogramu
def normalize_histogram(hist, num_pixels):
    return hist / num_pixels

# Funkcja do obliczania histogramu skumulowanego
def cumulative_histogram(hist):
    return np.cumsum(hist)

# Wyrównanie histogramu
def equalize_histogram(image):
    equalized = cv2.equalizeHist(image)
    return equalized

# Hiperbolizacja histogramu
def hyperbolize_histogram(image, alpha=-1/3):
    normalized = image / 255.0
    hyperbolized = np.power(normalized, 1 + alpha)
    hyperbolized = np.clip(hyperbolized * 255, 0, 255).astype(np.uint8)
    return hyperbolized

# Obliczenie histogramu
def calculate_histogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist

# Wyrównanie histogramu
equalized_image = equalize_histogram(image)
equalized_hist = calculate_histogram(equalized_image)

# Hiperbolizacja histogramu
hyperbolized_image = hyperbolize_histogram(image)
hyperbolized_hist = calculate_histogram(hyperbolized_image)

# Histogramy i histogramy skumulowane
original_hist = calculate_histogram(image)
normalized_hist = normalize_histogram(original_hist, image.size)
cumulative_hist = cumulative_histogram(normalized_hist)

# Rysowanie wyników
fig, axs = plt.subplots(3, 3, figsize=(15, 12))

# Oryginalny obraz i histogram
axs[0, 0].imshow(image, cmap='gray')
axs[0, 0].set_title("Oryginalny obraz")
axs[0, 0].axis('off')

axs[0, 1].plot(normalized_hist)
axs[0, 1].set_title("Znormalizowany histogram Hn(g)")

axs[0, 2].plot(cumulative_hist)
axs[0, 2].set_title("Skumulowany histogram Hs(g)")

# Wyrównany obraz i histogram
axs[1, 0].imshow(equalized_image, cmap='gray')
axs[1, 0].set_title("Obraz po wyrównaniu histogramu")
axs[1, 0].axis('off')

axs[1, 1].plot(equalized_hist)
axs[1, 1].set_title("Histogram po wyrównaniu")

axs[1, 2].axis('off')

# Hiperbolizowany obraz i histogram
axs[2, 0].imshow(hyperbolized_image, cmap='gray')
axs[2, 0].set_title("Obraz po hiperbolizacji histogramu")
axs[2, 0].axis('off')

axs[2, 1].plot(hyperbolized_hist)
axs[2, 1].set_title("Histogram po hiperbolizacji")

axs[2, 2].axis('off')

plt.tight_layout()
plt.savefig("all.png")
