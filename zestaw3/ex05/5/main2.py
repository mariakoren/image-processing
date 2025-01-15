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

# Zapis obrazów
cv2.imwrite('original_image.png', image)
cv2.imwrite('equalized_image.png', equalized_image)
cv2.imwrite('hyperbolized_image.png', hyperbolized_image)

# Zapis histogramów
def save_histogram(hist, title, filename):
    plt.figure()
    plt.plot(hist)
    plt.title(title)
    plt.xlabel("Poziom szarości")
    plt.ylabel("Częstość")
    plt.grid()
    plt.savefig(filename)
    plt.close()

save_histogram(normalized_hist, "Znormalizowany histogram Hn(g)", 'normalized_histogram.png')
save_histogram(cumulative_hist, "Skumulowany histogram Hs(g)", 'cumulative_histogram.png')
save_histogram(equalized_hist, "Histogram po wyrównaniu", 'equalized_histogram.png')
save_histogram(hyperbolized_hist, "Histogram po hiperbolizacji", 'hyperbolized_histogram.png')

print("Obrazy i histogramy zostały zapisane do plików.")
