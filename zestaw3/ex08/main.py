import cv2
import numpy as np
import matplotlib.pyplot as plt

# Wczytaj obraz
image = cv2.imread('CalunTurynski.png', cv2.IMREAD_GRAYSCALE)

# Funkcja do zapisywania histogramu
def save_histogram(image, step_name):
    plt.figure(figsize=(6, 4))
    plt.hist(image.ravel(), bins=256, range=(0, 255), color='black')
    plt.title(f'Histogram - {step_name}')
    plt.xlabel('Intensywność pikseli')
    plt.ylabel('Liczba pikseli')
    plt.savefig(f'histogram_{step_name}.png')  # Zapisz histogram
    plt.close()

# Funkcja do zapisywania obrazu
def save_image(image, step_name):
    cv2.imwrite(f'{step_name}.png', image)

# 1. Obliczenie histogramu oryginalnego obrazu
save_histogram(image, "Oryginalny")
save_image(image, "Oryginalny")

# 2. Wyrównanie histogramu
equalized_image = cv2.equalizeHist(image)
save_histogram(equalized_image, "Wyrównanie_Histogramu")
save_image(equalized_image, "Wyrównanie_Histogramu")

# 3. Dostosowanie kontrastu (ręcznie)
alpha = 1.5  # Współczynnik kontrastu
beta = 50    # Jasność
adjusted_image = cv2.convertScaleAbs(equalized_image, alpha=alpha, beta=beta)
save_histogram(adjusted_image, "Dostosowanie_Kontrastu")
save_image(adjusted_image, "Dostosowanie_Kontrastu")

# 4. Transformacja Gamma
gamma = 0.8  # Wartość gamma < 1 rozjaśni obraz
gamma_corrected = np.array(255 * (adjusted_image / 255) ** gamma, dtype='uint8')
save_histogram(gamma_corrected, "Transformacja_Gamma")
save_image(gamma_corrected, "Transformacja_Gamma")

# 5. Wygładzanie obrazu (filtr Gaussa)
smoothed_image = cv2.GaussianBlur(gamma_corrected, (5, 5), 0)
save_histogram(smoothed_image, "Wygładzanie")
save_image(smoothed_image, "Wygładzanie")

# 6. Obliczenie histogramu po przetworzeniu
save_histogram(smoothed_image, "Po_Przetworzeniu")
save_image(smoothed_image, "Po_Przetworzeniu")
