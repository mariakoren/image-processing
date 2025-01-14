import numpy as np
import cv2
import matplotlib.pyplot as plt

def sinusoidal_window(m, n, M, N):
    """Generuje okno sinusoidalne w (m,n) dla wymiarów obrazu M x N"""
    return np.sin(np.pi * m / M) * np.sin(np.pi * n / N)

def apply_sinusoidal_window(image_path):
    # Wczytanie obrazu
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError("Nie udało się wczytać obrazu.")

    # Wymiary obrazu
    M, N = img.shape
    
    # Utworzenie okna sinusoidalnego
    m = np.arange(M).reshape((M, 1))  # Indeksy wierszy
    n = np.arange(N).reshape((1, N))  # Indeksy kolumn
    
    window = sinusoidal_window(m, n, M, N)
    
    # Przemnożenie obrazu przez okno sinusoidalne
    processed_image = img * window
    
    # Normalizacja, aby piksele nie przekroczyły wartości 255
    processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)
    
    # Zwrócenie przetworzonego obrazu
    return processed_image

def show_images(original, processed):
    """Funkcja do wyświetlania obrazów: oryginalnego i przetworzonego"""
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Oryginalny obraz')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(processed, cmap='gray')
    plt.title('Przetworzony obraz')
    plt.axis('off')
    
    plt.show()

# Ścieżka do obrazu
image_path = 'ptaki.png'

# Przetwarzanie obrazu
original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
processed_img = apply_sinusoidal_window(image_path)

# Wyświetlanie wyników
show_images(original_img, processed_img)

# Zapisanie przetworzonego obrazu do pliku
output_path = 'ptaki-a.png'
cv2.imwrite(output_path, processed_img)
print(f'Przetworzony obraz zapisano do pliku: {output_path}')
