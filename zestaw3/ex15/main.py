import cv2
import numpy as np

# Funkcja do ładowania obrazów
def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Funkcja do obliczania średniej wartości obrazu
def calculate_mean(image):
    return np.mean(image)

# Funkcja do korekcji Flat-Field
def flat_field_correction(g, g_D, g_F):
    # Oblicz różnicę g_F - g_D
    g_F_minus_g_D = g_F - g_D
    
    # Oblicz średnią wartość dla g_F - g_D
    g_mean = calculate_mean(g_F_minus_g_D)
    
    # Oblicz różnicę g - g_D
    g_minus_g_D = g - g_D
    
    # Wykonaj korekcję Flat-Field
    g_C = (g_minus_g_D * g_mean) / g_F_minus_g_D
    
    return g_C

# Wczytanie obrazów
g = load_image('torus.png')        # Główny obraz
g_D = load_image('dark_frame.png')  # Dark frame
g_F = load_image('flat_frame.png')  # Flat frame

# Upewnij się, że obrazy mają tę samą wielkość
if g.shape != g_D.shape or g.shape != g_F.shape:
    raise ValueError("Obrazy muszą mieć ten sam rozmiar")

# Zastosowanie korekcji Flat-Field
g_C = flat_field_correction(g, g_D, g_F)

# Zapisz wynik do pliku (zapisujemy tylko obraz po korekcji)
cv2.imwrite('torus-a.png', g_C)
