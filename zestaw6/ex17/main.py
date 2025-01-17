import cv2
import numpy as np

# Wczytanie obrazu
image = cv2.imread('image.png')  # Zmień na ścieżkę do swojego obrazu
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Zamiana na obraz binarny (czarny, biały)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# Wykrywanie konturów
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Funkcja do obliczania proporcji i klasyfikacji
def classify_contour(contour):
    # Przybliżenie konturu do prostokąta otaczającego
    x, y, w, h = cv2.boundingRect(contour)
    
    # Obliczanie proporcji
    aspect_ratio = w / float(h)
    area = cv2.contourArea(contour)

    # Klasyfikacja na podstawie proporcji
    if aspect_ratio > 1 and area >3:  # Grubsze i długie
        return 'thick_long', (0, 255, 0)  # Zielony
    else: # Cieńsze i dłuższe
        return 'thin_long', (0, 0, 255)  # Czerwony
    # else:  # Cieńsze i krótsze
    #     return 'thin_short', (255, 0, 0)  # Niebieski

# Tworzenie kopii obrazu do oznaczenia
output_image = image.copy()

# Przechodzenie przez wszystkie kontury
for contour in contours:
    if cv2.contourArea(contour) > 500:  # Pomijamy zbyt małe obiekty
        label, color = classify_contour(contour)
        cv2.drawContours(output_image, [contour], -1, color, 3)  # Rysowanie konturów w odpowiednich kolorach

# Wyświetlanie wyników
cv2.imshow('classified_image', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
