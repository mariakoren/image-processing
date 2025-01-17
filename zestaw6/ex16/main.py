import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_image(title, img, cmap='gray'):
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.show()

# Krok 1: Wczytanie obrazu
image_path = 'StoLat.png'  # Podaj ścieżkę do obrazu
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
show_image("Oryginalny obraz w odcieniach szarości", gray)

# Krok 2: Progowanie
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
show_image("Obraz binarny", binary)

# Krok 3: Wykrywanie pionowych linii (potencjalnych linii taktowych)
kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_vertical)
show_image("Pionowe linie", vertical_lines)

# Krok 4: Znalezienie i policzenie pionowych linii
contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
result = image.copy()

# Rysowanie konturów i liczenie linii
num_measures = 0
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
    num_measures += 1

print(f"Liczba taktów w zapisie nutowym: {num_measures - 1}")

# Wyświetlenie wynikowego obrazu z zaznaczonymi liniami
show_image("Zaznaczone linie taktowe", result, cmap=None)

# Opcjonalne zapisywanie wynikowego obrazu
cv2.imwrite('result_with_measures.png', result)
