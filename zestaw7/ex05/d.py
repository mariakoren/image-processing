import cv2
import numpy as np

def quantize_angle(angle):
    """Przyporządkuj kierunki gradientu do głównych kierunków krawędzi."""
    # Możliwe kierunki: poziomy (0), pionowy (90), ukośne (45, 135)
    if (angle >= 0 and angle < 22.5) or (angle >= 157.5 and angle < 202.5) or (angle >= 337.5 and angle < 360):
        return 0  # poziomy
    elif (angle >= 22.5 and angle < 67.5) or (angle >= 202.5 and angle < 247.5):
        return 45  # ukośny
    elif (angle >= 67.5 and angle < 112.5) or (angle >= 247.5 and angle < 292.5):
        return 90  # pionowy
    else:
        return 135  # ukośny drugi kierunek

def process_image(input_file, output_file):
    # Wczytaj obraz w skali szarości
    img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Nie udało się wczytać obrazu.")

    # Oblicz gradienty Sobela
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Oblicz wielkość i kierunek gradientu
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    angle = np.arctan2(sobel_y, sobel_x) * (180 / np.pi)  # Konwersja na stopnie
    angle[angle < 0] += 360  # Ustawienie wartości kąta w zakresie [0, 360)

    # Kwantyzacja kierunków gradientu
    quantized_angles = np.vectorize(quantize_angle)(angle)

    # Przekształć kwantyzowane kierunki na wartości wizualne (np. 0-255)
    output_img = np.uint8((quantized_angles / 135) * 255)

    # Zapisz wynikowy obraz
    cv2.imwrite(output_file, output_img)

# Przykład użycia
input_file = "pajak-a.png"  # Zmień na ścieżkę do swojego pliku wejściowego
output_file = "pajak-d.png"  # Zmień na ścieżkę do swojego pliku wynikowego
process_image(input_file, output_file)
print("Obraz wynikowy zapisany do", output_file)
