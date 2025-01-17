import numpy as np
import cv2

def non_maximum_suppression(gradient_magnitude, gradient_direction):
    # Przygotowanie wynikowej macierzy
    output_image = np.zeros_like(gradient_magnitude)
    
    # Konwersja kierunku gradientu do stopni
    gradient_direction = gradient_direction * 180 / np.pi  # konwersja z radianów na stopnie
    gradient_direction[gradient_direction < 0] += 180  # poprawienie wartości ujemnych
    
    # Sprawdzamy dla każdego piksela
    for i in range(1, gradient_magnitude.shape[0] - 1):
        for j in range(1, gradient_magnitude.shape[1] - 1):
            # Pobranie kierunku gradientu dla danego piksela
            angle = gradient_direction[i, j]
            magnitude = gradient_magnitude[i, j]
            
            # Maski sąsiedztwa w zależności od kąta gradientu
            if (angle >= 337.5 or angle < 22.5) or (angle >= 157.5 and angle < 202.5):
                neighbor1 = gradient_magnitude[i, j+1]  # Piksel w prawo
                neighbor2 = gradient_magnitude[i, j-1]  # Piksel w lewo
            elif (angle >= 22.5 and angle < 67.5) or (angle >= 202.5 and angle < 247.5):
                neighbor1 = gradient_magnitude[i+1, j-1]  # Piksel w dolnym lewym rogu
                neighbor2 = gradient_magnitude[i-1, j+1]  # Piksel w górnym prawym rogu
            elif (angle >= 67.5 and angle < 112.5) or (angle >= 247.5 and angle < 292.5):
                neighbor1 = gradient_magnitude[i+1, j]  # Piksel poniżej
                neighbor2 = gradient_magnitude[i-1, j]  # Piksel powyżej
            else:
                neighbor1 = gradient_magnitude[i-1, j-1]  # Piksel w górnym lewym rogu
                neighbor2 = gradient_magnitude[i+1, j+1]  # Piksel w dolnym prawym rogu
            
            # Stłumienie wartości gradientu, jeżeli nie jest lokalnym maksimum
            if magnitude >= neighbor1 and magnitude >= neighbor2:
                output_image[i, j] = magnitude
            else:
                output_image[i, j] = 0
    
    return output_image

# Przykład użycia:

# Załaduj obraz i przekształć na skalę szarości
image = cv2.imread('pajak-a.png', cv2.IMREAD_GRAYSCALE)

# Oblicz gradient obrazu (np. Sobel)
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# Oblicz wartość gradientu i kierunku
gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)
gradient_direction = cv2.phase(sobel_x, sobel_y, angleInDegrees=True)

# Zastosuj Non-Maximum Suppression
nms_image = non_maximum_suppression(gradient_magnitude, gradient_direction)

# Zapisz wynikowy obraz
cv2.imwrite('pajak-e.png', nms_image)
