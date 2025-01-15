import numpy as np
import cv2
import matplotlib.pyplot as plt

# Wczytanie obrazu
image = cv2.imread('Sky_and_Water_I.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Zmiana kolorów z BGR na RGB

# Wymiary obrazu
height, width, _ = image.shape

# Macierz transformacji
M = np.array([[1, 0.5, 0],  # Przesunięcie w poziomie o 50%
              [0, 1, 0],    # Brak zmiany w pionie
              [0, 0, 1]])   # Współrzędne homogeniczne

# Funkcja do transformacji obrazu za pomocą macierzy
def apply_transformation(image, M):
    # Tworzenie nowego obrazu z nowymi wymiarami
    transformed_image = np.zeros_like(image)

    # Przekształcanie każdego punktu obrazu
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # Współrzędne punktu w obrazie
            old_coords = np.array([x, y, 1])  # Dodanie współrzędnej homogenicznej
            new_coords = M @ old_coords  # Mnożenie macierzy transformacji
            new_x, new_y = new_coords[:2]  # Nowe współrzędne

            # Jeśli nowe współrzędne mieszczą się w obrębie obrazu, kopiujemy piksel
            if 0 <= new_x < image.shape[1] and 0 <= new_y < image.shape[0]:
                transformed_image[int(new_y), int(new_x)] = image[y, x]

    return transformed_image

# Zastosowanie transformacji
transformed_image = apply_transformation(image, M)

# Funkcja do interpolacji najbliższego sąsiada (nearest neighbor)
def nearest_neighbor_interpolation(image, M):
    transformed_image = np.zeros_like(image)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            old_coords = np.array([x, y, 1])
            new_coords = M @ old_coords
            new_x, new_y = new_coords[:2]

            # Zaokrąglamy współrzędne do najbliższego punktu
            new_x, new_y = int(round(new_x)), int(round(new_y))

            # Sprawdzamy, czy nowe współrzędne mieszczą się w obrazie
            if 0 <= new_x < image.shape[1] and 0 <= new_y < image.shape[0]:
                transformed_image[y, x] = image[new_y, new_x]

    return transformed_image

# Funkcja do interpolacji dwuliniowej
def bilinear_interpolation(image, M):
    transformed_image = np.zeros_like(image)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            old_coords = np.array([x, y, 1])
            new_coords = M @ old_coords
            new_x, new_y = new_coords[:2]

            # Zaokrąglanie współrzędnych
            x1, y1 = int(np.floor(new_x)), int(np.floor(new_y))
            x2, y2 = min(x1 + 1, image.shape[1] - 1), min(y1 + 1, image.shape[0] - 1)

            # Sprawdzamy, czy współrzędne mieszczą się w granicach obrazu
            if 0 <= x1 < image.shape[1] and 0 <= y1 < image.shape[0]:
                Q11 = image[y1, x1]
            else:
                Q11 = 0  # Wartość 0, gdy współrzędna wykracza poza obraz

            if 0 <= x2 < image.shape[1] and 0 <= y1 < image.shape[0]:
                Q12 = image[y1, x2]
            else:
                Q12 = 0  # Wartość 0, gdy współrzędna wykracza poza obraz

            if 0 <= x1 < image.shape[1] and 0 <= y2 < image.shape[0]:
                Q21 = image[y2, x1]
            else:
                Q21 = 0  # Wartość 0, gdy współrzędna wykracza poza obraz

            if 0 <= x2 < image.shape[1] and 0 <= y2 < image.shape[0]:
                Q22 = image[y2, x2]
            else:
                Q22 = 0  # Wartość 0, gdy współrzędna wykracza poza obraz

            # Obliczenie wartości piksela za pomocą interpolacji dwuliniowej
            dx = new_x - x1
            dy = new_y - y1
            transformed_image[y, x] = (Q11 * (1 - dx) * (1 - dy) + 
                                        Q12 * dx * (1 - dy) + 
                                        Q21 * (1 - dx) * dy + 
                                        Q22 * dx * dy)

    return transformed_image
# Zastosowanie interpolacji nearest neighbor
nearest_neighbor_image = nearest_neighbor_interpolation(image, M)

# Zastosowanie interpolacji dwuliniowej
bilinear_image = bilinear_interpolation(image, M)

# Wyświetlenie wyników
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Wyświetlanie oryginalnego obrazu
ax[0].imshow(image)
ax[0].set_title("Oryginalny obraz")
ax[0].axis('off')

# Wyświetlanie obrazu po transformacji
ax[1].imshow(transformed_image)
ax[1].set_title("Obraz po transformacji")
ax[1].axis('off')

# Wyświetlanie obrazu po interpolacji nearest neighbor
ax[2].imshow(nearest_neighbor_image)
ax[2].set_title("Nearest Neighbor Interpolation")
ax[2].axis('off')

plt.show()

# Możemy także zapisać obrazy wynikowe, jeśli chcemy
cv2.imwrite('transformed_image.png', cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))
cv2.imwrite('nearest_neighbor_image.png', cv2.cvtColor(nearest_neighbor_image, cv2.COLOR_RGB2BGR))
cv2.imwrite('bilinear_image.png', cv2.cvtColor(bilinear_image, cv2.COLOR_RGB2BGR))
