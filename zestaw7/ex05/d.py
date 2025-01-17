import cv2
import numpy as np
import math

def compute_gradients(image_path):
    # Wczytanie obrazu w skali szarości
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Nie udało się wczytać obrazu. Sprawdź ścieżkę.")

    # Obliczenie gradientów kierunkowych za pomocą filtrów Sobela
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Gradienty zgodnie z normą L∞ (maksimum wartości)
    gx = np.max(np.abs(sobel_x), axis=0)
    gy = np.max(np.abs(sobel_y), axis=0)

    # Wartość gradientu G (norma euklidesowa)
    G = np.sqrt(sobel_x**2 + sobel_y**2)

    # Kąt gradientu θ (w stopniach)
    theta = np.arctan2(sobel_y, sobel_x) * (180 / np.pi)

    # Normalizacja kątów do przedziału [0, 360)
    theta = (theta + 360) % 360

    return gx, gy, G, theta

def assign_edge_directions(theta):
    """
    Przyporządkowuje kierunkom gradientu możliwe kierunki krawędzi.
    Zwraca macierz z przypisanymi kierunkami: poziome, pionowe, ukośne.
    """
    directions = np.empty_like(theta, dtype=object)
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            angle = theta[i, j]
            if (0 <= angle < 22.5) or (157.5 <= angle < 202.5) or (337.5 <= angle <= 360):
                directions[i, j] = "Horizontal"
            elif (22.5 <= angle < 67.5) or (202.5 <= angle < 247.5):
                directions[i, j] = "Diagonal /"
            elif (67.5 <= angle < 112.5) or (247.5 <= angle < 292.5):
                directions[i, j] = "Vertical"
            else:
                directions[i, j] = "Diagonal \\\""
    return directions

# def save_results_to_file(G, theta, directions, output_file):
#     with open(output_file, 'w') as f:
#         for i in range(G.shape[0]):
#             for j in range(G.shape[1]):
#                 f.write(f"Pixel({i},{j}): G={G[i, j]:.2f}, Theta={theta[i, j]:.2f}, Direction={directions[i, j]}\n")

def apply_gradients_to_image(G, theta, output_image_g, output_image_theta):
    # Normalizacja wartości G do przedziału 0-255
    G_normalized = cv2.normalize(G, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(output_image_g, G_normalized)

    # Normalizacja wartości Theta do przedziału 0-255 (mapowanie kątów)
    theta_normalized = cv2.normalize(theta, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(output_image_theta, theta_normalized)

# Przykład użycia
if __name__ == "__main__":
    image_path = "pajak-a.png"  # Zamień na ścieżkę do swojego obrazu
    output_image_g = "pajak-d-gradient.png"
    output_image_theta = "pajak-d-theta.png"

    try:
        gx, gy, G, theta = compute_gradients(image_path)
        directions = assign_edge_directions(theta)
        # save_results_to_file(G, theta, directions, output_file)
        apply_gradients_to_image(G, theta, output_image_g, output_image_theta)

        print(f"Obrazy gradientu G i Theta zapisane do plików: {output_image_g}, {output_image_theta}")
    except ValueError as e:
        print(e)