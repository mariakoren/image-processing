import cv2
import numpy as np
import matplotlib.pyplot as plt

# # Wczytaj obraz (w skali szarości lub kolorowy)
# image_name="pająk"
# image_path = f"{image_name}.png"  # Zamień na ścieżkę do obrazu
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# if image is None:
#     raise FileNotFoundError(f"Nie znaleziono obrazu w {image_path}")

# # Filtry Sobela
# sobel_x = np.array([[1, 0, -1],
#                     [2, 0, -2],
#                     [1, 0, -1]])
# sobel_y = np.array([[-1, -2, -1],
#                     [0,  0,  0],
#                     [1,  2,  1]])

# # Oblicz gradienty za pomocą filtru Sobela
# gradient_x = cv2.filter2D(image, -1, sobel_x)  # Gradient w kierunku poziomym
# gradient_y = cv2.filter2D(image, -1, sobel_y)  # Gradient w kierunku pionowym

def sobel(image_name):
    image_path = f"{image_name}.png"  # Zamień na ścieżkę do obrazu
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Nie znaleziono obrazu w {image_path}")

    # Filtry Sobela
    sobel_x = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])
    sobel_y = np.array([[-1, -2, -1],
                        [0,  0,  0],
                        [1,  2,  1]])

    # Oblicz gradienty za pomocą filtru Sobela
    gradient_x = cv2.filter2D(image, -1, sobel_x)  # Gradient w kierunku poziomym
    gradient_y = cv2.filter2D(image, -1, sobel_y)  # Gradient w kierunku pionowym


    cv2.imwrite(f"{image_name}-sobel-x.png", gradient_x)  # Gradient poziomy
    cv2.imwrite(f"{image_name}-sobel-y.png", gradient_y)  # Gradient pionowy

    # Dodatkowo wyświetl wyniki
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Oryginalny obraz")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Gradient X (Sobel X)")
    plt.imshow(gradient_x, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Gradient Y (Sobel Y)")
    plt.imshow(gradient_y, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

sobel("pajak-a-blue")
sobel("pajak-a-green")
sobel("pajak-a-red")
