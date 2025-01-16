import cv2
import numpy as np
import matplotlib.pyplot as plt

def zhang_suen_thinning(image):
    """Implementacja algorytmu Zhang-Suen do szkieletowania."""
    img = image.copy()
    changes = True
    
    while changes:
        changes = False
        
        # Subiteracja 1
        to_remove = []
        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                if img[i, j] == 1:
                    P2 = img[i-1, j]
                    P3 = img[i-1, j+1]
                    P4 = img[i, j+1]
                    P5 = img[i+1, j+1]
                    P6 = img[i+1, j]
                    P7 = img[i+1, j-1]
                    P8 = img[i, j-1]
                    P9 = img[i-1, j-1]

                    neighbors = [P2, P3, P4, P5, P6, P7, P8, P9]
                    C = sum((neighbors[k] == 0 and neighbors[(k + 1) % 8] == 1) for k in range(8))
                    N = sum(neighbors)
                    if (2 <= N <= 6 and C == 1 and P2 * P4 * P6 == 0 and P4 * P6 * P8 == 0):
                        to_remove.append((i, j))
        for i, j in to_remove:
            img[i, j] = 0
            changes = True

        # Subiteracja 2
        to_remove = []
        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                if img[i, j] == 1:
                    P2 = img[i-1, j]
                    P3 = img[i-1, j+1]
                    P4 = img[i, j+1]
                    P5 = img[i+1, j+1]
                    P6 = img[i+1, j]
                    P7 = img[i+1, j-1]
                    P8 = img[i, j-1]
                    P9 = img[i-1, j-1]

                    neighbors = [P2, P3, P4, P5, P6, P7, P8, P9]
                    C = sum((neighbors[k] == 0 and neighbors[(k + 1) % 8] == 1) for k in range(8))
                    N = sum(neighbors)
                    if (2 <= N <= 6 and C == 1 and P2 * P4 * P8 == 0 and P2 * P6 * P8 == 0):
                        to_remove.append((i, j))
        for i, j in to_remove:
            img[i, j] = 0
            changes = True

    return img

# Wczytaj obraz mikolaj.png
image = cv2.imread('mikolaj.png', cv2.IMREAD_GRAYSCALE)

# Konwertuj obraz na binarny
binary_image = np.where(image > 127, 0, 1).astype(np.uint8)

# Przygotowanie kopii do zaznaczania usuniętych pikseli
a_removed = binary_image.copy()
b_removed = binary_image.copy()

# Subiteracja 1 pierwszej iteracji
to_remove = []
for i in range(1, binary_image.shape[0] - 1):
    for j in range(1, binary_image.shape[1] - 1):
        if binary_image[i, j] == 1:
            P2 = binary_image[i-1, j]
            P3 = binary_image[i-1, j+1]
            P4 = binary_image[i, j+1]
            P5 = binary_image[i+1, j+1]
            P6 = binary_image[i+1, j]
            P7 = binary_image[i+1, j-1]
            P8 = binary_image[i, j-1]
            P9 = binary_image[i-1, j-1]

            neighbors = [P2, P3, P4, P5, P6, P7, P8, P9]
            C = sum((neighbors[k] == 0 and neighbors[(k + 1) % 8] == 1) for k in range(8))
            N = sum(neighbors)
            if (2 <= N <= 6 and C == 1 and P2 * P4 * P6 == 0 and P4 * P6 * P8 == 0):
                to_remove.append((i, j))
for i, j in to_remove:
    a_removed[i, j] = 0

# Cała pierwsza iteracja
to_remove = []
for i in range(1, a_removed.shape[0] - 1):
    for j in range(1, a_removed.shape[1] - 1):
        if a_removed[i, j] == 1:
            P2 = a_removed[i-1, j]
            P3 = a_removed[i-1, j+1]
            P4 = a_removed[i, j+1]
            P5 = a_removed[i+1, j+1]
            P6 = a_removed[i+1, j]
            P7 = a_removed[i+1, j-1]
            P8 = a_removed[i, j-1]
            P9 = a_removed[i-1, j-1]

            neighbors = [P2, P3, P4, P5, P6, P7, P8, P9]
            C = sum((neighbors[k] == 0 and neighbors[(k + 1) % 8] == 1) for k in range(8))
            N = sum(neighbors)
            if (2 <= N <= 6 and C == 1 and P2 * P4 * P8 == 0 and P2 * P6 * P8 == 0):
                to_remove.append((i, j))
for i, j in to_remove:
    b_removed[i, j] = 0

# Wykonaj algorytm Zhang-Suen
skeleton = zhang_suen_thinning(binary_image)

# Zapis wyników do plików
cv2.imwrite('original_binary_image.png', (binary_image * 255).astype(np.uint8))
cv2.imwrite('subiter1_removed.png', (a_removed * 255).astype(np.uint8))
cv2.imwrite('iter1_removed.png', (b_removed * 255).astype(np.uint8))
cv2.imwrite('skeleton.png', (skeleton * 255).astype(np.uint8))

# Wizualizacja wyników
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.title("Oryginalny obraz")
plt.imshow(binary_image, cmap='gray')

plt.subplot(2, 2, 2)
plt.title("Usunięte piksele - Subiteracja 1")
plt.imshow(a_removed, cmap='gray')

plt.subplot(2, 2, 3)
plt.title("Usunięte piksele - Cała iteracja 1")
plt.imshow(b_removed, cmap='gray')

plt.subplot(2, 2, 4)
plt.title("Szkielet")
plt.imshow(skeleton, cmap='gray')

plt.tight_layout()
# plt.show()
plt.savefig("all.png")
