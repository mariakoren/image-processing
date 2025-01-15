import cv2
import numpy as np

def global_contrast(image, grange=255):
    min_gray = np.min(image)
    max_gray = np.max(image)
    c_global = (max_gray - min_gray) / grange
    return c_global

def local_contrast(image):
    M, N = image.shape
    total_diff = 0
    count = 0
    
    # Przesuwanie okna 3x3 przez każdy piksel
    for m in range(1, M-1):
        for n in range(1, N-1):
            pixel_value = image[m, n]
            neighbors = [
                image[m-1, n-1], image[m -1, n], image[m-1, n+1],
                image[m, n-1],                   image[m, n+1],
                image[m+1, n-1], image[m+1, n], image[m+1, n+1]
            ]
            
            neighbors_mean = np.sum(neighbors)/8
            diff=np.abs(pixel_value-neighbors_mean)
            total_diff += diff

    c_local = total_diff / (M * N)
    return c_local

image_path1 = "torus-a.png" 
image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)

image_path2 = "torus.png" 
image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

c_global1 = global_contrast(image1)
c_global2 = global_contrast(image2)


print("Kontrast globalny:")
print(f"Torus-a: {c_global1}") # Torus-a: 1.0
print(f"Torus: {c_global2}") # Torus: 0.9137254901960784





