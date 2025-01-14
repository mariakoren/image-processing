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

image_path1 = "muchaA.png" 
image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)

image_path2 = "muchaB.png" 
image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

image_path3 = "muchaC.png" 
image3 = cv2.imread(image_path3, cv2.IMREAD_GRAYSCALE)

# image_path = "q3.png"
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# # image_image = Image.open(image_path).convert("L")
# # image_array = np.array(image_image, dtype = np.float64)
# c_global=global_contrast(image)
# c_local=local_contrast(image)
# print(f"globalny: {c_global}")
# print(f"lokalny  {c_local}")


# Obliczenia
c_global1 = global_contrast(image1)
c_local1 = local_contrast(image1)

c_global2 = global_contrast(image2)
c_local2 = local_contrast(image2)

c_global3 = global_contrast(image3)
c_local3 = local_contrast(image3)
# Wyświetlenie wyników


print("Kontrast globalny:")
print(f"Mucha A: {c_global1}")
print(f"Mucha B: {c_global2}")
print(f"Mucha C: {c_global3}")

print("Kontrast lokalny:")
print(f"Mucha A: {c_local1}")
print(f"Mucha B: {c_local2}")
print(f"Mucha C: {c_local3}")




