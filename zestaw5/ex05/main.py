import cv2
import numpy as np
 
# Wczytanie obrazu
image = cv2.imread("meduza.png", cv2.IMREAD_GRAYSCALE)
 
# Filtr medianowy
median_filtered = cv2.medianBlur(image, 3)
 
 
# Filtr średniozakresowy
def mid_range_filter(img, kernel_size=3):
    padded_img = np.pad(img, (kernel_size // 2, kernel_size // 2), mode="reflect")
    output = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            kernel = padded_img[i : i + kernel_size, j : j + kernel_size].astype(np.float32)
            min_val = np.min(kernel)
            max_val = np.max(kernel)
            output[i, j] = (min_val + max_val) / 2.0
    return np.clip(output, 0, 255).astype(np.uint8)
 
 
 
mid_range_filtered = mid_range_filter(image)
 
 
# Filtr średniej uciętej (k-trimmed-mean Filter)
def k_trimmed_mean_filter(img, k=2, kernel_size=3):
    padded_img = np.pad(img, (kernel_size // 2, kernel_size // 2), mode="reflect")
    output = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            kernel = np.sort(
                padded_img[i : i + kernel_size, j : j + kernel_size].flatten()
            )
            output[i, j] = np.mean(kernel[k:-k])
    return output
 
 
k_trimmed_mean_filtered = k_trimmed_mean_filter(image, k=2)
 
 
# Filtr k-Nearest Neighbor
def k_nearest_neighbor_filter(img, k=6, kernel_size=3):
    padded_img = np.pad(img, (kernel_size // 2, kernel_size // 2), mode="reflect")
    output = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            kernel = np.sort(
                padded_img[i : i + kernel_size, j : j + kernel_size].flatten()
            )
            output[i, j] = np.mean(kernel[:k])
    return output
 
 
k_nearest_neighbor_filtered = k_nearest_neighbor_filter(image, k=6)
 
 
# Filtr Symmetric Nearest Neighbor
def symmetric_nearest_neighbor_filter(img, kernel_size=3):
    padded_img = np.pad(img, (kernel_size // 2, kernel_size // 2), mode="reflect")
    output = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            kernel = padded_img[i : i + kernel_size, j : j + kernel_size].flatten()
            mean = np.mean(kernel)
            diffs = np.abs(kernel - mean)
            output[i, j] = np.mean(kernel[np.argsort(diffs)[:kernel_size]])
    return output
 
 
symmetric_nearest_neighbor_filtered = symmetric_nearest_neighbor_filter(image)
 
# Zapisz wyniki
cv2.imwrite("meduza-b.png", mid_range_filtered)
cv2.imwrite("meduza-c.png", k_trimmed_mean_filtered)
cv2.imwrite(
    "meduza-d.png", k_nearest_neighbor_filtered
)
cv2.imwrite(
    "meduza-e.png",
    symmetric_nearest_neighbor_filtered,
)
 
print("Done.")