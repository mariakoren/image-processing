import cv2
import numpy as np
from skimage import io, filters
 
# Load the image
image = io.imread('roze.png')
 
# Check the number of channels in the image
if len(image.shape) == 3 and image.shape[2] == 3:
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
else:
    gray_image = image
 
# (a)
_, global_thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
 
# (b)
def iterative_three_class_otsu(image, delta=2):
    prev_threshold = 0
    while True:
        t1, t2 = filters.threshold_multiotsu(image, classes=3)
        new_threshold = (t1 + t2) / 2
        if abs(new_threshold - prev_threshold) < delta:
            break
        prev_threshold = new_threshold
    return cv2.threshold(image, new_threshold, 255, cv2.THRESH_BINARY)[1]
 
iterative_thresh = iterative_three_class_otsu(gray_image)
 
# (c)
def local_otsu_threshold(image, window_size=11):
    pad_size = window_size // 2
    padded_image = np.pad(image, pad_size, mode='reflect')
    local_thresh_image = np.zeros_like(image)
 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            local_window = padded_image[i:i + window_size, j:j + window_size]
 
            if np.all(local_window == local_window[0, 0]):
                local_thresh = 127.5
            else:
                local_thresh = filters.threshold_otsu(local_window)
 
 
            local_thresh_image[i, j] = 255 if image[i, j] > local_thresh else 0
 
    return local_thresh_image
 
 
local_thresh = local_otsu_threshold(gray_image)
io.imsave('zadanie_4_c.png', local_thresh)
io.imsave('zadanie_4_a.png', global_thresh)
io.imsave('zadanie_4_b.png', iterative_thresh)