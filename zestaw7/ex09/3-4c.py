import cv2
import numpy as np
from skimage import io, filters
 
# Load the image
image = io.imread('Kopernik.png')
 
# Check the number of channels in the image
if len(image.shape) == 3 and image.shape[2] == 3:
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
else:
    gray_image = image
 

 
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
io.imsave('Kopernik-3.png', local_thresh)