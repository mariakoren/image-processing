import cv2
import numpy as np
from skimage.filters import threshold_otsu
from scipy.ndimage import generic_filter

def global_otsu_threshold(image):
    """Metoda globalnego progowania Otsu."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

def iterative_three_class_thresholding(image, delta=2):
    """Iteracyjne trójklasowe progowanie obrazu w oparciu o metodę Otsu."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    T_prev = threshold_otsu(gray_image)
    
    while True:
        # Podział na klasy
        lower_class = gray_image[gray_image < T_prev]
        upper_class = gray_image[gray_image >= T_prev]

        # Oblicz nowe progi
        T_lower = np.mean(lower_class) if len(lower_class) > 0 else 0
        T_upper = np.mean(upper_class) if len(upper_class) > 0 else 255

        T_new = (T_lower + T_upper) / 2

        # Warunek stopu
        if abs(T_new - T_prev) < delta:
            break

        T_prev = T_new

    # Tworzenie obrazu binarnego
    binary_image = np.zeros_like(gray_image)
    binary_image[gray_image < T_lower] = 85
    binary_image[(gray_image >= T_lower) & (gray_image < T_upper)] = 170
    binary_image[gray_image >= T_upper] = 255

    return binary_image

def local_otsu_threshold(image, window_size=11):
    """Progowanie lokalne z wartością progową obliczoną metodą Otsu w sąsiedztwie."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def local_otsu(block):
        """Oblicz próg Otsu dla lokalnego sąsiedztwa."""
        return threshold_otsu(block.reshape(-1))

    # Rozszerzenie obrazu o symetryczne odbicie
    padded_image = np.pad(gray_image, pad_width=window_size // 2, mode='reflect')
    
    # Filtracja lokalna
    local_thresholds = generic_filter(padded_image, local_otsu, size=(window_size, window_size))
    
    # Przycięcie do rozmiaru obrazu wejściowego (400x400)
    h, w = gray_image.shape
    local_thresholds = local_thresholds[:h, :w]
    
    # Tworzenie obrazu binarnego
    local_binary_image = (gray_image > local_thresholds).astype(np.uint8) * 255

    return local_binary_image

# Wczytanie obrazu
image = cv2.imread("roze.png")

# (a) Progowanie globalne
binary_global = global_otsu_threshold(image)
cv2.imwrite("binary_global.png", binary_global)

# (b) Iteracyjne trójklasowe progowanie
binary_iterative = iterative_three_class_thresholding(image)
cv2.imwrite("binary_iterative.png", binary_iterative)

# (c) Progowanie lokalne
binary_local = local_otsu_threshold(image, window_size=11)
cv2.imwrite("binary_local.png", binary_local)
