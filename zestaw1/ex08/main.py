import numpy as np
from PIL import Image

def floyd_steinberg_dithering(image, palette):
    """
    Zastosowanie algorytmu Floyd-Steinberga do ditheringu.
    
    :param image: Obraz wejściowy w skali szarości jako macierz numpy.
    :param palette: Lista wartości docelowych palety.
    :return: Przekształcony obraz jako macierz numpy.
    """
    img = image.copy()
    h, w = img.shape
    for y in range(h):
        for x in range(w):
            old_pixel = img[y, x]
            # Znajdź najbliższą wartość w palecie
            new_pixel = min(palette, key=lambda v: abs(v - old_pixel))
            img[y, x] = new_pixel
            error = old_pixel - new_pixel
            
            # Propaguj błąd na sąsiednie piksele
            if x + 1 < w:
                img[y, x + 1] += error * 7 / 16
            if y + 1 < h:
                if x > 0:
                    img[y + 1, x - 1] += error * 3 / 16
                img[y + 1, x] += error * 5 / 16
                if x + 1 < w:
                    img[y + 1, x + 1] += error * 1 / 16
    
    return np.clip(img, 0, 255).astype(np.uint8)

def dithering_to_1bit(image, threshold=39):
    """
    Redukcja obrazu do 1-bitowej palety z progiem T.
    
    :param image: Obraz wejściowy w skali szarości jako macierz numpy.
    :param threshold: Próg binarnego podziału.
    :return: Przekształcony obraz jako macierz numpy.
    """
    binary_palette = [0, 255]
    return floyd_steinberg_dithering(image, binary_palette)

def apply_thresholds(value):
    """
    Funkcja przypisująca wartość na podstawie progów z diagramu.
    """
    if value < 20:
        return 0
    elif value < 40:
        return 64
    elif value < 60:
        return 128
    elif value < 120:
        return 192
    else:
        return 255

def dithering_with_custom_thresholds(image):
    """
    Zastosowanie algorytmu Floyd-Steinberga do ditheringu z progami z diagramu.
    
    :param image: Obraz wejściowy w skali szarości jako macierz numpy.
    :return: Przekształcony obraz jako macierz numpy.
    """
    img = image.copy()
    h, w = img.shape
    for y in range(h):
        for x in range(w):
            old_pixel = img[y, x]
            new_pixel = apply_thresholds(old_pixel)
            img[y, x] = new_pixel
            error = old_pixel - new_pixel
            
            # Propaguj błąd na sąsiednie piksele
            if x + 1 < w:
                img[y, x + 1] += error * 7 / 16
            if y + 1 < h:
                if x > 0:
                    img[y + 1, x - 1] += error * 3 / 16
                img[y + 1, x] += error * 5 / 16
                if x + 1 < w:
                    img[y + 1, x + 1] += error * 1 / 16
    
    return np.clip(img, 0, 255).astype(np.uint8)

# Wczytaj obraz i wykonaj obie operacje ditheringu
if __name__ == "__main__":
    # Wczytaj obraz w skali szarości
    image_path = "stanczyk.png"  # Upewnij się, że plik znajduje się w tym katalogu
    image = Image.open(image_path).convert("L")
    image_np = np.array(image)

    # (a) Dithering do 1-bitowej palety
    result_1bit = dithering_to_1bit(image_np, threshold=39)
    Image.fromarray(result_1bit).save("stanczyk_1bit.png")

    # (b) Dithering do palety z progami zgodnymi z diagramem
    result_custom = dithering_with_custom_thresholds(image_np)
    Image.fromarray(result_custom).save("stanczyk_custom_thresholds.png")
