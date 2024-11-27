from PIL import Image
import numpy as np

# Matryca Bayera 8×8
BAYER_MATRIX_8x8 = np.array([
    [0, 32, 8, 40, 2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21]
]) / 64.0  # Normalizacja do zakresu 0-1

def dithering(image_path, output_path, levels, palette=None):
    # Wczytaj obraz w trybie grayscale
    img = Image.open(image_path).convert('L')
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalizacja do zakresu 0-1

    # Rozmiar obrazu
    h, w = img_array.shape

    # Rozszerzenie matrycy Bayera do rozmiaru obrazu
    bayer_matrix = np.tile(BAYER_MATRIX_8x8, (h // 8 + 1, w // 8 + 1))[:h, :w]

    # Skalowanie matrycy Bayera do liczby poziomów
    dither_thresholds = bayer_matrix / levels

    if palette is None:
        # Standardowy dithering do 1-bitowego obrazu (czarno-białego)
        dithered_img = (img_array > dither_thresholds).astype(np.uint8) * 255
    else:
        # Dithering z redukcją do określonej palety
        scaled_image = np.round(img_array * (len(palette) - 1))  # Skalowanie obrazów
        dithered_img = np.zeros_like(img_array)
        for i, value in enumerate(palette):
            dithered_img += (scaled_image == i).astype(np.uint8) * value
    
    # Konwersja do obrazu i zapis
    output_image = Image.fromarray(dithered_img.astype(np.uint8))
    output_image.save(output_path)

# Użycie funkcji
image_path = 'stanczyk.png'  # Podaj ścieżkę do swojego obrazu
output_path_bw = 'output_bw.png'  # Ścieżka do wyjściowego obrazu czarno-białego
output_path_palette = 'output_palette.png'  # Ścieżka do wyjściowego obrazu z paletą

# Redukcja do 1-bitowego obrazu (czarno-białego)
dithering(image_path, output_path_bw, levels=2)

# Redukcja do obrazu z paletą {50, 100, 150, 200}
dithering(image_path, output_path_palette, levels=4, palette=[50, 100, 150, 200])
