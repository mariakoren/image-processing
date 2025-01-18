import numpy as np
from PIL import Image
from scipy.ndimage import interpolation

# Wczytanie obrazu
image_path = "Kopernik.png"
image = Image.open(image_path).convert('L')  # Konwersja do skali szarości
image_array = np.array(image)

# Wymiary obrazu
height, width = image_array.shape

# Wymiary podregionu
subregion_size = 23

# Funkcja obliczająca próg na podstawie histogramu podregionu
def calculate_local_threshold(subregion):
    # Możemy przyjąć jako próg średnią wartość pikseli w podregionie
    return np.mean(subregion)

# Tworzenie tablicy progów
thresholds = np.zeros_like(image_array, dtype=float)

# Wyznaczanie progów lokalnych
for i in range(0, height - subregion_size, subregion_size):
    for j in range(0, width - subregion_size, subregion_size):
        # Wydzielanie podregionu
        subregion = image_array[i:i + subregion_size, j:j + subregion_size]
        # Obliczanie progu dla podregionu
        local_threshold = calculate_local_threshold(subregion)
        # Ustawienie wartości progowej w odpowiednich miejscach w obrazie
        thresholds[i:i + subregion_size, j:j + subregion_size] = local_threshold

# Interpolacja biliniowa wartości progowych
# Rozciąganie progów na cały obraz za pomocą interpolacji biliniowej
threshold_interpolated = interpolation.zoom(thresholds, (height / thresholds.shape[0], width / thresholds.shape[1]), order=1)

# Segmentacja obrazu na podstawie wyznaczonego progu
segmented_image_local = image_array > threshold_interpolated

# Progowanie globalne - obliczenie progu globalnego (np. średnia intensywność obrazu)
global_threshold = np.mean(image_array)
segmented_image_global = image_array > global_threshold

# Konwersja do obrazów (wartości 255 dla białych pikseli, 0 dla czarnych)
segmented_image_local = (segmented_image_local * 255).astype(np.uint8)
segmented_image_global = (segmented_image_global * 255).astype(np.uint8)

# Konwersja z powrotem do obrazów PIL
segmented_image_local = Image.fromarray(segmented_image_local)
segmented_image_global = Image.fromarray(segmented_image_global)

# Wyświetlenie wyników
segmented_image_local.show(title="Progowanie Zmiennym Progiem")
segmented_image_global.show(title="Progowanie Globalne")

# Zapisanie wyników do plików
segmented_image_local.save("segmented_Kopernik_local.png")
segmented_image_global.save("segmented_Kopernik_global.png")
