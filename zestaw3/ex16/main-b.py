import numpy as np
import cv2
import matplotlib.pyplot as plt

# Funkcja do wyodrębniania płaszczyzn bitowych
def extract_bit_plane(image, bit_position):
    """
    Ekstrahuje bitową płaszczyznę obrazu na określonej pozycji bitowej.
    :param image: obraz wejściowy w skali szarości.
    :param bit_position: pozycja bitu (0 - 7), od 0 do 7.
    :return: nowy obraz z wyodrębnioną płaszczyzną bitową.
    """
    # Tworzymy maskę dla konkretnego bitu
    mask = 1 << bit_position
    # Wykonujemy operację AND z maską
    bit_plane = np.bitwise_and(image, mask)
    # Przesuwamy bity do prawej, aby uzyskać wyraźny obraz
    bit_plane = bit_plane >> bit_position
    return bit_plane * 255  # Skalujemy do zakresu 0-255

# Funkcja do ukrywania obrazu w pierwszej płaszczyźnie bitowej
def hide_image_in_bit_plane(image, hidden_image):
    """
    Ukrywa obraz w pierwszej płaszczyźnie bitowej obrazu wejściowego.
    :param image: obraz wejściowy w skali szarości.
    :param hidden_image: obraz, który chcemy ukryć.
    :return: obraz z ukrytą informacją.
    """
    # Upewnijmy się, że obrazy mają ten sam rozmiar
    hidden_image_resized = cv2.resize(hidden_image, (image.shape[1], image.shape[0]))
    
    # Zamień obrazy na binarne (pierwsza płaszczyzna bitowa)
    hidden_image_bin = hidden_image_resized // 128  # Skala do 0 lub 1
    
    # Zrób kopię obrazu wejściowego
    result_image = image.copy()
    
    # Zastępujemy pierwszą płaszczyznę bitową obrazu wejściowego obrazem ukrytym
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Wstawiamy ukryty bit w pierwszą płaszczyznę
            result_image[i, j] = (result_image[i, j] & 0b11111110) | hidden_image_bin[i, j]
    
    return result_image

# Wczytaj obraz wejściowy (w skali szarości)
image_path = 'AlbertEinstein-modified.png'
input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Wczytaj obraz, który chcemy ukryć
hidden_image_path = 'image.png'
hidden_image = cv2.imread(hidden_image_path, cv2.IMREAD_GRAYSCALE)

# Sprawdzenie, czy obrazy zostały poprawnie wczytane
if input_image is None:
    raise FileNotFoundError(f"Nie można znaleźć obrazu wejściowego: {image_path}")
if hidden_image is None:
    raise FileNotFoundError(f"Nie można znaleźć obrazu do ukrycia: {hidden_image_path}")

# Ukrywanie obrazu w pierwszej płaszczyźnie bitowej
output_image = hide_image_in_bit_plane(input_image, hidden_image)

# Zapisz obraz wyjściowy
output_image_path = 'output_image.png'
cv2.imwrite(output_image_path, output_image)

# Wyświetl obraz wyjściowy
plt.imshow(output_image, cmap='gray')
plt.title("Obraz z ukrytą informacją")
plt.axis('off')
plt.show()

print(f"Obraz z ukrytą informacją zapisany jako: {output_image_path}")
