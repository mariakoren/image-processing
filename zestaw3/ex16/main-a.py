import numpy as np
import cv2
import matplotlib.pyplot as plt

# Wczytaj obraz w skali szarości
# image_path = 'AlbertEinstein-modified.png'
image_path = 'output_image.png'

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Sprawdzenie, czy obraz został poprawnie wczytany
if image is None:
    raise FileNotFoundError(f"Nie można znaleźć obrazu: {image_path}")

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

# Tworzymy wykres z 8 płaszczyznami bitowymi
plt.figure(figsize=(10, 10))
for i in range(8):
    bit_plane = extract_bit_plane(image, i)
    plt.subplot(3, 3, i+1)
    plt.imshow(bit_plane, cmap='gray')
    plt.title(f'Bit Plane {i}')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Funkcja do odczytania ukrytej informacji
def extract_hidden_message(image):
    """
    Ekstrahuje ukrytą wiadomość (tekst) w pierwszej płaszczyźnie bitowej.
    :param image: obraz wejściowy w skali szarości.
    :return: ukryty tekst.
    """
    # Wyodrębniamy pierwszą płaszczyznę bitową (najniższy bit)
    bit_plane = extract_bit_plane(image, 0)
    
    # Przekształcamy na wartości binarne
    binary_message = ""
    for row in bit_plane:
        for pixel in row:
            # Dodajemy wartość bitową do wiadomości
            binary_message += '1' if pixel > 127 else '0'
    
    # Konwertujemy z binarnego na tekst
    message = ""
    for i in range(0, len(binary_message), 8):
        byte = binary_message[i:i+8]
        if len(byte) == 8:
            message += chr(int(byte, 2))
    
    return message

# # Odczytanie ukrytej wiadomości
hidden_message = extract_hidden_message(image)
print("Ukryta wiadomość:", hidden_message)
