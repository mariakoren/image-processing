import numpy as np
import cv2
import matplotlib.pyplot as plt

# Funkcja do zmiany kontrastu na podstawie wartości punktów dla kanału RGB
def adjust_contrast(channel_values, contrast_function):
    """
    Funkcja modyfikuje wartości kanału na podstawie zadanej funkcji kontrastu.
    
    :param channel_values: Tablica wartości kanału (0-255)
    :param contrast_function: Funkcja kontrastu, która modyfikuje wartości.
    :return: Zmieniona tablica wartości kanału.
    """
    return np.vectorize(contrast_function)(channel_values)

# Funkcje kontrastu dla każdego z kanałów (niebieski, zielony, czerwony)

# Niebieski: w punkcie 0 ma wartość 128, w 31 ma 255, potem spada do 0 w punkcie 159
def blue_contrast(x):
    if x <= 31:
        return 128 + (x / 31) * (255 - 128)  # rośnie od 128 do 255
    elif x <= 95:
        return 255  # stała wartość 255
    elif x < 159:
        return 255 - ((x - 95) / (159 - 95)) * 255  # spada od 255 do 0
    else:
        return 0  # pozostaje na 0

# Zielony: dla wartości od 0 do 31 ma wartość 0, potem rośnie do 255 w 95, potem spada do 0 w 233
def green_contrast(x):
    if x <= 31:
        return 0  # stała wartość 0
    elif x <= 95:
        return (x - 31) / (95 - 31) * 255  # rośnie od 0 do 255
    elif x <= 159:
        return 255
    elif x < 223:
        return 255 - ((x - 159) / (223 - 159)) * 255  # spada od 255 do 0
    else:
        return 0  # pozostaje na 0

# Czerwony: do 95 ma wartość 0, potem rośnie do 255 w 159, pozostaje 255 do 223, potem spada do 128 w 255
def red_contrast(x):
    if x <= 95:
        return 0  # stała wartość 0
    elif x <= 159:
        return (x - 95) / (159 - 95) * 255  # rośnie od 0 do 255
    elif x <= 223:
        return 255  # stała wartość 255
    else:
        return 255 - ((x - 223) / (255 - 223)) * 127  # spada od 255 do 128

# Wczytaj obraz
image_path = 'czaszka-a.png'  # Zmień to na ścieżkę do swojego obrazu
image = cv2.imread(image_path)

# Przekształć obraz do przestrzeni RGB (OpenCV używa BGR)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Ustalanie kontrastów dla każdego kanału (Blue, Green, Red)
blue_channel = np.array([blue_contrast(x) for x in range(256)], dtype=np.uint8)
green_channel = np.array([green_contrast(x) for x in range(256)], dtype=np.uint8)
red_channel = np.array([red_contrast(x) for x in range(256)], dtype=np.uint8)

# Stworzenie nowego obrazu z zmienionymi wartościami kontrastu
image_rgb_new = image_rgb.copy()

# Modyfikacja kanałów RGB na podstawie funkcji kontrastu
image_rgb_new[:, :, 0] = red_channel[image_rgb[:, :, 2]]    # Kanał czerwony
image_rgb_new[:, :, 1] = green_channel[image_rgb[:, :, 1]]  # Kanał zielony
image_rgb_new[:, :, 2] = blue_channel[image_rgb[:, :, 0]]  # Kanał niebieski


# Wyświetlenie oryginalnego obrazu i zmodyfikowanego obrazu
plt.figure(figsize=(10, 5))

# Oryginalny obraz
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Oryginalny obraz")
plt.axis("off")

# Obraz po transformacji kontrastu
plt.subplot(1, 2, 2)
plt.imshow(image_rgb_new)
plt.title("Obraz po transformacji kontrastu")
plt.axis("off")

plt.show()
