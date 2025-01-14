import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Funkcja do wczytania obrazu i konwersji na tablicę numpy
def load_image(image_path):
    image = Image.open(image_path)
    image = image.convert('L')  # Konwertujemy na obraz w odcieniach szarości (8-bitowy)
    return np.array(image, dtype=float)

# Funkcja do obliczenia g3 = sqrt(g1^2 + g2^2)
def compute_g3(g1, g2):
    return np.sqrt(g1**2 + g2**2)

# Wczytanie obrazów g1 i g2
# g1 = load_image('Escher-8bit-a.png') 
# g2 = load_image('Escher-8bit-b.png') 

g1 = load_image('Escher-32bit-a.png') 
g2 = load_image('Escher-32bit-b.png') 

# Obliczenie obrazu g3
g3 = compute_g3(g1, g2)

# Skala wynikowego obrazu (konwersja do zakresu 0-255)
g3 = np.clip(g3, 0, 255).astype(np.uint8)

# Zapisanie obrazu g3
output_image = Image.fromarray(g3)
# output_image.save('g3-8bit.png')
output_image.save('g3-32bit.png')


# Wyświetlenie obrazu g3
plt.imshow(g3, cmap='gray')
plt.title('Obraz g3')
plt.axis('off')
plt.show()
