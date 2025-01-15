import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import io

# Załaduj obraz
image = io.imread('smok.png', as_gray=True)

# Zdefiniuj filtr uśredniający 3x3 (h1)
h1 = np.ones((3, 3)) / 9  # Filtr uśredniający 3x3

# Zastosuj splot (konwolucję) obrazu z filtrem h1
g1 = ndimage.convolve(image, h1)

# Wyświetl obraz g1
plt.imshow(g1, cmap='gray')
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Usuń marginesy
plt.savefig('g1.png', bbox_inches='tight', pad_inches=0)
plt.close()

# Zdefiniuj filtr gradientowy h2 (różnicowy w kierunku pionowym)
h2 = np.array([[-1, 0, 1]])

# Zastosuj splot (konwolucję) obrazu g1 z filtrem h2
g2 = ndimage.convolve(g1, h2)

# Wyświetl obraz g2
plt.imshow(g2, cmap='gray')
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Usuń marginesy
plt.savefig('g2.png', bbox_inches='tight', pad_inches=0)
plt.close()

# Połącz filtry h1 i h2 przez splot, aby uzyskać h3
h3 = ndimage.convolve(h1, h2)

# Wyświetl filtr h3
plt.imshow(h3, cmap='gray')
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Usuń marginesy
plt.savefig('h3.png', bbox_inches='tight', pad_inches=0)
plt.close()

# Zastosuj splot (konwolucję) obrazu g1 z filtrem h3
g3 = ndimage.convolve(g1, h3)

# Wyświetl obraz g3
plt.imshow(g3, cmap='gray')
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Usuń marginesy
plt.savefig('g3.png', bbox_inches='tight', pad_inches=0)
plt.close()
