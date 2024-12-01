import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from skimage import io, color

def process_image(image_path, size_ratios=[0.1, 0.2]):
    """
    Przetwarzanie obrazu: aproksymacja widma i rekonstrukcja obrazów.
    
    Parameters:
        image_path (str): Ścieżka do obrazu.
        size_ratios (list): Lista proporcji widma do wykorzystania (np. [0.1, 0.2]).
    """
    # 1. Wczytanie obrazu i konwersja na skalę szarości (jeśli wymagane)
    image = io.imread(image_path)
    if len(image.shape) == 3:  # Jeśli obraz jest w kolorze
        image = color.rgb2gray(image)
    image = image.astype(np.float32) / 255.0  # Normalizacja do zakresu [0, 1]
    
    # 2. Transformacja Fouriera
    fft_image = fft2(image)
    fft_shifted = fftshift(fft_image)  # Przesunięcie widma do centrum
    
    # 3. Aproksymacja widma
    def apply_fourier_mask(fft_shifted, size_ratio):
        rows, cols = fft_shifted.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros_like(fft_shifted, dtype=np.bool_)
        half_size = int(size_ratio * min(rows, cols) // 2)
        mask[crow - half_size : crow + half_size, ccol - half_size : ccol + half_size] = 1
        return fft_shifted * mask

    reconstructions = []
    profiles = []
    for size_ratio in size_ratios:
        fft_limited = apply_fourier_mask(fft_shifted, size_ratio)
        reconstructed = np.abs(ifft2(ifftshift(fft_limited)))
        reconstructions.append(reconstructed)
        profiles.append(reconstructed[image.shape[0] // 2, :])  # Profil środkowy

    # 4. Profil liniowy oryginalnego obrazu
    original_profile = image[image.shape[0] // 2, :]

    # 5. Wizualizacja wyników
    fig, axs = plt.subplots(len(size_ratios) + 1, 3, figsize=(12, 6 * (len(size_ratios) + 1)))

    # Oryginał
    axs[0, 0].imshow(image, cmap='gray')
    axs[0, 0].set_title("Oryginalny obraz")
    axs[0, 0].axis('off')

    axs[0, 1].imshow(np.log(1 + np.abs(fft_shifted)), cmap='gray')
    axs[0, 1].set_title("Widmo oryginalne")
    axs[0, 1].axis('off')

    axs[0, 2].plot(original_profile)
    axs[0, 2].set_title("Profil liniowy (oryginał)")

    for i, (reconstructed, size_ratio) in enumerate(zip(reconstructions, size_ratios)):
        axs[i + 1, 0].imshow(reconstructed, cmap='gray')
        axs[i + 1, 0].set_title(f"Rekonstrukcja {int(size_ratio * 100)}% widma")
        axs[i + 1, 0].axis('off')

        axs[i + 1, 1].imshow(np.log(1 + np.abs(apply_fourier_mask(fft_shifted, size_ratio))), cmap='gray')
        axs[i + 1, 1].set_title(f"Widmo {int(size_ratio * 100)}%")
        axs[i + 1, 1].axis('off')

        axs[i + 1, 2].plot(profiles[i])
        axs[i + 1, 2].set_title(f"Profil liniowy ({int(size_ratio * 100)}%)")

    plt.tight_layout()
    # plt.show()
    plt.savefig("results.png")

process_image("czarne.png", size_ratios=[0.1, 0.2])
