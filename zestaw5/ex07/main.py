import cv2
import numpy as np
import matplotlib.pyplot as plt

# Ładowanie obrazów
image_path = 'Webbs_First_Deep_Field.jpg'
template_path = 'wzorzecSMACS.jpg'

# Wczytanie obrazu i wzorca
image = cv2.imread(image_path)
template = cv2.imread(template_path)

# Konwersja obrazów na przestrzeń kolorów RGB (OpenCV domyślnie używa BGR)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)

# Funkcja do wykonania korelacji dla poszczególnych kanałów RGB
def correlation(image, template):
    correlations = []
    for i in range(3):  # Iteracja po kanałach R, G, B
        img_channel = image[:, :, i]
        tmpl_channel = template[:, :, i]
        
        # Obliczenie korelacji
        result = cv2.matchTemplate(img_channel, tmpl_channel, cv2.TM_CCOEFF_NORMED)
        correlations.append(result)
    return correlations

# Przeprowadzamy korelację dla każdego kanału RGB
correlations = correlation(image_rgb, template_rgb)

# Obliczanie współczynnika korelacji dla każdego kanału
correlation_scores = [np.max(correlation) for correlation in correlations]
print(f'Współczynniki korelacji dla kanałów RGB: {correlation_scores}')

# Znalezienie pięciu najbardziej prawdopodobnych pozycji wzorca
def find_best_match_positions(correlation_result):
    # Zwrócenie pozycji, w których korelacja jest najwyższa
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(correlation_result)
    return max_loc, max_val

# Zbieranie najlepszych pozycji dla każdego kanału
best_positions = []
for correlation in correlations:
    best_position, max_value = find_best_match_positions(correlation)
    best_positions.append((best_position, max_value))

# Wyświetlenie wyników
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Oryginalny obraz
axs[0].imshow(image_rgb)
axs[0].set_title('Oryginalny obraz')
axs[0].axis('off')

# Nałożenie pozycji wzorca na obraz
for position, _ in best_positions:
    top_left = position
    bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
    cv2.rectangle(image_rgb, top_left, bottom_right, (255, 0, 0), 3)

axs[1].imshow(image_rgb)
axs[1].set_title('Znalezione pozycje wzorca')
axs[1].axis('off')

plt.show()

# Zapisz wynikowy obraz
cv2.imwrite('matched_image.jpg', cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
