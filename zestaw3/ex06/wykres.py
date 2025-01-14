import numpy as np
import matplotlib.pyplot as plt

# Definicja funkcji
def blue_contrast(x):
    if x < 31:
        return 128 + (x / 31) * (255 - 128)  # rośnie od 128 do 255
    elif x <= 95:
        return 255  # stała wartość 255
    elif x < 159:
        return 255 - ((x - 95) / (159 - 95)) * 255  # spada od 255 do 0
    else:
        return 0  # pozostaje na 0

def green_contrast(x):
    if x < 31:
        return 0  # stała wartość 0
    elif x <= 95:
        return (x - 31) / (95 - 31) * 255  # rośnie od 0 do 255
    elif x <= 159:
        return 255
    elif x < 223:
        return 255 - ((x - 159) / (223 - 159)) * 255  # spada od 255 do 0
    else:
        return 0  # pozostaje na 0

def red_contrast(x):
    if x < 95:
        return 0  # stała wartość 0
    elif x <= 159:
        return (x - 95) / (159 - 95) * 255  # rośnie od 0 do 255
    elif x <= 223:
        return 255  # stała wartość 255
    else:
        return 255 - ((x - 223) / (255 - 223)) * 127


x_values = np.linspace(0, 255, 256)
blue_values = np.array([blue_contrast(x) for x in x_values])
green_values = np.array([green_contrast(x) for x in x_values])
red_values = np.array([red_contrast(x) for x in x_values])

# Tworzenie wykresu
plt.figure(figsize=(10, 6))
plt.plot(x_values, red_values, label='Czerwony', color='red')
plt.plot(x_values, green_values, label='Zielony', color='green')
plt.plot(x_values, blue_values, label='Niebieski', color='blue')

plt.title("Funkcje kontrastu kolorów")
plt.xlabel("Wartości x")
plt.ylabel("Intensywność koloru")
plt.legend()
plt.grid(True)
plt.show()
