import numpy as np
from PIL import Image

image_path="PlytkaFresnela.png"
original_image=Image.open(image_path).convert("L")
original_array = np.array(original_image)

for i in range(5, 50, 5):
    simpled_array=original_array[::i, ::i]
    Image.fromarray(simpled_array).save(f"line/plytka-line-{i}.png")