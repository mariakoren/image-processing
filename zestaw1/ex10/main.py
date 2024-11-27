import numpy as np
from PIL import Image

def apply_dithering(image, dithering_matrix, levels):
    matrix_size = dithering_matrix.shape[0]
    normalized_matrix = dithering_matrix / (matrix_size**2 - 1)
    img_array = np.array(image, dtype=np.float32) / 255.0 
    thresholds = np.linspace(0, 1, levels, endpoint=False)
    height, width = img_array.shape
    output = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            threshold = normalized_matrix[y % matrix_size, x % matrix_size]
            pixel_value = img_array[y, x]
            quantized_value = int((pixel_value + threshold) * (levels - 1))
            output[y, x] = min(quantized_value, levels - 1) * (255 // (levels - 1))
    
    return Image.fromarray(output, mode='L')


dithering_matrix = np.array([
    [6, 14, 2, 8],
    [4,  0, 10, 11],
    [12, 15, 5, 1],
    [9,  3, 13, 7]
])

input_image = Image.open("stanczyk.png").convert("L")
levels = 16 
output_image = apply_dithering(input_image, dithering_matrix, levels)
output_image.save("stanczyk_d.png")
output_image.show()
