from PIL import Image
import numpy as np

def extended_nearest_neighbor(image, new_width, new_height):
    image_array = np.array(image)
    original_height, original_width, _ = image_array.shape
    scaled_image_array = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    scale_x = original_width / new_width
    scale_y = original_height / new_height

    for i in range(new_height):
        for j in range(new_width):
            x1 = int(j * scale_x)
            y1 = int(i * scale_y)
            x2 = min(x1 + 1, original_width - 1)
            y2 = min(y1 + 1, original_height - 1)
            color1 = image_array[y1, x1]
            color2 = image_array[y2, x2]
            avg_color = np.mean([color1, color2], axis=0)
            scaled_image_array[i, j] = avg_color


    return Image.fromarray(scaled_image_array)


image_path = 'potworek.png' 
image = Image.open(image_path)
new_width = 600
new_height = 360

result_image = extended_nearest_neighbor(image, new_width, new_height)
result_image.show() 
result_image.save("potworek-b.png") 
