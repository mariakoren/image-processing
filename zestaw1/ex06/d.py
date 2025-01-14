from PIL import Image
import numpy as np

# def average_of_extremes(image_path, new_width, new_height):
#     img = Image.open(image_path)
#     img = img.convert("RGB")
#     img_array = np.array(img)
#     old_height, old_width, _ = img_array.shape
#     scale_x = old_width / new_width
#     scale_y = old_height / new_height
#     new_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    
#     for y in range(new_height):
#         for x in range(new_width):
#             old_x = int(x * scale_x)
#             old_y = int(y * scale_y)
#             top_left = img_array[old_y, old_x]
#             top_right = img_array[old_y, min(old_x + 1, old_width - 1)]
#             bottom_left = img_array[min(old_y + 1, old_height - 1), old_x]
#             bottom_right = img_array[min(old_y + 1, old_height - 1), min(old_x + 1, old_width - 1)]
#             pixels = [top_left, top_right, bottom_left, bottom_right]
#             pixel_values = [np.mean(pixel) for pixel in pixels]
#             min_val = min(pixel_values)
#             max_val = max(pixel_values)
#             avg_val = (min_val + max_val) / 2
#             new_pixel = np.array([avg_val, avg_val, avg_val], dtype=np.uint8)
#             new_image[y, x] = new_pixel
    
#     result_img = Image.fromarray(new_image)
#     return result_img

def custom_interpolation(image, new_width, new_height):
    old_width, old_height = image.size
    old_pixels = np.array(image)
    new_pixels = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for y in range(new_height):
        for x in range(new_width):
            # Map new image coordinates to old image coordinates
            old_x = x * old_width / new_width
            old_y = y * old_height / new_height
            x1, y1 = int(old_x), int(old_y)
            x2, y2 = min(x1 + 1, old_width - 1), min(y1 + 1, old_height - 1)

            # Get the four neighboring pixels
            neighbors = [
                old_pixels[y1, x1],
                old_pixels[y1, x2],
                old_pixels[y2, x1],
                old_pixels[y2, x2]
            ]
            
            # Calculate the average color of the neighbors
            avg_color = np.mean(neighbors, axis=0)
            new_pixels[y, x] = np.clip(avg_color, 0, 255)

    return Image.fromarray(new_pixels)

# Open the image
image_path = 'potworek.png'
image = Image.open(image_path)

# Set the new dimensions
new_width = 600
new_height = 360

# Apply custom interpolation
result_image = custom_interpolation(image, new_width, new_height)

# Show and save the result
result_image.show()
result_image.save("potworek-d.png")
