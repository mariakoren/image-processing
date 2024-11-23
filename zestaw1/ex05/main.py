from PIL import Image, ImageDraw
import random

def apply_pointillism(image_path, dot_size=5, num_dots=10000):
    img = Image.open(image_path)
    width, height = img.size
    draw = ImageDraw.Draw(img)
    for _ in range(num_dots):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        pixel_color = img.getpixel((x, y))
        draw.ellipse(
            (x - dot_size // 2, y - dot_size // 2, x + dot_size // 2, y + dot_size // 2),
            fill=pixel_color
        )
    return img

image_path = 'q.png'

for _ in range(5):
    dot_size = random.randint(1, 20)
    num_dots = random.randint(1, 20)
    output_image = apply_pointillism(image_path, dot_size=dot_size, num_dots=10000*num_dots)
    output_image.show()
    output_image.save(f'results/q-{dot_size}-{num_dots}.png')
