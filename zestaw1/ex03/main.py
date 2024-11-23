from PIL import Image

def convert_to_grayscale(image_path, output_path):
    img = Image.open(image_path)
    img = img.convert('RGB')
    grayscale_img = img.convert('L', (0.299, 0.587, 0.114, 0))
    grayscale_img.save(output_path)

input_image = 'kolory.png'
output_image = 'szare-wagi.png'

convert_to_grayscale(input_image, output_image)