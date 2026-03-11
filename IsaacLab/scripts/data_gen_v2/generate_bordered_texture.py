import numpy as np
from PIL import Image

size = 512
border = 40  # Thicker border
img = np.zeros((size, size, 3), dtype=np.uint8)

# Center color: vivid but light blue
img[:, :] = [80, 140, 255]

# Pure black borders
img[:border, :, :] = [0, 0, 0]
img[-border:, :, :] = [0, 0, 0]
img[:, :border, :] = [0, 0, 0]
img[:, -border:, :] = [0, 0, 0]

Image.fromarray(img).save("/home/user/桌面/image2bricks/image_bricks/assets/dataset_v2/bordered_blue.png")
print("Saved bordered_blue.png")
