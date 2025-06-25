import numpy as np
from PIL import Image

"""
Source: https://github.com/mohits2806/TumorScope
"""

def weight_init(image):
    # img = image.convert('RGB').resize((200, 200))
    img = Image.fromarray(np.uint8(image * 255)).convert('RGB').resize((200, 200))
    img_array = np.array(img) / 255.0
    return img_array.reshape(1, 200, 200, 3)