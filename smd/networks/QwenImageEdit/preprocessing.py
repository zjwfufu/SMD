"""Image sizes used by the edit model's VLM and VAE branches."""
import math

import numpy as np
from PIL import Image
import torch

CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 1024 * 1024


def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    return max(32, round(width / 32) * 32), max(32, round(height / 32) * 32)


def resize_image(image, target_area):
    size = calculate_dimensions(target_area, image.width / image.height)
    return image.convert('RGB').resize(size, Image.Resampling.BICUBIC)


def source_pixels(image):
    image = resize_image(image, VAE_IMAGE_SIZE)
    return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 127.5 - 1
