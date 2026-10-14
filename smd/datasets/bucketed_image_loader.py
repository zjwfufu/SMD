# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aspect-ratio crop utilities for image datasets."""
import math
from typing import List, Tuple
from PIL import Image

def generate_crop_size_list(
    base_size: int = 1024,
    patch_size: int = 32,
    max_ratio: float = 4.0,
) -> List[Tuple[int, int]]:
    """Generate constant-area crop sizes for variable-resolution training.

    All buckets have approximately the same pixel area (≈ base_size²) but
    different aspect ratios, from ``max_ratio:1`` to ``1:max_ratio``.  Both
    dimensions are multiples of ``patch_size``.

    Args:
        base_size: Target side length for the 1:1 bucket.  Total pixel area
            ≈ base_size².
        patch_size: Granularity — every bucket dimension is a multiple of this.
            Must be a multiple of 16 (VAE + Flux 2×2 packing requirement).
        max_ratio: Maximum allowed aspect ratio (long/short side).

    Returns:
        List of ``(H, W)`` tuples sorted by descending height.

    Example::

        >>> sizes = generate_crop_size_list(1024, 32, 4.0)
        >>> len(sizes)    # 97 buckets
        >>> sizes[0]      # (2048, 512)  — tallest
        >>> sizes[-1]     # (512, 2048)  — widest
    """
    assert patch_size % 16 == 0, f"patch_size must be a multiple of 16, got {patch_size}"
    assert max_ratio >= 1.0
    num_patches = round((base_size / patch_size) ** 2)
    crop_size_list: List[Tuple[int, int]] = []
    wp, hp = num_patches, 1
    while wp > 0:
        if max(wp, hp) / min(wp, hp) <= max_ratio:
            crop_size_list.append((wp * patch_size, hp * patch_size))
        if (hp + 1) * wp <= num_patches:
            hp += 1
        else:
            wp -= 1
    return crop_size_list

def smart_crop_to_bucket(
    image: Image.Image,
    buckets: List[Tuple[int, int]],
) -> Tuple[Tuple[int, int], Image.Image]:
    """Crop *image* to the closest aspect-ratio bucket.

    1. Find bucket whose aspect ratio is closest (log-scale).
    2. Resize so image covers the bucket (short-side match).
    3. Center-crop to exact bucket dims.

    Args:
        image: PIL Image (any mode — will be converted to RGB).
        buckets: List of (H, W) tuples.

    Returns:
        ``(bucket_hw, cropped_image)``
    """
    image = image.convert("RGB")
    w, h = image.size  # PIL is (W, H)
    ar_img = w / h

    # Find closest bucket by log aspect ratio
    best_bucket = min(
        buckets,
        key=lambda b: abs(math.log(b[1] / b[0]) - math.log(ar_img)),
    )
    return best_bucket, crop_image_to_bucket(image, best_bucket)

def crop_image_to_bucket(image: Image.Image, bucket_hw: Tuple[int, int]) -> Image.Image:
    """Cover-resize and center-crop an image to one explicit ``(H, W)`` bucket."""
    image = image.convert("RGB")
    w, h = image.size
    bh, bw = bucket_hw

    # Resize so image covers bucket (short-side match)
    scale = max(bw / w, bh / h)
    new_w = round(w * scale)
    new_h = round(h * scale)
    image = image.resize((new_w, new_h), Image.BICUBIC)

    # Center crop
    left = (new_w - bw) // 2
    top = (new_h - bh) // 2
    image = image.crop((left, top, left + bw, top + bh))

    return image
