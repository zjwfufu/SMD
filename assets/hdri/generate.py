"""Generate the bundled, procedural studio lighting map without external assets."""
import argparse
import os
from pathlib import Path

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import cv2
import numpy as np


def studio_lighting(height=128):
    longitude = (np.arange(2 * height) + 0.5) * np.pi / height - np.pi
    latitude = np.pi / 2 - (np.arange(height) + 0.5) * np.pi / height
    lon, lat = np.meshgrid(longitude, latitude)
    direction = np.stack([np.cos(lat) * np.cos(lon), np.sin(lat),
                          np.cos(lat) * np.sin(lon)], axis=-1)
    rgb = np.full((*lon.shape, 3), 0.18, dtype=np.float64)
    for center, color, width in [
        ((1, 1, 1), (6.0, 5.4, 4.8), 0.16),
        ((-1, 0.5, -1), (1.2, 1.4, 1.8), 0.3),
    ]:
        axis = np.asarray(center, dtype=np.float64)
        axis /= np.linalg.norm(axis)
        lobe = np.exp((direction @ axis - 1) / width**2)
        rgb += lobe[..., None] * np.asarray(color)
    return rgb.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[2] / 'assets/hdri/forest.exr')
    args = parser.parse_args()
    rgb = studio_lighting()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), rgb[..., ::-1].copy(),
                       [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]):
        raise RuntimeError('Failed to write the EXR lighting map')
    decoded = cv2.imread(str(args.output), cv2.IMREAD_UNCHANGED)
    if decoded is None or not np.array_equal(decoded[..., ::-1], rgb):
        raise RuntimeError('EXR lighting map failed its lossless round-trip check')
    print(f'Lighting map written and verified: {args.output} ({args.output.stat().st_size} bytes)')


if __name__ == '__main__':
    main()
