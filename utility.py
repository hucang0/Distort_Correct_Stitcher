# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root for license information.

import numpy as np 
import tifffile

def save_fiji(img, fname, dimension_order='TZCYX', normalization=False):
    if normalization:
        img_fiji = np.array(img/img.max()*(2**15)).astype(np.uint16)
    else:
        img_fiji = np.array(img).astype(np.uint16)
    tifffile.imwrite(fname, img_fiji.astype('uint16'), shape=img_fiji.shape,
                imagej=True, metadata={'axes':dimension_order})

def assemble(tiles, positions):
    tiles = np.round(tiles).astype(np.uint16)
    h, w = tiles.shape[-2:]
    positions = positions.copy()

    y_coords = positions[:, 0] - positions[:, 0].min()
    x_coords = positions[:, 1] - positions[:, 1].min()

    H = int((y_coords + h).max() + h * 1.2)
    W = int((x_coords + w).max() + w * 1.2)
    oy, ox = h // 4, w // 4  # Offsets
    b = 64  # Border size

    is_single_channel = tiles.ndim == 3  # Check if tiles are single-channel
    if is_single_channel:
        canvas = np.zeros((H, W), dtype=np.uint16)
        for idx in range(len(tiles)):
            y = y_coords[idx]
            x = x_coords[idx]
            yt = int(round(y + oy))
            xt = int(round(x + ox))
            canvas[yt+b:yt+h-b, xt+b:xt+w-b] = tiles[idx, b:-b, b:-b]
    else:
        channels = tiles.shape[1]
        canvas = np.zeros((channels, H, W), dtype=np.uint16)
        for idx in range(tiles.shape[0]):
            y = y_coords[idx]
            x = x_coords[idx]
            yt = int(round(y + oy))
            xt = int(round(x + ox))
            canvas[:, yt+b:yt+h-b, xt+b:xt+w-b] = tiles[idx, :, b:-b, b:-b]
    return canvas
