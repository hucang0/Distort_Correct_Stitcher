# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root for license information.

import numpy as np 
import cupy as cp 
from cucim.skimage import transform as cucim_transform

def radial_distortion_inverse_mapping(coords, h, w, k1, k2=0.0, k3=0.0):
    y_u, x_u = coords  # coords are in (2, N) shape

    # Normalize coordinates to [-1, 1]
    x_c = (x_u - w / 2.0) / (w / 2.0)
    y_c = (y_u - h / 2.0) / (h / 2.0)
    r2 = x_c ** 2 + y_c ** 2

    # Compute radial distortion factor
    radial = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3

    # Map to distorted coordinates
    x_d = x_c * radial
    y_d = y_c * radial

    # Convert back to image coordinates
    x_d = x_d * (w / 2.0) + w / 2.0
    y_d = y_d * (h / 2.0) + h / 2.0
    return cp.vstack((y_d, x_d))  # Shape: (2, N)

def undistort_image(image, k1, k2=0.0, k3=0.0):
    h, w = image.shape[:2]
    image_gpu = cp.array(image, dtype=cp.float32)
    y, x      = cp.indices((h, w), dtype=cp.float32)
    coords    = cp.stack((y.ravel(), x.ravel()))  # Shape: (2, N)

    # Compute the inverse mapping on GPU
    coords_d = radial_distortion_inverse_mapping(coords, h, w, k1, k2, k3)
    coords_d = coords_d.reshape(2, h, w)

    # Perform the warping using cucim's warp function
    undistorted_image = cucim_transform.warp(
        image_gpu,
        inverse_map=coords_d,
        order=5,  # Higher-order spline interpolation
        mode='constant',
        cval=0.0,
        clip=True,
        preserve_range=True,
    )
    del image_gpu, coords, coords_d, y, x
    return undistorted_image

def undistort_tiles_batched(tiles, k1s, batch_size=20):
    tiles_corr = cp.array(tiles, dtype=cp.float32)
    num_tiles = tiles_corr.shape[0]

    for k1 in k1s:
        # Process tiles in batches to manage memory usage
        for start_idx in range(0, num_tiles, batch_size):
            end_idx = min(start_idx + batch_size, num_tiles)
            batch_tiles = tiles_corr[start_idx:end_idx]

            # Apply distortion correction to each tile in the batch
            batch_corr_gpu = cp.array([
                undistort_image(tile_gpu, k1) for tile_gpu in batch_tiles
            ])

            # Update tiles_corr with corrected batch
            tiles_corr[start_idx:end_idx] = batch_corr_gpu

            # Free GPU memory of the batch
            del batch_tiles, batch_corr_gpu
    cp._default_memory_pool.free_all_blocks()
    return tiles_corr

def undistort_tiles(tiles, k1s):
    tiles_corr = cp.array(tiles, dtype=cp.float32)
    for k1 in k1s:
        tiles_corr = cp.array([undistort_image(tile, k1) for tile in tiles_corr])
    return tiles_corr
