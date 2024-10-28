# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root for license information.

import cupy as cp
from scipy.ndimage import shift as ndshift
import numpy as np
import pandas as pd
import networkx as nx
import mutualinformation_single as mi
import mutualinformation_multichannel as mc 
import distortion as ds
from numba import njit 
import matplotlib.pyplot as plt
import torch 


# chromatic correction 
def compute_chromatic(tiles):
    ref_ch = 0
    tar_chs=[1, 2, 3]
    dydxs  = []
    for tile in tiles:
        dydx = []
        ref8 = np.round( (tile[ref_ch] - tile[ref_ch].min()) / tile[ref_ch].max()* 255 ).astype(np.uint8)
        for ch in tar_chs:
            tar8 = np.round( (tile[ch] - tile[ch].min()) / tile[ch].max()* 255 ).astype(np.uint8)
            t, _ = mi.align(ref8, tar8)
            dydx.append(t)
        dydxs.append(dydx)
    torch.cuda.empty_cache()
    return np.array(dydxs) 

from sklearn.linear_model import RANSACRegressor, LinearRegression
def ransac_average(shifts):
    averages = []
    outliers = [] 
    for ch in range(shifts.shape[1]):
        X = np.arange(shifts.shape[0]).reshape(-1, 1)  # Dummy independent variable
        y = shifts[:, ch]
        
        # Initialize RANSAC with the updated argument name
        ransac = RANSACRegressor(estimator=LinearRegression(), residual_threshold=2.0)
        ransac.fit(X, y)
        
        # Check for inliers
        inliers = ransac.inlier_mask_
        if inliers is not None and inliers.sum() > 0:  # If inliers exist
            inlier_average = y[inliers].mean()
        else:  # No inliers case, use default value 0
            inlier_average = 0
        averages.append(inlier_average)
        outliers.append(~inliers)  # Mask outliers where inliers is False
    return np.array(averages), np.vstack(outliers).T 

# Combining both y and x shift averages with default fallback [0, 0]
def compute_ransac_average_shifts(dydxs, residual_threshold=4.0):
    y_shifts = dydxs[:, :, 0]
    x_shifts = dydxs[:, :, 1]

    # Get RANSAC average for y and x shifts separately
    average_y_shifts, outlier_y = ransac_average(y_shifts)
    average_x_shifts, outlier_x = ransac_average(x_shifts)

    # Combine into final result for each color channel
    average_dydx = np.vstack([average_y_shifts, average_x_shifts]).T
    combined_outliers = np.logical_or(outlier_y, outlier_x)
    tile_indices_with_outliers = np.unique(np.where(combined_outliers)[0])
    return average_dydx, combined_outliers, tile_indices_with_outliers

def chrom_correct(tiles, dydxs):
    dydxs, _, _ = compute_ransac_average_shifts(dydxs)
    tiles_corrected = tiles.copy()
    for tile_corrected in tiles_corrected:
        for ch, dydx in enumerate(dydxs):
            tile_corrected[ch+1] = ndshift(tile_corrected[ch+1], -dydx, mode='nearest')
    return tiles_corrected.astype(np.float32)  

from tqdm import tqdm
def pr_stitching(tiles, positions, vis=False):
    gammas = [0.1]*4 + [0.2]*3 + [0.3]*2 + [0.4, 0.5, 0.6, 0.8, 0.9, 1]
    tiles_corrected = tiles.copy()

    for gamma in tqdm(gammas):
        G = construct_matrix_colors(tiles_corrected, positions, gamma=gamma)
        if vis: 
            ds.visualize_graph(G)
        positions, _ = optimize_shifts_with_graph(G, positions) 
    return positions


@njit(cache=True)   
def find_overlap_coords(y1, x1, y2, x2, h, w):
    overlap_x_start = max(x1, x2)
    overlap_x_end   = min(x1 + w, x2 + w)
    overlap_y_start = max(y1, y2)
    overlap_y_end   = min(y1 + h, y2 + h)
    
    if overlap_x_start >= overlap_x_end or overlap_y_start >= overlap_y_end:
        return None, None  
    
    return (overlap_y_start-y1, overlap_y_end-y1, overlap_x_start-x1, overlap_x_end-x1), \
           (overlap_y_start-y2, overlap_y_end-y2, overlap_x_start-x2, overlap_x_end-x2)


def ncc_torch(image1: np.ndarray, image2: np.ndarray) -> float:
    image1_torch = torch.tensor(image1, device="cuda", dtype=torch.float32)
    image2_torch = torch.tensor(image2, device="cuda", dtype=torch.float32)

    image1_flat = image1_torch.view(-1)
    image2_flat = image2_torch.view(-1)

    image1_mean_sub = image1_flat - image1_flat.mean()
    image2_mean_sub = image2_flat - image2_flat.mean()

    numerator = torch.dot(image1_mean_sub, image2_mean_sub)
    denominator = torch.norm(image1_mean_sub) * torch.norm(image2_mean_sub)

    result = (numerator / denominator).item()        
    del image1_torch, image2_torch, image1_flat, image2_flat, image1_mean_sub, image2_mean_sub
    return result


def ncc_cupy(image1: np.ndarray, image2: np.ndarray) -> float:
    image1_cp = cp.asarray(image1)
    image2_cp = cp.asarray(image2)
    
    image1_flat = image1_cp.ravel()
    image2_flat = image2_cp.ravel()
    
    image1_mean_sub = image1_flat - cp.mean(image1_flat)
    image2_mean_sub = image2_flat - cp.mean(image2_flat)
    
    n = cp.dot(image1_mean_sub, image2_mean_sub)    
    d = cp.linalg.norm(image1_mean_sub) * cp.linalg.norm(image2_mean_sub)    
    del image1_cp, image2_cp, image1_flat, image2_flat, image1_mean_sub, image2_mean_sub
    return (n / d).item()


def visualize_graph(G, item='weight'):
    pos = nx.spring_layout(G)  # Position nodes using the spring layout
    plt.figure(figsize=(4, 4))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=220, 
            edge_color='gray', font_size=6)
    edge_labels = nx.get_edge_attributes(G, item)
    edge_labels = {k: f'{v:.3f}' for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.title("Overlap Graph"); plt.show()


from skimage.registration import phase_cross_correlation as pcc
from skimage.filters import unsharp_mask as sharpen 
def cal_shift_ij(i, j, tiles, positions0, h, w, gamma=0.1, usepcc=False):
    positions = positions0.copy()
    pos_i = positions[i]
    pos_j = positions[j]
    dydx_out = np.array([0, 0])
    for _ in range(2):
        yx1, yx2 = find_overlap_coords(pos_i[0], pos_i[1], pos_j[0], pos_j[1], h, w)
        if yx1 is not None and yx2 is not None:
            ref = tiles[i, yx1[0]:yx1[1], yx1[2]:yx1[3]].copy()
            tar = tiles[j, yx2[0]:yx2[1], yx2[2]:yx2[3]].copy()
            ref = ref/ref.max()
            tar = tar/tar.max()
            ref = sharpen(ref, radius=1, amount=1)  # not necessary 
            tar = sharpen(tar, radius=1, amount=1)
            ref8 = np.uint8(ref/ref.max()*255)
            tar8 = np.uint8(tar/tar.max()*255)

            if usepcc:
                t, _, _ = pcc(ref8, tar8)
                dydx = np.round(t * gamma).astype(int)
            else:
                t, _ = mi.align(ref8, tar8)
                dydx = np.round(np.array(t) * gamma).astype(int)
    
            pos_j -= dydx
        else:
            dydx = np.array([0, 0])
        dydx_out += dydx
    return dydx_out 


def cal_shift_ij_colors(i, j, tiles, positions0, h, w, gamma=0.1):
    positions = positions0.copy()
    pos_i = positions[i]
    pos_j = positions[j]
    dydx_out = np.array([0, 0])
    for _ in range(2):
        yx1, yx2 = find_overlap_coords(pos_i[0], pos_i[1], pos_j[0], pos_j[1], h, w)
        if yx1 is not None and yx2 is not None:
            ref = tiles[i, :, yx1[0]:yx1[1], yx1[2]:yx1[3]].copy()
            tar = tiles[j, :, yx2[0]:yx2[1], yx2[2]:yx2[3]].copy()
            ref8 = ref.transpose(1,2,0)
            tar8 = tar.transpose(1,2,0)
            t, _ = mc.align(ref8, tar8)
            dydx = np.round(np.array(t) * gamma).astype(int)    
            pos_j -= dydx
        else:
            dydx = np.array([0, 0])
        dydx_out += dydx
    return dydx_out 


def extract_data_from_graph(G):
    nodes = list(G.nodes())
    num_tiles = len(nodes)

    # Map node IDs to indices (in case node IDs are not sequential)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}

    pairs = []
    shifts_ij = []
    weights = []

    for i, j, data in G.edges(data=True):
        idx_i = node_to_idx[i]
        idx_j = node_to_idx[j]
        pairs.append((idx_i, idx_j))
        shifts_ij.append(np.array(data['shift']))
        weights.append(data['weight'])

    return num_tiles, pairs, shifts_ij, weights, node_to_idx, idx_to_node


from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr
def optimize_shifts_with_graph(G, positions):
    num_tiles, pairs, shifts_ij, weights, node_to_idx, idx_to_node = extract_data_from_graph(G)
    fixed_tile_idx = node_to_idx[0] if 0 in node_to_idx else 0  # Fix tile 0

    # List of variable tiles (excluding the fixed tile)
    variable_indices = [idx for idx in range(num_tiles) if idx != fixed_tile_idx]
    num_variables = 2 * len(variable_indices)  # Two variables per tile (Δy, Δx)
    num_equations = 2 * len(pairs)  # Each pair provides two equations

    # Map tile indices to variable indices
    tile_to_var_idx = {idx: var_idx * 2 for var_idx, idx in enumerate(variable_indices)}

    # Initialize matrices
    A = lil_matrix((num_equations, num_variables))
    b = np.zeros(num_equations)

    equation_idx = 0
    for (idx_i, idx_j), shift_ij, w_ij in zip(pairs, shifts_ij, weights):
        sqrt_w = np.sqrt(w_ij)

        # Check if the tiles are variable or fixed
        i_is_variable = idx_i != fixed_tile_idx
        j_is_variable = idx_j != fixed_tile_idx

        # y-component equation
        if i_is_variable:
            A[equation_idx, tile_to_var_idx[idx_i]] = -sqrt_w
        if j_is_variable:
            A[equation_idx, tile_to_var_idx[idx_j]] = sqrt_w
        b[equation_idx] = sqrt_w * shift_ij[0]
        equation_idx += 1

        # x-component equation
        if i_is_variable:
            A[equation_idx, tile_to_var_idx[idx_i] + 1] = -sqrt_w
        if j_is_variable:
            A[equation_idx, tile_to_var_idx[idx_j] + 1] = sqrt_w
        b[equation_idx] = sqrt_w * shift_ij[1]
        equation_idx += 1

    # Convert A to CSR format for efficient arithmetic and matrix vector operations
    A = A.tocsr()

    # Solve the linear system using least squares
    result = lsqr(A, b)
    x = result[0]  # Solution vector

    # Retrieve shifts
    shifts = np.zeros((num_tiles, 2))
    for idx in range(num_tiles):
        if idx == fixed_tile_idx:
            shifts[idx] = [0.0, 0.0]  # Fixed tile
        else:
            var_idx = tile_to_var_idx[idx]
            shifts[idx] = x[var_idx:var_idx + 2]

    # Map shifts back to tile IDs
    shifts_ordered = {}
    for idx, shift in enumerate(shifts):
        node = idx_to_node[idx]
        shifts_ordered[node] = shift

    # Correct positions
    positions_corrected = positions.copy()
    for node in G.nodes():
        positions_corrected[node] -= np.round(shifts_ordered[node]).astype(int)

    return positions_corrected, shifts_ordered


def construct_matrix_colors(tiles, positions, th = -0.1, gamma=1):
    G = nx.Graph()
    num_tiles = len(tiles)
    h, w = tiles.shape[-2:]

    for i in range(num_tiles):
        G.add_node(i)
    
    for i in range(num_tiles):
        for j in range(i + 1, num_tiles):
            pos_i = positions[i]
            pos_j = positions[j]
            yx1, yx2 = find_overlap_coords(pos_i[0], pos_i[1], pos_j[0], pos_j[1], h, w)
            if yx1 is not None and yx2 is not None:
                # # Extract overlapping regions
                area = (yx1[1] - yx1[0]) * (yx1[3] - yx1[2])  # area as weight

                # Compute the shift (dydx) between overlapping regions
                dydx = cal_shift_ij_colors(i, j, tiles, positions, h, w, gamma=gamma)
                correlation = 0.1
                if correlation > th:
                    G.add_edge(j, i, weight=area, shift=dydx)
    return G

def cost_boundary_ncc(k1, tiles, positions):
    tiles_corrected_gpu = ds.undistort_tiles_batched(tiles, [k1]) 
    tiles_corrected = cp.asnumpy(tiles_corrected_gpu)
    del tiles_corrected_gpu

    # Initialize total NCC
    tot_pcc = 0.0
    num_tiles = len(tiles_corrected)
    h, w = tiles_corrected.shape[-2:]
    boundary_size = 400  

    for i in range(num_tiles):
        for j in range(i + 1, num_tiles):
            pos_i = positions[i]
            pos_j = positions[j]
            yx1, yx2 = find_overlap_coords(
                pos_i[0], pos_i[1], pos_j[0], pos_j[1], h, w)

            if yx1 is not None and yx2 is not None:
                tile_i_overlap = tiles_corrected[i, yx1[0]:yx1[1], yx1[2]:yx1[3]]
                tile_j_overlap = tiles_corrected[j, yx2[0]:yx2[1], yx2[2]:yx2[3]]
                # Determine the overlap dimensions
                overlap_h = yx1[1] - yx1[0]
                overlap_w = yx1[3] - yx1[2]

                if overlap_w < overlap_h:
                    # Lateral overlap (side-by-side tiles)
                    boundary_h = min(boundary_size, overlap_h//2)

                    # Extract the top and bottom boundary, dropping the middle part
                    tile_i = np.vstack((tile_i_overlap[:boundary_h,:], tile_i_overlap[-boundary_h:,:]))
                    tile_j = np.vstack((tile_j_overlap[:boundary_h,:], tile_j_overlap[-boundary_h:,:]))
                else:
                    # Vertical overlap (stacked tiles)
                    boundary_w = min(boundary_size, overlap_w//2)

                    # Extract the left and right boundary, dropping the middle part
                    tile_i = np.hstack((
                        tile_i_overlap[:,:boundary_w],  # Left
                        tile_i_overlap[:,-boundary_w:]  # Right
                    ))
                    tile_j = np.hstack((
                        tile_j_overlap[:,:boundary_w],  # Left
                        tile_j_overlap[:,-boundary_w:]  # Right
                    ))
                correlation = ncc_torch(tile_i, tile_j)
                tot_pcc += correlation
    return -tot_pcc

from joblib import Parallel, delayed
def sharpen_tiles(tiles):
    tiles_corrected = np.copy(tiles)
    ncolors = tiles_corrected.shape[1]
    def sharpen_one(tile):
        tile_corrected = np.copy(tile)
        tile_corrected = sharpen(tile_corrected, radius=1, amount=1)
        return tile_corrected
    
    for ch in range(ncolors):  # normalization
        tiles_corrected[:, ch] = tiles_corrected[:,ch] / tiles_corrected[:, ch].max()
        tiles_corrected[:, ch] = np.array( Parallel(n_jobs=10)(delayed(sharpen_one)(tile) for tile in tiles_corrected[:, ch]) )
    return tiles_corrected.astype(np.float32)

from scipy.optimize import minimize_scalar
def compute_k1_recursive_colors(tiles, positions, outlier_tiles, gamma=0.5, vis=False):  # tiles have multipe channels
    ncolors = tiles.shape[1]
    k1_bounds = (-0.005, 0.005) 
    tiles_corrected = tiles.copy() 

    # remove outlier tiles
    tile_good_ids = list( set(range(len(tiles))) - set(outlier_tiles) )

    k1color_s = []
    for _ in tqdm(range(25)):  #22 iterations is enough
        G = construct_matrix_colors(tiles_corrected, positions, gamma=gamma)
        if vis: 
            visualize_graph(G)
        positions, _ = optimize_shifts_with_graph(G, positions)

        k1color = []
        for c in range(ncolors):
            result = minimize_scalar(
                cost_boundary_ncc,
                bounds = k1_bounds,
                args   = (tiles_corrected[tile_good_ids, c], positions[tile_good_ids]),
                method = 'bounded',
                options= {'xatol': 1e-7}  # Tolerance for convergence
            )
            k1 = result.x * gamma 
            k1color.append(k1)
            tiles_corrected_gpu = ds.undistort_tiles_batched(tiles_corrected[:, c], [k1])
            tiles_corrected[:, c] = cp.asnumpy( tiles_corrected_gpu )    
            print(f'color: {c},  k1: {k1:.6f}')
            del tiles_corrected_gpu
            cp._default_memory_pool.free_all_blocks()   
        torch.cuda.empty_cache()
        k1color_s.append(k1color)
    return positions, np.array(k1color_s)