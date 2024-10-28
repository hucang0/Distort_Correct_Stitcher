#Adapted from code originally authored by Johan Öfverstedt (2021). 
#Licensed under the MIT License. Modifications made by [Hu Cang, 2024].

import torch
import numpy as np
import torch.nn.functional as F
import scipy.ndimage

VALUE_TYPE = torch.float32

def compute_entropy(C, N, eps=1e-7):
    p = C / N
    return p * torch.log2(torch.clamp(p, min=eps))

def float_compare(A, c):
    return torch.clamp(1 - torch.abs(A - c), 0.0)

def fft_of_levelsets(A, Q, packing, setup_fn):
    fft_list = []
    for a_start in range(0, Q, packing):
        a_end = min(a_start + packing, Q)
        levelsets = []
        for a in range(a_start, a_end):
            levelsets.append(float_compare(A, a))
        A_cat = torch.cat(levelsets, 0)
        del levelsets
        ffts = setup_fn(A_cat)
        del A_cat
        fft_list.append((ffts, a_start, a_end))
    return fft_list

def fft(A):
    spectrum = torch.fft.rfft2(A)
    return spectrum

def ifft(Afft):
    res = torch.fft.irfft2(Afft)
    return res

def fftconv(A, B):
    C = A * B
    return C

def corr_target_setup(A):
    B = fft(A)
    return B

def corr_template_setup(B):
    B_FFT = torch.conj(fft(B))
    return B_FFT

def corr_apply(A, B, sz, do_rounding=True):
    C = fftconv(A, B)
    C = ifft(C)
    C = C[:sz[0], :sz[1], :sz[2], :sz[3]]
    if do_rounding:
        C = torch.round(C)
    return C

def create_float_tensor(shape, on_gpu, fill_value=None):
    if on_gpu:
        res = torch.cuda.FloatTensor(shape[0], shape[1], shape[2], shape[3])
        if fill_value is not None:
            res.fill_(fill_value)
        return res
    else:
        if fill_value is not None:
            res = np.full((shape[0], shape[1], shape[2], shape[3]), fill_value=fill_value, dtype='float32')
        else:
            res = np.zeros((shape[0], shape[1], shape[2], shape[3]), dtype='float32')
        return torch.tensor(res, dtype=torch.float32)

def to_tensor(A, on_gpu=True):
    if torch.is_tensor(A):
        A_tensor = A.cuda(non_blocking=True) if on_gpu else A
        if A_tensor.ndim == 2:
            A_tensor = torch.reshape(A_tensor, (1, 1, A_tensor.shape[0], A_tensor.shape[1]))
        elif A_tensor.ndim == 3:
            A_tensor = torch.reshape(A_tensor, (1, A_tensor.shape[0], A_tensor.shape[1], A_tensor.shape[2]))
        return A_tensor
    else:
        return to_tensor(torch.tensor(A, dtype=VALUE_TYPE), on_gpu=on_gpu)

### End helper functions

def align_translation(A, B, M_A, M_B, Q_A, Q_B, overlap=0.5, enable_partial_overlap=True, normalize_mi=False, on_gpu=True, save_maps=False):
    eps = 1e-7
    maps = []

    A_tensor = to_tensor(A, on_gpu=on_gpu)
    B_tensor = to_tensor(B, on_gpu=on_gpu)

    if A_tensor.shape[-1] < 1024:
        packing = min(Q_B, 64)
    elif A_tensor.shape[-1] <= 2048:
        packing = min(Q_B, 8)
    elif A_tensor.shape[-1] <= 4096:
        packing = min(Q_B, 4)
    else:
        packing = min(Q_B, 1)

    # Create all constant masks if not provided
    if M_A is None:
        M_A = create_float_tensor(A_tensor.shape, on_gpu, 1.0)
    else:
        M_A = to_tensor(M_A, on_gpu)
        A_tensor = torch.round(M_A * A_tensor + (1 - M_A) * (Q_A + 1))
    if M_B is None:
        M_B = create_float_tensor(B_tensor.shape, on_gpu, 1.0)
    else:
        M_B = to_tensor(M_B, on_gpu)

    # Pad for overlap
    if enable_partial_overlap:
        partial_overlap_pad_sz = (round(B.shape[-1] * (1.0 - overlap)), round(B.shape[-2] * (1.0 - overlap)))
        A_tensor = F.pad(A_tensor, (partial_overlap_pad_sz[0], partial_overlap_pad_sz[0],
                                    partial_overlap_pad_sz[1], partial_overlap_pad_sz[1]), mode='constant', value=Q_A + 1)
        M_A = F.pad(M_A, (partial_overlap_pad_sz[0], partial_overlap_pad_sz[0],
                          partial_overlap_pad_sz[1], partial_overlap_pad_sz[1]), mode='constant', value=0)
    else:
        partial_overlap_pad_sz = (0, 0)

    ext_ashape = A_tensor.shape
    ext_bshape = B_tensor.shape
    b_pad_shape = torch.tensor(A_tensor.shape, dtype=torch.long) - torch.tensor(B_tensor.shape, dtype=torch.long)
    ext_valid_shape = b_pad_shape + 1
    batched_valid_shape = ext_valid_shape + torch.tensor([packing - 1, 0, 0, 0])

    # Precompute FFTs of A and M_A
    M_A_FFT = corr_target_setup(M_A)

    A_ffts = []
    for a in range(Q_A):
        A_ffts.append(corr_target_setup(float_compare(A_tensor, a)))

    del A_tensor
    del M_A

    if normalize_mi:
        H_MARG = create_float_tensor(ext_valid_shape, on_gpu, 0.0)
        H_AB = create_float_tensor(ext_valid_shape, on_gpu, 0.0)
    else:
        MI = create_float_tensor(ext_valid_shape, on_gpu, 0.0)

    # Prepare B (no rotation)
    B_tensor_padded = F.pad(B_tensor, (0, ext_ashape[-1] - ext_bshape[-1],
                                       0, ext_ashape[-2] - ext_bshape[-2],
                                       0, 0, 0, 0), mode='constant', value=Q_B + 1)
    M_B_padded = F.pad(M_B, (0, ext_ashape[-1] - ext_bshape[-1],
                             0, ext_ashape[-2] - ext_bshape[-2],
                             0, 0, 0, 0), mode='constant', value=0)
    B_tensor_padded = torch.round(M_B_padded * B_tensor_padded + (1 - M_B_padded) * (Q_B + 1))

    M_B_FFT = corr_template_setup(M_B_padded)
    N = torch.clamp(corr_apply(M_A_FFT, M_B_FFT, ext_valid_shape), min=eps)

    b_ffts = fft_of_levelsets(B_tensor_padded, Q_B, packing, corr_template_setup)

    for bext in range(len(b_ffts)):
        b_fft = b_ffts[bext]
        E_M = torch.sum(compute_entropy(corr_apply(M_A_FFT, b_fft[0], batched_valid_shape), N, eps), dim=0)
        if normalize_mi:
            H_MARG = torch.sub(H_MARG, E_M)
        else:
            MI = torch.sub(MI, E_M)
        del E_M

        for a in range(Q_A):
            A_fft_cuda = A_ffts[a]

            if bext == 0:
                E_M = compute_entropy(corr_apply(A_fft_cuda, M_B_FFT, ext_valid_shape), N, eps)
                if normalize_mi:
                    H_MARG = torch.sub(H_MARG, E_M)
                else:
                    MI = torch.sub(MI, E_M)
                del E_M
            E_J = torch.sum(compute_entropy(corr_apply(A_fft_cuda, b_fft[0], batched_valid_shape), N, eps), dim=0)
            if normalize_mi:
                H_AB = torch.sub(H_AB, E_J)
            else:
                MI = torch.add(MI, E_J)
            del E_J
            del A_fft_cuda
        del b_fft
        if bext == 0:
            del M_B_FFT

    del B_tensor_padded

    if normalize_mi:
        MI = torch.clamp((H_MARG / (H_AB + eps) - 1), 0.0, 1.0)

    if save_maps:
        maps.append(MI.cpu().numpy())

    (max_n, _) = torch.max(torch.reshape(N, (-1,)), 0)
    N_filt = torch.lt(N, overlap * max_n)
    MI[N_filt] = 0.0
    del N_filt, N

    MI_vec = torch.reshape(MI, (-1,))
    (val, ind) = torch.max(MI_vec, -1)

    sz_x = int(ext_valid_shape[3].cpu().numpy())
    y = ind // sz_x
    x = ind % sz_x

    # Adjust translations to account for padding
    translation_y = -(y - partial_overlap_pad_sz[1])
    translation_x = -(x - partial_overlap_pad_sz[0])

    # Convert to scalar values using .item()
    val = val.item()
    translation_y = translation_y.item()
    translation_x = translation_x.item()

    result = (val, translation_y, translation_x)

    if save_maps:
        return result, maps
    else:
        return result, None

def warp_image_translation(ref_image, flo_image, param, mode='nearest', bg_value=0.0, inv=False):
    translation = TranslationTransform(2)
    translation.set_param(0, param[1])  # x translation
    translation.set_param(1, param[2])  # y translation

    if inv:
        t = translation.invert()
    else:
        t = translation

    out_shape = ref_image.shape[:2] + flo_image.shape[2:]
    flo_image_out = np.zeros(out_shape, dtype=flo_image.dtype)
    if flo_image.ndim == 3:
        for i in range(flo_image.shape[2]):
            bg_val_i = np.array(bg_value)
            if bg_val_i.shape[0] == flo_image.shape[2]:
                bg_val_i = bg_val_i[i]
            t.warp(flo_image[:, :, i], flo_image_out[:, :, i], in_spacing=np.ones(2,), out_spacing=np.ones(2,), mode=mode, bg_value=bg_val_i)
    else:
        t.warp(flo_image, flo_image_out, in_spacing=np.ones(2,), out_spacing=np.ones(2,), mode=mode, bg_value=bg_value)
    return flo_image_out

# Necessary transformation classes
class TransformBase:
    def __init__(self, dim, nparam):
        self.dim = dim
        self.param = np.zeros((nparam,))

    def get_dim(self):
        return self.dim

    def get_params(self):
        return self.param

    def set_params(self, params):
        self.param[:] = params[:]

    def get_param(self, index):
        return self.param[index]

    def set_param(self, index, value):
        self.param[index] = value

    def set_params_const(self, value):
        self.param[:] = value

    def copy(self):
        t = self.copy_child()
        t.set_params(self.get_params())
        return t

    def copy_child(self):
        raise NotImplementedError

    def transform(self, pnts):
        raise NotImplementedError

    def warp(self, In, Out, in_spacing=None, out_spacing=None, mode='spline', bg_value=0.0):
        linspaces = [np.linspace(0, Out.shape[i] * out_spacing[i], Out.shape[i], endpoint=False) for i in range(Out.ndim)]
        grid = np.array(np.meshgrid(*linspaces, indexing='ij'))
        grid = grid.reshape((Out.ndim, np.prod(Out.shape)))
        grid = np.moveaxis(grid, 0, 1)
        grid_transformed = self.transform(grid)
        if in_spacing is not None:
            grid_transformed[:, :] = grid_transformed[:, :] * (1.0 / in_spacing[:])
        grid_transformed = np.moveaxis(grid_transformed, 0, 1)
        grid_transformed = grid_transformed.reshape((Out.ndim,) + Out.shape)
        if mode == 'spline' or mode == 'cubic':
            scipy.ndimage.map_coordinates(In, coordinates=grid_transformed, output=Out, cval=bg_value)
        elif mode == 'linear':
            scipy.ndimage.map_coordinates(In, coordinates=grid_transformed, output=Out, order=1, cval=bg_value)
        elif mode == 'nearest':
            scipy.ndimage.map_coordinates(In, coordinates=grid_transformed, output=Out, order=0, cval=bg_value)

class TranslationTransform(TransformBase):
    def __init__(self, dim):
        TransformBase.__init__(self, dim, dim)

    def copy_child(self):
        return TranslationTransform(self.get_dim())

    def transform(self, pnts):
        offset = self.get_params()
        return pnts + offset

    def invert(self):
        self_inv = self.copy()
        self_inv.set_params(-self_inv.get_params())
        return self_inv

### Usage example
def align(ref, tar):
    h, w = ref.shape[0], ref.shape[1]
    Q = 16  # Number of quantization levels 8 works before
    ref8 = ref.copy().astype(np.float32)
    tar8 = tar.copy().astype(np.float32)

    ref_q = image2cat_kmeans(ref8, Q)
    tar_q = image2cat_kmeans(tar8, Q)
    M = np.ones((h, w), dtype='bool')

    overlap = 0.5
    on_gpu = True

    param, _ = align_translation(ref_q, tar_q, M, M, Q, Q, overlap=overlap,
                                 enable_partial_overlap=True, normalize_mi=False,
                                 on_gpu=on_gpu, save_maps=False)
    # Extract translation as a list of two numbers
    translation = [param[1], param[2]]  # [translation_y, translation_x]
    return translation, param  # Return both translation and param

def transform_img(ref, tar, param):
    img_recover = warp_image_translation(ref.astype(np.float32), tar.astype(np.float32),
                                         param, mode='nearest', bg_value=0)
    return img_recover.astype(ref.dtype)

# Additional required function
from sklearn.cluster import MiniBatchKMeans
def image2cat_kmeans(I, k, batch_size=100, max_iter=1000, random_seed=1000):
    total_shape = I.shape
    spatial_shape = total_shape
    channels = 1
    if k == 1:
        return np.zeros(spatial_shape, dtype='int')
    I_lin = I.reshape(-1, channels)
    kmeans = MiniBatchKMeans(n_clusters=k, max_iter=max_iter, batch_size=batch_size, random_state=random_seed).fit(I_lin)
    I_res = kmeans.labels_
    return I_res.reshape(spatial_shape)


