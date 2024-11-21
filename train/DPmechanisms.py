import pickle
import numpy as np
import math
import sympy as sp
from utils import transform, discrete
import torch


#################################ADD NOISE#######################################
def add_encoding(updates, epsilon):
    C = (np.e ** epsilon + 1) / (np.e ** epsilon - 1)
    updates = np.where(np.random.rand() < (0.5 + updates * ((np.e ** epsilon - 1) / (2 * np.e ** epsilon + 2))), C, -C)
    return updates


def add_laplace(updates, sensitivity, epsilon):
    '''
    inject laplacian noise to a vector
    '''
    # lambda_ = sensitivity * 1.0 / epsilon
    # updates += np.random.laplace(loc=0, scale=lambda_, size=updates.shape)
    # return updates
    # 计算拉普拉斯噪声的尺度参数 lambda
    lambda_ = sensitivity * 1.0 / epsilon

    if isinstance(updates, np.ndarray):
        # 如果是 NumPy 数组，直接生成拉普拉斯噪声并添加
        noise = np.random.laplace(loc=0, scale=lambda_, size=updates.shape)
        updates += noise
    elif isinstance(updates, torch.Tensor):
        # 如果是 PyTorch 张量，生成拉普拉斯噪声并添加
        noise = torch.from_numpy(np.random.laplace(loc=0, scale=lambda_, size=updates.shape)).to(updates.device)
        updates += noise
    else:
        raise TypeError("Unsupported type for add_laplace. Expected np.ndarray or torch.Tensor.")

    return updates


def add_gaussian(updates, eps, delta, sensitivity):
    '''
    inject gaussian noise to a vector
    '''
    sigma = (sensitivity / eps) * math.sqrt(2 * math.log(1.25 / delta))
    updates += np.random.normal(0, sigma)
    return updates


def one_gaussian(eps, delta, sensitivity):
    '''
    sample a gaussian noise for a scalar
    '''
    sigma = (sensitivity / eps) * math.sqrt(2 * math.log(1.25 / delta))
    return np.random.normal(0, sigma)


def one_laplace(eps, sensitivity):
    '''
    sample a laplacian noise for a scalar
    '''
    return np.random.laplace(loc=0, scale=sensitivity / eps)


def full_randomizer(vector, clip_C, eps, delta, mechanism, left=0, right=1):
    clipped = np.clip(vector, -clip_C, clip_C)
    normalized_updates = transform(clipped, -clip_C, clip_C, left, right)
    if mechanism == 'gaussian':
        perturbed = add_gaussian(normalized_updates, eps, delta, sensitivity=right - left)
    elif mechanism == 'laplace':
        perturbed = add_laplace(normalized_updates, sensitivity=1, epsilon=eps)
    return perturbed


def sampling_randomizer(vector, choices, clip_C, eps, delta, mechanism, left=0, right=1):
    vector = np.clip(vector, -clip_C, clip_C)
    for i, v in enumerate(vector):
        if i in choices:
            normalize_v = transform(vector[i], -clip_C, clip_C, left, right)
            if mechanism == 'gaussian':
                vector[i] = normalize_v + one_gaussian(eps, delta, right - left)
            elif mechanism == 'laplace':
                vector[i] = normalize_v + one_laplace(eps, right - left)
        else:
            vector[i] = 0
    return vector


def encoding_randomizer(vector, eps, sp_ratio=0.1):
    maxv = max(abs(vector))
    vector = vector/abs(maxv)
    d = vector.size
    non_top_idx = np.argsort(np.abs(vector), axis=0)[:d-int(d * sp_ratio)]
    vector[non_top_idx] = 0
    vector = add_encoding(vector, eps)
    return vector
