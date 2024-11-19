#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
import numpy as np
from abc import abstractmethod
import scipy
from scipy.stats import skellam
import scipy.optimize as optimize
from scipy import sparse, optimize
from scipy.special import gamma, softmax
import cvxpy
import pdb
from scipy.sparse.csr import csr_matrix
from tqdm import tqdm
import os
from datetime import datetime
import pickle
import time


class CompressedMechanism:
    
    def __init__(self, budget, epsilon):
        self.budget = budget
        self.epsilon = epsilon
        return
    
    def dither(self, x, b, p=None):
        """
        Given x in [0,1], return a randomized dithered output in {0, 1, ..., 2^b - 1}.
        """
        assert np.all(x >= 0) and np.all(x <= 1)
        B = 2 ** b
        k = np.floor((B-1) * x)
        prob = 1 - (B-1) * (x - k/(B-1))
        k += np.random.random(k.shape) > prob
        return k.astype(int)
    
    @abstractmethod
    def privatize(self, x):
        """
        Privatizes a vector of values in [0,1] to binary vectors.
        """
        return
    
    @abstractmethod
    def decode(self, l):
        """
        Decodes binary vectors to an array of real values.
        """
        return
        
        
class RandomizedResponseMechanism(CompressedMechanism):
    
    def _privatize_bit(self, x, epsilon):
        """
        Privatizes a vector of bits using the binary randomized response mechanism.
        """
        assert np.all(np.logical_or(x == 0, x == 1))
        prob = 1 / (1 + math.exp(-epsilon))
        mask = np.random.random(x.shape) > prob
        z = np.logical_xor(mask, x).astype(int)
        return z
    
    def binary_repr(self, x):
        """
        Converts an array of integers to a 2D array of bits using binary representation.
        """
        l = [np.fromiter(map(int, np.binary_repr(a, width=self.budget)), int) for a in x]
        return np.stack(l, 0)
    
    def int_repr(self, l):
        """
        Converts a 2D array of bits into an array of integers using binary representation.
        """
        powers = np.power(2, np.arange(self.budget-1, -0.5, -1))
        return l.dot(powers)
    
    def privatize(self, x):
        z = self.dither(x, self.budget)
        l = self.binary_repr(z)
        l = self._privatize_bit(l, float(self.epsilon/self.budget))
        return l
    
    def decode(self, l):
        assert l.shape[1] == self.budget
        a_0 = -1 / (math.exp(self.epsilon/self.budget) - 1)
        a_1 = math.exp(self.epsilon/self.budget) / (math.exp(self.epsilon/self.budget) - 1)
        l = a_0 + l * (a_1 - a_0)
        return self.int_repr(l) / (2**self.budget - 1)
    
    
class MultinomialSamplingMechanism(CompressedMechanism):
    
    def __init__(self, budget, epsilon, input_bits, norm_bound, p, **kwargs):
        """
        Parent class that supports sampling from a 2^budget-dimensional distribution defined by
        a sampling probability matrix P and an output vector alpha.
        
        Arguments:
        budget     - Number of bits in the output.
        epsilon    - DP/metric-DP parameter epsilon.
        input_bits - Number of bits in the quantized input.
        norm_bound - A priori bound on the norm of the input before quantization; ignored if p=None.
        p          - Which p-norm to use for the norm bound parameter.
        """
        super().__init__(budget, epsilon)
        print("MultinomialSamplingMechanism, init......")
        self.input_bits = input_bits
        self.norm_bound = norm_bound
        self.p = p
        result = self.optimize(**kwargs)
        print("result,--------",result)
        if result is not None:
            self.P, self.alpha = result[0], result[1]
        
        print("self.P.shape",self.P.shape)
        return
    
    def dither(self, x, b):
        """
        Dithers x coordinate-wise to a grid of size 2^b.
        If self.p is set, perform rejection sampling until dithered vector does not exceed self.norm_bound.
        """
        assert np.all(x >= 0) and np.all(x <= 1)
        B = 2 ** b
        k = np.floor((B-1) * x)
        prob = 1 - (B-1) * (x - k/(B-1))
        while True:
            output = k + (np.random.random(k.shape) > prob)
            if self.p is None or np.linalg.norm(output / (B-1) - 0.5, self.p) <= self.norm_bound:
                break
        return output.astype(int)
    
    @abstractmethod
    def optimize(self, **kwargs):
        """
        Optimizes self.P and self.alpha for multinomial sampling.
        """
        return
    
    def privatize(self, x):
        # 将梯度向量展平为一维
        original_shape = x.shape
        flatten_gradient_vector = x.flatten()

        # 调用 dither 函数对展平的梯度进行扰动
        z = self.dither(flatten_gradient_vector, self.input_bits)

        B = 2**self.budget
        range_B = np.arange(B).astype(int)

        # 使用矢量化方式对扰动后的结果进行采样
        z_reshaped = np.array([np.random.choice(range_B, p=self.P[int(a)]) for a in z])

        # 恢复为原始形状
        z_reshaped = z_reshaped.reshape(original_shape)
        return z_reshaped



        # for a in z:
        #     # assert np.isclose(self.P[a].sum(), 1), f"Probabilities do not sum to 1: {self.P[a]}"
        #     assert len(self.P[a].shape) == 1, f"Expected self.P[{a}] to be 1-dimensional, but got shape {self.P[a].shape}"

        # z = np.array([np.random.choice(range_B, p=self.P[int(a)]) for a in z])
        # return z
    
    def decode(self, z):
        assert z.min() >= 0 and z.max() < 2**self.budget
        return self.alpha[z.astype(int)]
    
    def mse_and_bias_squared(self, P=None, alpha=None):
        """
        Evaluate MSE loss and bias squared.
        """
        if P is None and alpha is None:
            P = self.P
            alpha = self.alpha
        B = 2 ** self.input_bits
        target = np.arange(0, 1+1/B, 1/(B-1))
        mse_loss = (P * np.power(target[:, None] - alpha[None, :], 2)).sum(1).mean()
        bias_sq = np.power(P @ alpha - target, 2).mean()
        return mse_loss, bias_sq
    

class RAPPORMechanism(MultinomialSamplingMechanism):
    
    def __init__(self, budget, epsilon, input_bits, norm_bound=0.5, p=None, **kwargs):
        super().__init__(budget, epsilon, budget, norm_bound, p, **kwargs)     # ignores input bits
        return
    
    def optimize(self):
        B = 2**self.budget
        prob = B / (B + math.exp(self.epsilon) - 1)
        P = prob / B * np.ones((B, B)) + (1 - prob) * np.eye(B)
        print(P.shape)
        assert P.shape == (B, B), f"Unexpected shape for self.P: {P.shape}"

        target = np.arange(0, 1+1/B, 1/(B-1))
        alpha = np.linalg.solve(P, target)
        return P, alpha
    