import numpy as np
from mpi4py import MPI
import torch
import torch.nn as nn

class MatrixQuantizer:
    def __init__(self, num_bits=1, zero_threshold=True):
        self.num_bits = num_bits
        self.zero_threshold = zero_threshold
        
    def quantize(self, matrix):
        # Perform 1-bit quantization (sign-based quantization)
        return np.sign(matrix)  # Quantizes to +1 or -1
    
    def unquantize(self, quantized_matrix):
        # Since 1-bit quantization only retains the sign, use it as is for unquantization
        return quantized_matrix

class QuantizedSGDCommunicator:
    def __init__(self):
        self.num_bits = 1
        self.quantizer = MatrixQuantizer(self.num_bits)
        self.error_feedback = None  # Store error feedback for compensation
        
    def initialize_error_feedback(self, shape):
        if self.error_feedback is None:
            self.error_feedback = np.zeros(shape)
        
    def local_quantize(self, local_gradients):
        # Initialize error feedback if not already done
        self.initialize_error_feedback(local_gradients.shape)
        
        # Add error feedback to local gradients
        adjusted_gradients = local_gradients + self.error_feedback

        # Quantize adjusted gradients
        quantized_local_gradients = self.quantizer.quantize(adjusted_gradients)

        # Calculate new error feedback
        self.error_feedback = adjusted_gradients - quantized_local_gradients
        
        return quantized_local_gradients

    def apply_1bit_sgd_quantization(self, local_gradients):
        # Perform local quantization with error feedback
        quantized_local_gradients = self.local_quantize(local_gradients)
        
        return quantized_local_gradients
