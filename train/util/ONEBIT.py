import numpy as np
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

class OneBit:
    def __init__(self, epsilon):
        self.num_bits = 1
        self.quantizer = MatrixQuantizer(self.num_bits)
        self.error_feedback = None  # Store error feedback for compensation
        self.epsilon = epsilon
        

    def initialize_error_feedback(self, model):
        if self.error_feedback is None:
            self.error_feedback = {}
            for name, param in model.module.named_parameters() if hasattr(model, 'module') else model.named_parameters():
                self.error_feedback[name] = np.zeros(param.shape)
    # def initialize_error_feedback(self, model):
    #     if self.error_feedback is None:
    #         self.error_feedback = {}
    #         for name, param in model.named_parameters():
    #             if param.grad is not None:
    #                 self.error_feedback[name] = np.zeros(param.grad.shape)
        
    def local_quantize(self, name, param):
        # Add error feedback to local gradients
        local_gradients = param.grad.cpu().numpy()
        adjusted_gradients = local_gradients + self.error_feedback[name]

        # Quantize adjusted gradients
        quantized_local_gradients = self.quantizer.quantize(adjusted_gradients)

        # Calculate new error feedback
        self.error_feedback[name] = adjusted_gradients - quantized_local_gradients
        
        return quantized_local_gradients



    # def initialize_error_feedback(self, model):
    #     if self.error_feedback is None:
    #         self.error_feedback = {}
    #         for param in model.parameters():
    #             if param.grad is not None:
    #                 self.error_feedback[param] = np.zeros(param.grad.shape)
    #     # else:
    #     #     for param in model.parameters():
    #     #         if param.grad is not None and param not in self.error_feedback:
    #     #             self.error_feedback[param] = np.zeros(param.grad.shape)
    #     #         elif param.grad is not None and self.error_feedback[param].shape != param.grad.shape:
    #     #             # Resize the error feedback if the shape has changed, retaining previous values where possible
    #     #             new_error_feedback = np.zeros(param.grad.shape)
    #     #             min_shape = tuple(min(old, new) for old, new in zip(self.error_feedback[param].shape, param.grad.shape))
    #     #             new_error_feedback[np.ix_(*[range(dim) for dim in min_shape])] = self.error_feedback[param][np.ix_(*[range(dim) for dim in min_shape])]
    #     #             self.error_feedback[param] = new_error_feedback
        
    # def local_quantize(self, param):
    #     # Add error feedback to local gradients
    #     local_gradients = param.grad.cpu().numpy()
    #     adjusted_gradients = local_gradients + self.error_feedback[param]

    #     # Quantize adjusted gradients
    #     quantized_local_gradients = self.quantizer.quantize(adjusted_gradients)

    #     # Calculate new error feedback
    #     self.error_feedback[param] = adjusted_gradients - quantized_local_gradients
        
    #     return quantized_local_gradients


    def apply_1bit_sgd_quantization(self, name, param):
        # Perform local quantization with error feedback
        quantized_local_gradients = self.local_quantize(name, param)
        return quantized_local_gradients