from sympy import symbols, exp, Rational, lambdify
import numpy as np
import torch
import DPmechanisms as DP

class PPGC:
    def __init__(self, epsilon, out_bits,model):
        self.epsilon = epsilon
        self.C = float((exp(epsilon) + 1) / (exp(epsilon) - 1))
        self.qutibit = out_bits
        self.outputs, self.probabilities = self.calculate_quantization(self.qutibit)
        self.error_feedback = {}
        for name, param in model.module.named_parameters() if hasattr(model, 'module') else model.named_parameters():
            self.error_feedback[name] = np.zeros(param.shape)


    def calculate_quantization(self, qutibit):
        outbit = 2 ** qutibit
        x = symbols('x')
        probabilities, outputs, pa_i = [], [], []

        for i in range(1, outbit // 2 + 1):
            output_value = (outbit - (2 * i - 1)) * self.C
            outputs.append(float(output_value))
            abottom_i = 2 * (outbit // 2) * (outbit - (i * 2) + 1)
            a_i = (float(exp(self.epsilon) - 1) / (abottom_i * float(exp(self.epsilon) + 1)))
            pa_i.append(a_i)
            p_i = x * a_i + Rational(1, outbit)
            probabilities.append(lambdify(x, p_i))

        for i in range(outbit // 2):
            outputs.append(-outputs[outbit // 2 - i - 1])
            a_i = -pa_i[outbit // 2 - i - 1]
            p_i = x * a_i + Rational(1, outbit)
            probabilities.append(lambdify(x, p_i))

        return np.array(outputs, dtype=np.float32), probabilities

    def map_gradient(self, name, param):
        gradient_vector = param.grad.cpu().numpy()
        #gradient_vector = gradient_vector + self.error_feedback[name]

        # 将梯度向量展平为一维
        original_shape = gradient_vector.shape
        flatten_gradient_vector = gradient_vector.flatten()

        # 计算 L2 范数并进行归一化
        l2_norm = np.linalg.norm(flatten_gradient_vector, ord=2)
        if l2_norm == 0:  # 避免除以零
            l2_norm = 1.0
        normalized_gradient_vector = flatten_gradient_vector / l2_norm

        # 优化的量化过程
        quantized_gradient = np.zeros_like(normalized_gradient_vector, dtype=np.float32)

        # 预先计算概率矩阵，使用矢量化处理
        prob_values = np.zeros((normalized_gradient_vector.size, len(self.outputs)), dtype=np.float32)
        for idx, grad in enumerate(normalized_gradient_vector):
            prob_values[idx] = [prob(float(grad)) for prob in self.probabilities]
        prob_values = np.clip(prob_values, 0, 1)
        prob_sums = prob_values.sum(axis=-1, keepdims=True)
        prob_values /= prob_sums  # 归一化概率

        # 使用矢量化随机选择输出
        random_values = np.random.rand(normalized_gradient_vector.size, len(self.outputs))
        chosen_indices = (random_values < np.cumsum(prob_values, axis=-1)).argmax(axis=-1)
        quantized_gradient = self.outputs[chosen_indices]

        # 恢复为原始形状
        quantized_gradient = quantized_gradient.reshape(original_shape)
        #self.error_feedback[name] = gradient_vector - quantized_gradient

        # return quantized_gradient
        return quantized_gradient * l2_norm


class QSGD:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    # QSGD 量化函数
    def quantize(self, x, d):
        """Quantize the tensor x to d levels based on absolute value coefficient-wise."""
                # print("normalized_gradient:",normalized_gradient)
        if self.epsilon > 0:
            sensitivity = x.max() - x.min()
            x = DP.add_laplace(x, sensitivity, self.epsilon)
            normalized_gradient = DP.add_laplace(normalized_gradient, sensitivity, epsilon)

        sign = np.sign(x).astype(np.int8)  # 保留原向量的符号信息
        norm = np.sqrt(np.sum(np.square(x)))  # 计算原向量的 L2 范数
        normalized_gradient = x / norm  # 计算每个元素量化后的级别（归一化幅值）
        
        # print("DP normalized_gradient:",normalized_gradient)
        level_float = d * normalized_gradient
        # level_float = d * np.abs(x) / norm  # 计算每个元素量化后的级别（归一化幅值）
        previous_level = np.floor(level_float)
        is_next_level = np.random.rand(*x.shape) < (level_float - previous_level)
        new_level = previous_level + is_next_level
        new_level = new_level * sign  # 保留符号信息
        return new_level.astype(np.int8), norm  # 将符号应用到 new_level 上

    # 反量化函数
    def dequantize(self, new_level, norm, d):
        """Dequantize the tensor to original scale."""
        return norm * new_level / d
    

class TernGrad:
    def __init__(self, epsilon):
        self.epsilon = epsilon
        return
    
    def compress(self, tensor):
        shape = tensor.size()
        tensor = tensor.flatten()

        if self.epsilon > 0 :
            sensitivity = tensor.max() - tensor.min()
            tensor = DP.add_laplace(tensor, sensitivity, self.epsilon)

        std = (tensor - torch.mean(tensor)) ** 2
        std = torch.sqrt(torch.mean(std))
        c = 2.5 * std.item()
        gradient = torch.clamp(tensor, -c, c)
        abs_gradient = gradient.abs()
        scalar = abs_gradient.max()
        
        sign_gradient = gradient.sign() * scalar
        rnd_sample = torch.empty_like(tensor).uniform_(0, scalar.item())
        sign_gradient[rnd_sample >= abs_gradient] = 0
        new_sign = sign_gradient.sign()  # -1, 0, 1

        tensor_compressed = new_sign.type(torch.int8), scalar.flatten()

        return tensor_compressed, shape

    def decompress(self, tensor_compressed, shape):
        tensor_compressed, scalar = tensor_compressed
        sign = tensor_compressed.type(torch.float32)
        tensor_decompressed = sign * scalar
        return tensor_decompressed.view(shape)
    

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

        
    def local_quantize(self, name, param):
        # Add error feedback to local gradients
        local_gradients = param.grad.cpu().numpy()
        adjusted_gradients = local_gradients + self.error_feedback[name]

        # Quantize adjusted gradients
        quantized_local_gradients = self.quantizer.quantize(adjusted_gradients)

        # Calculate new error feedback
        self.error_feedback[name] = adjusted_gradients - quantized_local_gradients
        
        return quantized_local_gradients


    def apply_1bit_sgd_quantization(self, name, param):
        # Perform local quantization with error feedback
        quantized_local_gradients = self.local_quantize(name, param)
        return quantized_local_gradients
    


class TopK():
    def __init__(self, compress_ratio):
        self.compress_ratio = compress_ratio


    def sparsify(tensor, compress_ratio):
        tensor = tensor.flatten()
        k = max(1, int(tensor.numel() * compress_ratio))
        _, indices = torch.topk(tensor.abs(), k, sorted=False,)
        values = torch.gather(tensor, 0, indices)
        return values, indices


    def desparsify(values, indices, numel):
        tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
        tensor_decompressed.scatter_(0, indices, values)
        return tensor_decompressed

    def compress(self, tensor):
        values, indices = self.sparsify(tensor, self.compress_ratio)
        
        ctx = tensor.numel(), tensor.size()
        return values, indices, ctx

    def decompress(self, values, indices, ctx):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        numel, shape = ctx
        tensor_decompressed = self.desparsify(values, indices, numel)
        return tensor_decompressed.view(shape)
