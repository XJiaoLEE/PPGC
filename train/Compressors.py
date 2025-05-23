from sympy import symbols, exp, Rational, lambdify
import numpy as np
import torch
import DPmechanisms as DP

class PPGC:
    def __init__(self, epsilon, out_bits):
        self.epsilon = epsilon
        self.C = float((exp(epsilon) + 1) / (exp(epsilon) - 1))
        self.qutibit = out_bits
        self.outputs, self.probabilities = self.calculate_quantization(self.qutibit)
        # self.error_feedback = {}
        # for name, param in model.module.named_parameters() if hasattr(model, 'module') else model.named_parameters():
        #     self.error_feedback[name] = np.zeros(param.shape)


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

    # def compress(self, param):
    #     #---------在这里加入PPAGC的自适应模块，当quitbit=0时，进行自适应
    #     #根据每次输入的梯度，计算梯度的范围，然后计算quitbit，进而计算outputs和probabilities


    #     #gradient_vector = gradient_vector + self.error_feedback[name]

    #     # 将梯度向量展平为一维
    #     original_shape = param.shape
    #     flatten_gradient_vector = param.flatten()

    #     # 计算梯度绝对值的最大值并进行归一化
    #     max_abs_value = np.max(np.abs(flatten_gradient_vector))
    #     if max_abs_value == 0:  # 避免除以零
    #         max_abs_value = 1.0
    #     normalized_gradient_vector = flatten_gradient_vector / max_abs_value

    #     # # 计算 L2 范数并进行归一化
    #     # l2_norm = np.linalg.norm(flatten_gradient_vector, ord=2)
    #     # if l2_norm == 0:  # 避免除以零
    #     #     l2_norm = 1.0
    #     # normalized_gradient_vector = flatten_gradient_vector / l2_norm

    #     # 优化的量化过程
    #     quantized_gradient = np.zeros_like(normalized_gradient_vector, dtype=np.float32)

    #     # 预先计算概率矩阵，使用矢量化处理
    #     prob_values = np.zeros((normalized_gradient_vector.size, len(self.outputs)), dtype=np.float32)
    #     for idx, grad in enumerate(normalized_gradient_vector):
    #         prob_values[idx] = [prob(float(grad)) for prob in self.probabilities]
    #     prob_values = np.clip(prob_values, 0, 1)
    #     prob_sums = prob_values.sum(axis=-1, keepdims=True)
    #     prob_values /= prob_sums  # 归一化概率

    #     # 使用矢量化随机选择输出
    #     random_values = np.random.rand(normalized_gradient_vector.size, len(self.outputs))
    #     chosen_indices = (random_values < np.cumsum(prob_values, axis=-1)).argmax(axis=-1)
    #     quantized_gradient = self.outputs[chosen_indices]

    #     # 恢复为原始形状
    #     quantized_gradient = quantized_gradient.reshape(original_shape)
    #     #self.error_feedback[name] = gradient_vector - quantized_gradient

    #     return quantized_gradient
    #     # return quantized_gradient * l2_norm

    def compress(self, param,clipping_value=0):
        #---------在这里加入PPAGC的自适应模块，当quitbit=0时，进行自适应
        #根据每次输入的梯度，计算梯度的范围，然后计算quitbit，进而计算outputs和probabilities


        #gradient_vector = gradient_vector + self.error_feedback[name]

        # 将梯度向量展平为一维
        original_shape = param.shape
        flatten_gradient_vector = param.flatten()
        if clipping_value != 0:
            # flatten_gradient_vector = torch.clamp(flatten_gradient_vector,min=-clipping_value,max=clipping_value)
            flatten_gradient_vector = np.clip(flatten_gradient_vector,-clipping_value,clipping_value)
            normalized_gradient_vector = flatten_gradient_vector / clipping_value

        # 计算梯度绝对值的最大值并进行归一化
        else:
            max_abs_value = np.max(np.abs(flatten_gradient_vector))
            if max_abs_value == 0:  # 避免除以零
                max_abs_value = 1.0
            normalized_gradient_vector = flatten_gradient_vector / max_abs_value

        # # 计算 L2 范数并进行归一化
        # l2_norm = np.linalg.norm(flatten_gradient_vector, ord=2)
        # if l2_norm == 0:  # 避免除以零
        #     l2_norm = 1.0
        # normalized_gradient_vector = flatten_gradient_vector / l2_norm

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

        return quantized_gradient
        # return quantized_gradient * l2_norm



class QSGD:
    def __init__(self, epsilon, out_bits):
        self.epsilon = epsilon
        self.out_bits = out_bits

    # QSGD 量化函数
    def compress(self, x):
        """Quantize the tensor x to d levels based on absolute value coefficient-wise."""
                # print("normalized_gradient:",normalized_gradient)
        if self.epsilon > 0:
            sensitivity = x.max() - x.min()
            x = DP.add_laplace(x, sensitivity, self.epsilon)
            # normalized_gradient = DP.add_laplace(normalized_gradient, sensitivity, self.epsilon)

        norm = np.sqrt(np.sum(np.square(x)))
        level_float = self.out_bits * np.abs(x) / norm
        previous_level = np.floor(level_float)
        is_next_level = np.random.rand(*x.shape) < (level_float - previous_level)
        new_level = previous_level + is_next_level
        return norm * np.sign(x) *  new_level / self.out_bits
        

    # 反量化函数
    def dequantize(self, new_level, norm, d):
        """Dequantize the tensor to original scale."""
        return norm * new_level / d
    


        # return np.sign(x) *  new_level , norm


        # sign = np.sign(x).astype(np.int8)  # 保留原向量的符号信息
        # norm = np.sqrt(np.sum(np.square(x)))  # 计算原向量的 L2 范数
        # normalized_gradient = x / norm  # 计算每个元素量化后的级别（归一化幅值）
        
        # # print("DP normalized_gradient:",normalized_gradient)
        # level_float = d * normalized_gradient
        # # level_float = d * np.abs(x) / norm  # 计算每个元素量化后的级别（归一化幅值）
        # previous_level = np.floor(level_float)
        # is_next_level = np.random.rand(*x.shape) < (level_float - previous_level)
        # new_level = previous_level + is_next_level
        # new_level = new_level * sign  # 保留符号信息
        # return new_level.astype(np.int8), norm  # 将符号应用到 new_level 上
    
class TernGrad:
    def __init__(self, epsilon):
        self.epsilon = epsilon
        return
    
    def compress(self, param):
        param_np = param.grad.cpu().numpy()
        shape = param_np.shape
        """
        Compresses the given NumPy array using TernGrad technique.
        """
        if self.epsilon > 0:
            sensitivity = param_np.max() - param_np.min()
            param_np = DP.add_laplace(param_np, sensitivity, self.epsilon)

        # Flatten the tensor for easier processing
        tensor = param_np.flatten()
        
        # Calculate standard deviation and clamp the values
        mean = np.mean(tensor)
        std = np.sqrt(np.mean((tensor - mean) ** 2))
        c = 2.5 * std
        gradient = np.clip(tensor, -c, c)
        abs_gradient = np.abs(gradient)
        scalar = abs_gradient.max()
        
        # Generate the sign of gradient with a random threshold for sparsification
        sign_gradient = np.sign(gradient) * scalar
        rnd_sample = np.random.uniform(0, scalar, size=tensor.shape)
        sign_gradient[rnd_sample >= abs_gradient] = 0
        new_sign = np.sign(sign_gradient)  # -1, 0, 1
        return new_sign.reshape(shape)
        # tensor_compressed = new_sign, scalar
        # return tensor_compressed, tensor_np.shape

    def decompress(self, tensor_compressed, shape):
        """
        Decompresses the given compressed tensor to its original shape.
        """
        tensor_compressed, scalar = tensor_compressed
        sign = tensor_compressed.astype(np.float32)
        tensor_decompressed = sign * scalar
        return tensor_decompressed.reshape(shape)



# class TernGrad:
#     def __init__(self, epsilon):
#         self.epsilon = epsilon
#         return
    
#     def compress(self, tensor):

#         # if self.epsilon > 0 :
#         #     sensitivity = tensor.max() - tensor.min()
#         #     tensor = DP.add_laplace(tensor, sensitivity, self.epsilon)
#         if self.epsilon > 0:
#             # 将 tensor 转换为 NumPy 数组
#             tensor_np = tensor.cpu().numpy()
#             sensitivity = tensor_np.max() - tensor_np.min()
#             # 添加拉普拉斯噪声
#             tensor_np = DP.add_laplace(tensor_np, sensitivity, self.epsilon)
#             # 将添加噪声后的 NumPy 数组转换回 PyTorch 张量
#             # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#             tensor = torch.tensor(tensor_np, dtype=tensor.dtype, device=tensor.device)

        
#         shape = tensor.size()
#         tensor = tensor.flatten()

#         std = (tensor - torch.mean(tensor)) ** 2
#         std = torch.sqrt(torch.mean(std))
#         c = 2.5 * std.item()
#         gradient = torch.clamp(tensor, -c, c)
#         abs_gradient = gradient.abs()
#         scalar = abs_gradient.max()
        
#         sign_gradient = gradient.sign() * scalar
#         rnd_sample = torch.empty_like(tensor).uniform_(0, scalar.item())
#         sign_gradient[rnd_sample >= abs_gradient] = 0
#         new_sign = sign_gradient.sign()  # -1, 0, 1

#         tensor_compressed = new_sign, scalar.flatten()
#         # tensor_compressed = new_sign.type(torch.int8), scalar.flatten()

#         return tensor_compressed, shape

#     def decompress(self, tensor_compressed, shape):
#         tensor_compressed, scalar = tensor_compressed
#         sign = tensor_compressed.type(torch.float32)
#         tensor_decompressed = sign * scalar
#         return tensor_decompressed.view(shape)
    

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

    def local_quantize(self, local_gradients):
        # add noise
        if self.epsilon > 0:
            sensitivity = local_gradients.max() - local_gradients.min()
            local_gradients = DP.add_laplace(local_gradients, sensitivity, self.epsilon)

        # Quantize adjusted gradients
        quantized_local_gradients = self.quantizer.quantize(local_gradients)
        
        return quantized_local_gradients

    def compress(self, local_gradients):
        # Perform local quantization with error feedback
        quantized_local_gradients = self.local_quantize(local_gradients)
        return quantized_local_gradients
    

    # def local_quantize(self, name, param):
    #     # Add error feedback to local gradients
    #     local_gradients = param.grad.cpu().numpy()

    #     adjusted_gradients = local_gradients + self.error_feedback[name]

    #     # add noise
    #     if self.epsilon > 0:
    #         sensitivity = adjusted_gradients.max() - adjusted_gradients.min()
    #         adjusted_gradients = DP.add_laplace(adjusted_gradients, sensitivity, self.epsilon)

    #     # Quantize adjusted gradients
    #     quantized_local_gradients = self.quantizer.quantize(adjusted_gradients)

    #     # Calculate new error feedback
    #     self.error_feedback[name] = adjusted_gradients - quantized_local_gradients
        
    #     return quantized_local_gradients


    # def compress(self, name, param):
    #     # Perform local quantization with error feedback
    #     quantized_local_gradients = self.local_quantize(name, param)
    #     return quantized_local_gradients
    

class TopK:
    def __init__(self, compress_ratio):
        self.compress_ratio = compress_ratio

    def sparsify(self, param_np):
        # param_np = param_np.flatten()
        k = max(1, int(param_np.size * self.compress_ratio))
        indices = np.argpartition(np.abs(param_np), -k)[-k:]
        values = param_np[indices]
        return values, indices

    def desparsify(self, values, indices, numel):
        tensor_decompressed = np.zeros(numel, dtype=values.dtype)
        tensor_decompressed[indices] = values
        return tensor_decompressed

    def compress(self, param_np):
        values, indices = self.sparsify(param_np)
        ctx = param_np.size, param_np.shape
        return values, indices, ctx

    def decompress(self, values, indices, ctx):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        numel, shape = ctx
        tensor_decompressed = self.desparsify(values, indices, numel)
        return tensor_decompressed.reshape(shape)

# class TopK():
#     def __init__(self, compress_ratio):
#         self.compress_ratio = compress_ratio


#     def sparsify(tensor):
#         tensor = tensor.flatten()
#         k = max(1, int(tensor.numel() * self.compress_ratio))
#         _, indices = torch.topk(tensor.abs(), k, sorted=False,)
#         values = torch.gather(tensor, 0, indices)
#         return values, indices


#     def desparsify(values, indices, numel):
#         tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
#         tensor_decompressed.scatter_(0, indices, values)
#         return tensor_decompressed

#     def compress(self, tensor):
#         values, indices = self.sparsify(tensor, self.compress_ratio)
        
#         ctx = tensor.numel(), tensor.size()
#         return values, indices, ctx

#     def decompress(self, values, indices, ctx):
#         """Decompress by filling empty slots with zeros and reshape back using the original shape"""
#         numel, shape = ctx
#         tensor_decompressed = self.desparsify(values, indices, numel)
#         return tensor_decompressed.view(shape)



# q = [0, 1, 0.5, 0.2, -0.3, -0.9]
# q = np.array(q)
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # q = tensor = torch.tensor(q, device=device)
# terngrad_ = TopK(0.5)
# q1 , indices= terngrad_.sparsify(q)
# grad_np = np.zeros_like(q )
# grad_np[indices] = q1 
# p = terngrad_.compress(q)
# print("量化后的级别:", q1)
# print("量化后的级别:", p)
# print("量化后的级别:", grad_np)