import numpy as np
import DPmechanisms as DP

class QSGD:
    def __init__(self):
        pass

    # QSGD 量化函数
    def quantize(self, x, d, epsilon = 0):
        """Quantize the tensor x to d levels based on absolute value coefficient-wise."""
        sign = np.sign(x).astype(np.int8)  # 保留原向量的符号信息
        norm = np.sqrt(np.sum(np.square(x)))  # 计算原向量的 L2 范数
        normalized_gradient = x / norm  # 计算每个元素量化后的级别（归一化幅值）
        
        print("normalized_gradient:",normalized_gradient)
        if epsilon > 0:
            normalized_gradient = DP.add_laplace(normalized_gradient, 2, epsilon)
            norm1 = np.sqrt(np.sum(np.square(normalized_gradient)))  # 计算原向量的 L2 范数
            normalized_gradient = normalized_gradient / norm1  # 计算每个元素量化后的级别（归一化幅值）
        print("DP normalized_gradient:",normalized_gradient)
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

# 示例代码
q = [0, 1, 0.5, 0.2, -0.3, -0.9]
q = np.array(q)
qsgd = QSGD()
new_level, norm = qsgd.quantize(q, 1,1)
print("量化后的级别:", new_level)
print("原向量的范数:", norm)
print("反量化后的结果:", qsgd.dequantize(new_level, norm, 1))



# import numpy as np

# class QSGD:
#     def __init__(self):
#         pass

#     # QSGD量化函数
#     def quantize(self, x, d, out_bits):
#         """Quantize the tensor x to d levels based on absolute value coefficient-wise."""
#         sign = np.sign(x).astype(np.int8)  # 保留符号信息
#         norm = np.sqrt(np.sum(np.square(x)))  # 计算原向量的 L2 范数
#         level_float = d * np.abs(x) / norm
#         previous_level = np.floor(level_float)
#         is_next_level = np.random.rand(*x.shape) < (level_float - previous_level)
#         new_level = previous_level + is_next_level
#         new_level = new_level.astype(np.int8)  # 转换为 int8 类型
#         return sign, new_level, norm

#     # 将量化后的梯度进行自定义编码以提升传输效率
#     def encode(self, sign, new_level, out_bits):
#         """Custom encode the quantized gradient into a more compact representation."""
#         # 使用符号位和 out_bits 位来编码每个量化值
#         encoded_bits = []
#         for s, level in zip(sign, new_level):
#             # 符号位：0 表示正，1 表示负
#             sign_bit = '0' if s >= 0 else '1'
#             # 量化级别位，使用 out_bits 位表示
#             level_bits = format(abs(level), f'0{out_bits}b')
#             encoded_bits.append(sign_bit + level_bits)

#         # 将位字符串合并为字节（8 位）
#         encoded_string = ''.join(encoded_bits)
#         # 将位字符串分割为 8 位长的片段，并转换为字节
#         byte_array = bytearray(int(encoded_string[i:i + 8], 2) for i in range(0, len(encoded_string), 8))
#         return byte_array

#     # 解码函数
#     def decode(self, byte_array, original_length, out_bits):
#         """Decode the compact representation back to the original quantized gradient."""
#         # 将字节数组转换为位字符串
#         bit_string = ''.join(f'{byte:08b}' for byte in byte_array)
#         quantized_gradient = []
#         sign = []
#         # 每 (1 + out_bits) 位对应一个量化后的值
#         step = 1 + out_bits
#         for i in range(0, original_length * step, step):
#             sign_bit = bit_string[i]
#             level_bits = bit_string[i + 1:i + step]
#             # 解码符号位
#             s = 1 if sign_bit == '0' else -1
#             sign.append(s)
#             # 解码量化级别
#             level = int(level_bits, 2)
#             quantized_gradient.append(s * level)

#         return np.array(quantized_gradient, dtype=np.int8)

#     # 反量化函数
#     def dequantize(self, new_level, norm, d):
#         """Dequantize the tensor to original scale."""
#         return norm * new_level / d

# # 示例代码
# q = [0, 1, 0.5, 0.2, -0.3, -0.9]
# q = np.array(q)
# qsgd = QSGD()
# out_bits = 1
# sign, new_level, norm = qsgd.quantize(q, 1, out_bits)
# print("量化后的符号位:", sign)
# print("量化后的级别:", new_level)
# print("原向量的范数:", norm)

# encoded = qsgd.encode(sign, new_level, out_bits)
# print("编码后的字节:", encoded)

# decoded = qsgd.decode(encoded, len(new_level), out_bits)
# print("解码后的量化级别:", decoded)

# # 反量化后的结果
# dequantized = qsgd.dequantize(decoded, norm, 1)
# print("反量化后的结果:", dequantized)
