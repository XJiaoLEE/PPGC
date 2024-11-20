import numpy as np

class QSGD:
    def __init__(self):
        pass

    # QSGD 量化函数
    def quantize(self, x, d):
        """Quantize the tensor x to d levels based on absolute value coefficient-wise."""
        sign = np.sign(x)  # 保留原向量的符号信息
        norm = np.sqrt(np.sum(np.square(x)))  # 计算原向量的 L2 范数
        level_float = d * np.abs(x) / norm  # 计算每个元素量化后的级别（归一化幅值）
        previous_level = np.floor(level_float)
        is_next_level = np.random.rand(*x.shape) < (level_float - previous_level)
        new_level = previous_level + is_next_level
        return sign * new_level, norm  # 将符号应用到 new_level 上

    # 反量化函数
    def dequantize(self, new_level, norm, d):
        """Dequantize the tensor to original scale."""
        return norm * new_level / d

# 示例代码
q = [0, 1, 0.5, 0.2, -0.3, -0.9]
q = np.array(q)
qsgd = QSGD()
new_level, norm = qsgd.quantize(q, 1)
print("量化后的级别:", new_level)
print("原向量的范数:", norm)
print("反量化后的结果:", qsgd.dequantize(new_level, norm, 1))
