from sympy import symbols, exp, Rational, lambdify
import numpy as np


# # QSGD量化函数
# def quantize(x, d):
#     """Quantize the tensor x to d levels based on absolute value coefficient-wise."""
#     norm = np.sqrt(np.sum(np.square(x)))
#     level_float = d * np.abs(x) / norm
#     previous_level = np.floor(level_float)
#     is_next_level = np.random.rand(*x.shape) < (level_float - previous_level)
#     new_level = previous_level + is_next_level
#     return np.sign(x) * norm * new_level / d

class QSGD:
    def __init__(self):
        pass
    # QSGD量化函数
    def quantize(self, x, d):
        """Quantize the tensor x to d levels based on absolute value coefficient-wise."""
        norm = np.sqrt(np.sum(np.square(x)))
        level_float = d * np.abs(x) / norm
        previous_level = np.floor(level_float)
        is_next_level = np.random.rand(*x.shape) < (level_float - previous_level)
        new_level = previous_level + is_next_level
        return new_level, norm

    # 反量化函数
    def dequantize(self, new_level, norm, d):
        """Dequantize the tensor to original scale."""
        return np.sign(new_level) * norm * new_level / d

q=[0,1,0.5,0.2,-0.3,-0.9]
q = np.array(q) 
qsgd = QSGD()
new_level, norm = qsgd.quantize(q,1)
print(new_level, norm )
print(qsgd.dequantize(new_level, norm ,1))
