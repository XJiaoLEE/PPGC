# from sympy import symbols, exp, Rational
# import numpy as np

# class PPGC:
#     def __init__(self, epsilon,out_bits):
#         self.epsilon = epsilon
#         self.C = float((exp(epsilon) + 1) / (exp(epsilon) - 1))
#         self.qutibit = out_bits
#         self.outputs, self.probabilities = self.calculate_quantization(self.qutibit)

#     def calculate_quantization(self, qutibit):
#         outbit = 2 ** qutibit
#         x = symbols('x')
#         probabilities, outputs, pa_i = [], [], []

#         for i in range(1, outbit // 2 + 1):
#             output_value = (outbit - (2 * i - 1)) * self.C
#             outputs.append(float(output_value))
#             abottom_i = 2 * (outbit // 2) * (outbit - (i * 2) + 1)
#             a_i = (float(exp(self.epsilon) - 1) / (abottom_i * float(exp(self.epsilon) + 1)))
#             pa_i.append(a_i)
#             p_i = x * a_i + Rational(1, outbit)
#             probabilities.append(p_i)

#         for i in range(outbit // 2):
#             outputs.append(-outputs[outbit // 2 - i - 1])
#             a_i = -pa_i[outbit // 2 - i - 1]
#             p_i = x * a_i + Rational(1, outbit)
#             probabilities.append(p_i)

#         return outputs, probabilities

#     def map_gradient(self, gradient_vector, out_bits):
#         # outputs, probabilities = self.calculate_quantization(out_bits)
#         l2_norm = np.linalg.norm(gradient_vector.flatten(), ord=2)
#         if l2_norm == 0:  # 避免除以零
#             l2_norm = 1.0
#         normalized_gradient_vector = gradient_vector / l2_norm
#         quantized_gradient = np.zeros_like(normalized_gradient_vector, dtype=np.float32)

#         # 优化为一次性符号替换
#         for idx, grad in np.ndenumerate(gradient_vector):
#             prob_values = np.array([float(prob.subs('x', grad)) for prob in self.probabilities], dtype=np.float64)
#             #prob_values /= prob_values.sum()

#             # 矢量化随机选择
#             quantized_gradient[idx] = np.random.choice(self.outputs, p=prob_values)

#         return quantized_gradient


from sympy import symbols, exp, Rational, lambdify
import numpy as np

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
        # l2_norm = np.linalg.norm(flatten_gradient_vector, ord=2)
        l2_norm = np.sqrt(np.sum(np.square(original_shape)))
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

        return quantized_gradient * l2_norm

