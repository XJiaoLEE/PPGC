from sympy import symbols, exp, Rational
import numpy as np

class PPGC:
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.C = float((exp(epsilon) + 1) / (exp(epsilon) - 1))

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
            probabilities.append(p_i)

        for i in range(outbit // 2):
            outputs.append(-outputs[outbit // 2 - i - 1])
            a_i = -pa_i[outbit // 2 - i - 1]
            p_i = x * a_i + Rational(1, outbit)
            probabilities.append(p_i)

        return outputs, probabilities

    def map_gradient(self, gradient_vector, out_bits):
        outputs, probabilities = self.calculate_quantization(out_bits)
        quantized_gradient = np.zeros_like(gradient_vector, dtype=np.float32)

        # 优化为一次性符号替换
        for idx, grad in np.ndenumerate(gradient_vector):
            prob_values = np.array([float(prob.subs('x', grad)) for prob in probabilities], dtype=np.float64)
            #prob_values /= prob_values.sum()

            # 矢量化随机选择
            quantized_gradient[idx] = np.random.choice(outputs, p=prob_values)

        return quantized_gradient
