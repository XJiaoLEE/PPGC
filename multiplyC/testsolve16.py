import numpy as np
from sympy import symbols, exp, Rational

# 定义符号变量
a1, a2, a3, a4, a5, a6, a7, a8, x, epsilon = symbols('a1 a2 a3 a4 a5 a6 a7 a8 x epsilon')

# 确定 a1 到 a8 的表达式
a1_value = (exp(epsilon) - 1) / (2 * 8 * 15 * (exp(epsilon) + 1))
a2_value = (exp(epsilon) - 1) / (2 * 8 * 13 * (exp(epsilon) + 1))
a3_value = (exp(epsilon) - 1) / (2 * 8 * 11 * (exp(epsilon) + 1))
a4_value = (exp(epsilon) - 1) / (2 * 8 * 9 * (exp(epsilon) + 1))
a5_value = (exp(epsilon) - 1) / (2 * 8 * 7 * (exp(epsilon) + 1))
a6_value = (exp(epsilon) - 1) / (2 * 8 * 5 * (exp(epsilon) + 1))
a7_value = (exp(epsilon) - 1) / (2 * 8 * 3 * (exp(epsilon) + 1))
a8_value = (exp(epsilon) - 1) / (2 * 8 * 1 * (exp(epsilon) + 1))

# 定义16个概率的表达式
p1 = a1_value * x + Rational(1, 16)
p2 = a2_value * x + Rational(1, 16)
p3 = a3_value * x + Rational(1, 16)
p4 = a4_value * x + Rational(1, 16)
p5 = a5_value * x + Rational(1, 16)
p6 = a6_value * x + Rational(1, 16)
p7 = a7_value * x + Rational(1, 16)
p8 = a8_value * x + Rational(1, 16)
p9 = -a8_value * x + Rational(1, 16)
p10 = -a7_value * x + Rational(1, 16)
p11 = -a6_value * x + Rational(1, 16)
p12 = -a5_value * x + Rational(1, 16)
p13 = -a4_value * x + Rational(1, 16)
p14 = -a3_value * x + Rational(1, 16)
p15 = -a2_value * x + Rational(1, 16)
p16 = -a1_value * x + Rational(1, 16)

# 检查的隐私不等式条件 (将分母移到右侧)
privacy_inequality_1 = p1 <= p2 * exp(epsilon)
privacy_inequality_2 = p2 <= p3 * exp(epsilon)
privacy_inequality_3 = p3 <= p4 * exp(epsilon)
privacy_inequality_4 = p4 <= p5 * exp(epsilon)
privacy_inequality_5 = p5 <= p6 * exp(epsilon)
privacy_inequality_6 = p6 <= p7 * exp(epsilon)
privacy_inequality_7 = p7 <= p8 * exp(epsilon)
privacy_inequality_8 = p8 <= p9 * exp(epsilon)
privacy_inequality_9 = p9 <= p10 * exp(epsilon)
privacy_inequality_10 = p10 <= p11 * exp(epsilon)
privacy_inequality_11 = p11 <= p12 * exp(epsilon)
privacy_inequality_12 = p12 <= p13 * exp(epsilon)
privacy_inequality_13 = p13 <= p14 * exp(epsilon)
privacy_inequality_14 = p14 <= p15 * exp(epsilon)
privacy_inequality_15 = p15 <= p16 * exp(epsilon)

# 转换为数值函数以便测试
p_funcs = [lambda x_val, eps_val, i=i: eval(f'p{i+1}').subs({x: x_val, epsilon: eps_val}).evalf() for i in range(16)]
privacy_funcs = [
    lambda x_val, eps_val, i=i: p_funcs[i](x_val, eps_val) <= p_funcs[i+1](x_val, eps_val) * np.exp(eps_val)
    for i in range(15)
]

# 设置 x 和 epsilon 的测试值范围
x_values = np.linspace(-1, 0.99, 20)
epsilon_values = np.linspace(0.1, 5, 20)

# 记录是否所有条件都满足
all_conditions_met = True

# 逐一测试所有组合
for x_val in x_values:
    for eps_val in epsilon_values:
        # 检查概率是否为非负
        p_values = [f(x_val, eps_val) for f in p_funcs]
        if any(p < 0 for p in p_values):
            print(f"失败: 在 x = {x_val}, epsilon = {eps_val} 时，概率为负")
            all_conditions_met = False
            break

        # 检查隐私不等式是否满足
        if not all(f(x_val, eps_val) for f in privacy_funcs):
            print(f"失败: 在 x = {x_val}, epsilon = {eps_val} 时，隐私不等式不满足")
            all_conditions_met = False
            break
    if not all_conditions_met:
        break

# 最终结果
if all_conditions_met:
    print("成功: 所有不等式在所有测试值下均满足")
else:
    print("验证失败: 存在不满足条件的测试值")
