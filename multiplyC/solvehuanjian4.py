import numpy as np
from sympy import symbols, exp, Rational

# 定义符号变量
a2, x, epsilon = symbols('a2 x epsilon')

# 定义每个不等式右侧的表达式
R1 = (2*x*exp(epsilon) - 2*x - 3*exp(2*epsilon) + 3) / (4*x * (3*exp(2*epsilon) + 4*exp(epsilon) + 1))
R2 = (exp(epsilon) - 1) / (4*x * (exp(epsilon) + 1))
R3 = (2*x*exp(2*epsilon) - 2*x*exp(epsilon) - 3*exp(2*epsilon) + 3) / (4*x * (exp(2*epsilon) + 4*exp(epsilon) + 3))

# 定义 a1 与 a2 的关系
a1_relation = (-2 * a2 * exp(epsilon) - 2 * a2 + exp(epsilon) - 1) / (6 * (exp(epsilon) + 1))

# 定义四个概率的表达式
p1 = a1_relation * x + Rational(1, 4)
p2 = a2 * x + Rational(1, 4)
p3 = -a2 * x + Rational(1, 4)
p4 = -a1_relation * x + Rational(1, 4)

# 转换为数值函数以便测试
R1_func = lambda x_val, eps_val: R1.subs({x: x_val, epsilon: eps_val}).evalf()
R2_func = lambda x_val, eps_val: R2.subs({x: x_val, epsilon: eps_val}).evalf()
R3_func = lambda x_val, eps_val: R3.subs({x: x_val, epsilon: eps_val}).evalf()

p1_func = lambda a2_val, x_val, eps_val: p1.subs({a2: a2_val, x: x_val, epsilon: eps_val}).evalf()
p2_func = lambda a2_val, x_val, eps_val: p2.subs({a2: a2_val, x: x_val, epsilon: eps_val}).evalf()
p3_func = lambda a2_val, x_val, eps_val: p3.subs({a2: a2_val, x: x_val, epsilon: eps_val}).evalf()
p4_func = lambda a2_val, x_val, eps_val: p4.subs({a2: a2_val, x: x_val, epsilon: eps_val}).evalf()

# 设置 x 和 epsilon 的测试值范围
x_values = np.linspace(-1, 1, 20)
x_values = x_values[x_values != 0]  # 移除 x=0
epsilon_values = np.linspace(0.1, 5, 20)  # 取正的 epsilon 值

# 记录最小的上界
min_upper_bound = float('inf')
best_x = None
best_epsilon = None
best_R_values = None

# 逐一测试所有组合
for x_val in x_values:
    for eps_val in epsilon_values:
        # 计算当前组合下的 R1, R2, R3
        R1_val = R1_func(x_val, eps_val)
        R2_val = R2_func(x_val, eps_val)
        R3_val = R3_func(x_val, eps_val)
        
        # 取最小值作为当前 a2 的上界
        current_min = min(R1_val, R2_val, R3_val)
        
        # 检查 p1, p2, p3, p4 是否为非负数
        if all(p >= 0 for p in [p1_func(current_min, x_val, eps_val),
                                p2_func(current_min, x_val, eps_val),
                                p3_func(current_min, x_val, eps_val),
                                p4_func(current_min, x_val, eps_val)]):
            # 更新最小上界和对应的 x, epsilon 值
            if current_min < min_upper_bound:
                min_upper_bound = current_min
                best_x = x_val
                best_epsilon = eps_val
                best_R_values = (R1_val, R2_val, R3_val)

# 输出结果
print(f"最终的 a2 上界: {min_upper_bound}")
print(f"对应的 x 值: {best_x}")
print(f"对应的 epsilon 值: {best_epsilon}")
print(f"此时的 R1, R2, R3 值: {best_R_values}")
