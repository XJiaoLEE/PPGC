from sympy import symbols, Eq, solve, exp, Rational, simplify
import numpy as np

# 定义符号变量
a1, a2, x, epsilon = symbols('a1 a2 x epsilon')

# 定义 C 的取值，保持符号化处理
C = (exp(epsilon) + 1) / (exp(epsilon) - 1)

# 定义四个概率的表达式
p1 = a1 * x + Rational(1, 4)
p2 = a2 * x + Rational(1, 4)
p3 = -a2 * x + Rational(1, 4)
p4 = -a1 * x + Rational(1, 4)

# 期望条件：p1 * 3C + p2 * C + p3 * (-C) + p4 * (-3C) = x
output_condition = Eq(p1 * C + p2 * C *1/3+ p3 * (-C*1/3) + p4 * (- C), x)

# 根据期望条件求解 a1 和 a2 的关系
a1_a2_relation = solve(output_condition, a1)[0]
print("a1 和 a2 的关系:", a1_a2_relation)

# # 添加非负性约束
# non_negative_constraints = [
#     p1.subs(a1, a1_a2_relation) >= 0,
#     p2 >= 0,
#     p3 >= 0,
#     p4.subs(a1, a1_a2_relation) >= 0
# ]




# # 将 a1 的表达式代入隐私不等式并手动化简
# # 隐私不等式 1： p1 <= p2 * e^epsilon
# privacy_inequality_1 = (p1 <= p2 * exp(epsilon)).subs(a1, a1_a2_relation)
# privacy_inequality_1 = privacy_inequality_1.rewrite('Add').simplify()  # 进一步化简

# # 手动移项，将 a2 单独放在一侧
# lhs_1 = privacy_inequality_1.lhs - privacy_inequality_1.rhs
# privacy_inequality_1_solved = solve(lhs_1, a2)

# # 隐私不等式 2： p2 <= p3 * e^epsilon
# privacy_inequality_2 = (p2 <= p3 * exp(epsilon)).subs(a1, a1_a2_relation)
# privacy_inequality_2 = privacy_inequality_2.rewrite('Add').simplify()  # 进一步化简

# # 手动移项，将 a2 单独放在一侧
# lhs_2 = privacy_inequality_2.lhs - privacy_inequality_2.rhs
# privacy_inequality_2_solved = solve(lhs_2, a2)

# # 隐私不等式 3： p3 <= p4 * e^epsilon
# privacy_inequality_3 = (p3 <= p4 * exp(epsilon)).subs(a1, a1_a2_relation)
# privacy_inequality_3 = privacy_inequality_3.rewrite('Add').simplify()  # 进一步化简

# # 手动移项，将 a2 单独放在一侧
# lhs_3 = privacy_inequality_3.lhs - privacy_inequality_3.rhs
# privacy_inequality_3_solved = solve(lhs_3, a2)


# # 输出 a2 与 x 和 epsilon 的关系
# print("a2 与 x 和 epsilon 的关系（从隐私不等式 1 得到）:", privacy_inequality_1_solved)
# print("a2 与 x 和 epsilon 的关系（从隐私不等式 2 得到）:", privacy_inequality_2_solved)
# print("a2 与 x 和 epsilon 的关系（从隐私不等式 3 得到）:", privacy_inequality_3_solved)



# 将 a1 替换为关于 a2 的表达式，以便后续处理
p1_sub = p1.subs(a1, a1_a2_relation)
p4_sub = p4.subs(a1, a1_a2_relation)

# 隐私不等式和非负性约束
privacy_inequality_1 = p1_sub <= p2 * exp(epsilon)
privacy_inequality_2 = p2 <= p3 * exp(epsilon)
privacy_inequality_3 = p3 <= p4_sub * exp(epsilon)

# 非负性约束的表达式
non_negative_constraints = [
    p1_sub >= 0,
    p2 >= 0,
    p3 >= 0,
    p4_sub >= 0
]

# 函数：求解不等式的 a2 范围
def solve_a2_inequality(inequality):
    lhs = inequality.lhs - inequality.rhs  # 将不等式移项
    return solve(lhs, a2)

# 对每个隐私不等式和非负性约束求解 a2 范围
privacy_inequality_1_solved = solve_a2_inequality(privacy_inequality_1)
privacy_inequality_2_solved = solve_a2_inequality(privacy_inequality_2)
privacy_inequality_3_solved = solve_a2_inequality(privacy_inequality_3)

non_negative_solutions = [solve_a2_inequality(constraint) for constraint in non_negative_constraints]

# 输出 a2 的解
print("a2 与 x 和 epsilon 的关系（从隐私不等式 1 得到）:", privacy_inequality_1_solved)
print("a2 与 x 和 epsilon 的关系（从隐私不等式 2 得到）:", privacy_inequality_2_solved)
print("a2 与 x 和 epsilon 的关系（从隐私不等式 3 得到）:", privacy_inequality_3_solved)
print("\na2 的范围（非负性约束）:")
for i, sol in enumerate(non_negative_solutions, start=1):
    print(f"非负性约束 {i}: {sol}")


# 定义符号变量
a2, x, epsilon = symbols('a2 x epsilon')

# 定义每个不等式右侧的表达式
R1 = privacy_inequality_1_solved[0]
R2 = privacy_inequality_2_solved[0]
R3 = privacy_inequality_3_solved[0]

# 定义 a1 与 a2 的关系
a1_relation = a1_a2_relation

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
best_R = 0

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
                best_R = [R1_val, R2_val, R3_val].index(min(R1_val, R2_val, R3_val)) + 1

# 输出结果
print(f"最终的 a2 上界: {min_upper_bound}")
print(f"对应的 x 值: {best_x}")
print(f"对应的 epsilon 值: {best_epsilon}")
print(f"此时的 R1, R2, R3 值: {best_R_values}")
print(f"此时的 best_R 值: {best_R}")

# a2_final  = [R1,R2,R3][best_R-1].rewrite(exp).subs(x,0.5)
a2_final  = [R1,R2,R3][best_R-1].subs(x,-1)
print(f"最终的 a2: {a2_final }")
# a1_final  = a1_a2_relation.subs(a2,a2_final ).simplify().rewrite(exp)
a1_final  = a1_a2_relation.subs(a2,a2_final ).simplify()
print(f"最终的 a1: {a1_final }")