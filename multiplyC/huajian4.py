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
output_condition = Eq(p1 * 3 * C + p2 * C + p3 * (-C) + p4 * (-3 * C), x)

# 根据期望条件求解 a1 和 a2 的关系
a1_a2_relation = solve(output_condition, a1)[0]
print("a1 和 a2 的关系:", a1_a2_relation)

# 将 a1 的表达式代入隐私不等式并手动化简
# 隐私不等式 1： p1 <= p2 * e^epsilon
privacy_inequality_1 = (p1 <= p2 * exp(epsilon)).subs(a1, a1_a2_relation)
privacy_inequality_1 = privacy_inequality_1.rewrite('Add').simplify()  # 进一步化简

# 手动移项，将 a2 单独放在一侧
lhs_1 = privacy_inequality_1.lhs - privacy_inequality_1.rhs
privacy_inequality_1_solved = solve(lhs_1, a2)

# 隐私不等式 2： p2 <= p3 * e^epsilon
privacy_inequality_2 = (p2 <= p3 * exp(epsilon)).subs(a1, a1_a2_relation)
privacy_inequality_2 = privacy_inequality_2.rewrite('Add').simplify()  # 进一步化简

# 手动移项，将 a2 单独放在一侧
lhs_2 = privacy_inequality_2.lhs - privacy_inequality_2.rhs
privacy_inequality_2_solved = solve(lhs_2, a2)

# 隐私不等式 3： p3 <= p4 * e^epsilon
privacy_inequality_3 = (p3 <= p4 * exp(epsilon)).subs(a1, a1_a2_relation)
privacy_inequality_3 = privacy_inequality_3.rewrite('Add').simplify()  # 进一步化简

# 手动移项，将 a2 单独放在一侧
lhs_3 = privacy_inequality_3.lhs - privacy_inequality_3.rhs
privacy_inequality_3_solved = solve(lhs_3, a2)

# 输出 a2 与 x 和 epsilon 的关系
print("a2 与 x 和 epsilon 的关系（从隐私不等式 1 得到）:", privacy_inequality_1_solved)
print("a2 与 x 和 epsilon 的关系（从隐私不等式 2 得到）:", privacy_inequality_2_solved)
print("a2 与 x 和 epsilon 的关系（从隐私不等式 3 得到）:", privacy_inequality_3_solved)



a2_final = privacy_inequality_2_solved[0].subs(x,10)  #.rewrite('exp')
print("a2_final:",a2_final)
a1_final = a1_a2_relation.subs(a2,a2_final)   #.rewrite('exp')
print("a1_final:",a1_final)



# 确定 a2 和 a1 的表达式
C = (exp(epsilon) + 1) / (exp(epsilon) - 1)
a2_value = a2_final
a1_value = a1_final

# 定义四个概率的表达式
p1 = a1_value * x + Rational(1, 4)
p2 = a2_value * x + Rational(1, 4)
p3 = -a2_value * x + Rational(1, 4)
p4 = -a1_value * x + Rational(1, 4)

# 检查的隐私不等式条件 (将分母移到右侧)
privacy_inequality_1 = p1 <= p2 * exp(epsilon)
privacy_inequality_2 = p2 <= p3 * exp(epsilon)
privacy_inequality_3 = p3 <= p4 * exp(epsilon)

# 转换为数值函数以便测试
p1_func = lambda x_val, eps_val: p1.subs({x: x_val, epsilon: eps_val}).evalf()
p2_func = lambda x_val, eps_val: p2.subs({x: x_val, epsilon: eps_val}).evalf()
p3_func = lambda x_val, eps_val: p3.subs({x: x_val, epsilon: eps_val}).evalf()
p4_func = lambda x_val, eps_val: p4.subs({x: x_val, epsilon: eps_val}).evalf()

privacy_inequality_1_func = lambda x_val, eps_val: p1_func(x_val, eps_val) <= p2_func(x_val, eps_val) * np.exp(eps_val)
privacy_inequality_2_func = lambda x_val, eps_val: p2_func(x_val, eps_val) <= p3_func(x_val, eps_val) * np.exp(eps_val)
privacy_inequality_3_func = lambda x_val, eps_val: p3_func(x_val, eps_val) <= p4_func(x_val, eps_val) * np.exp(eps_val)

# 设置 x 和 epsilon 的测试值范围
x_values = np.linspace(-1, 0.99, 20)
#x_values = x_values[x_values != 0]  # 移除 x=0
epsilon_values = np.linspace(0.1, 5, 20)

# 记录是否所有条件都满足
all_conditions_met = True

# 逐一测试所有组合
for x_val in x_values:
    for eps_val in epsilon_values:
        # 检查概率是否为非负
        p1_val = p1_func(x_val, eps_val)
        p2_val = p2_func(x_val, eps_val)
        p3_val = p3_func(x_val, eps_val)
        p4_val = p4_func(x_val, eps_val)
        
        # 检查是否满足概率非负
        if any(p < 0 for p in [p1_val, p2_val, p3_val, p4_val]):
            print(f"失败: 在 x = {x_val}, epsilon = {eps_val} 时，概率为负")
            all_conditions_met = False
            break

        # 检查隐私不等式是否满足
        if not (privacy_inequality_1_func(x_val, eps_val) and privacy_inequality_2_func(x_val, eps_val) and privacy_inequality_3_func(x_val, eps_val)):
            print(f"失败: 在 x = {x_val}, epsilon = {eps_val} 时，隐私不等式不满足")
            print(privacy_inequality_1_func(x_val, eps_val))
            print(privacy_inequality_2_func(x_val, eps_val),x_val,eps_val)
            print(privacy_inequality_3_func(x_val, eps_val))
            all_conditions_met = False
            break
    if not all_conditions_met:
        break

# 最终结果
if all_conditions_met:
    print("成功: 所有不等式在所有测试值下均满足")
else:
    print("验证失败: 存在不满足条件的测试值")


output1_squared = (3 * C)**2

output2_squared = C**2

output3_squared = (-C)**2

output4_squared = (-3 * C)**2



# 计算 E[X^2]

expectation_X2 = p1 * output1_squared + p2 * output2_squared + p3 * output3_squared + p4 * output4_squared



# 计算方差 Var(X) = E[X^2] - (E[X])^2

variance = expectation_X2 - x**2

variance_simplified = variance.simplify()

print(variance_simplified)