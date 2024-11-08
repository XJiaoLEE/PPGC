from sympy import symbols, exp, simplify

# 定义符号变量
epsilon, x = symbols('epsilon x')

# 定义 a 的表达式
a = (exp(epsilon) - 1) / (2 * (exp(epsilon) + 1))

# 定义 p 和 q 的表达式
p = a * x + 1/2
q = -a * x + 1/2

# 不等式条件 p / q <= e^epsilon
inequality = (p / q) <= exp(epsilon)

# 选择一些 epsilon 值来验证不等式
epsilon_val = [1, 2, 4,100]

# 检查不等式在 x ∈ [-1, 1) 范围内的结果
for eps in epsilon_val:
    print(f"验证 epsilon = {eps}")
    for x_val in [-1, -0.9, -0.5, 0, 0.5, 0.9,1]:
        result = inequality.subs({x: x_val, epsilon: eps})
        is_satisfied = simplify(result)
        print(f"x = {x_val}: 不等式成立？ {is_satisfied}")
    print("-" * 30)
