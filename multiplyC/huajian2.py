from sympy import symbols, Eq, solve, exp, simplify

# 定义符号变量
a, C, x, epsilon = symbols('a C x epsilon')

# 定义 p 和 q 的表达式
p = a * x + 1/2
q = -a * x + 1/2

# 隐私不等式 p/q <= e^epsilon
inequality = Eq(p / q, exp(epsilon))

# 使用 solve 求解 a 的通用取值范围
a_epsilon_relation = solve(inequality, a)

print("a 的取值范围:", a_epsilon_relation)

# 条件 3: 输出条件 pC + q(-C) = x
# 使用求得的 a 的范围求解 C 的通用取值范围
output_condition = Eq(p * C + q * (-C), x)

# 将 a 的通用解代入输出条件，求解 C 的范围
C_solution = solve(output_condition.subs(a, a_epsilon_relation[0]), C)

print("C 的取值范围:", C_solution)
