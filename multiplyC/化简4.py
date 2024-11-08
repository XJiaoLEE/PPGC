from sympy import symbols, Eq, solve, exp, Rational, simplify

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
def process_inequality(inequality):
    lhs, rhs = inequality.lhs, inequality.rhs
    simplified = simplify(lhs - rhs)
    if simplified > 0:
        direction = ">"
    elif simplified < 0:
        direction = "<"
    else:
        direction = "="
    return f"{lhs} {direction}= {rhs}"

# 隐私不等式 1： p1 <= p2 * e^epsilon
privacy_inequality_1 = (p1 <= p2 * exp(epsilon)).subs(a1, a1_a2_relation)
privacy_inequality_1 = simplify(privacy_inequality_1)  # 化简
privacy_inequality_1_result = process_inequality(privacy_inequality_1)

# 隐私不等式 2： p2 <= p3 * e^epsilon
privacy_inequality_2 = (p2 <= p3 * exp(epsilon)).subs(a1, a1_a2_relation)
privacy_inequality_2 = simplify(privacy_inequality_2)  # 化简
privacy_inequality_2_result = process_inequality(privacy_inequality_2)

# 隐私不等式 3： p3 <= p4 * e^epsilon
privacy_inequality_3 = (p3 <= p4 * exp(epsilon)).subs(a1, a1_a2_relation)
privacy_inequality_3 = simplify(privacy_inequality_3)  # 化简
privacy_inequality_3_result = process_inequality(privacy_inequality_3)

# 输出每个不等式的结果，带有不等号方向
print("a2 与 x 和 epsilon 的关系（从隐私不等式 1 得到）:", privacy_inequality_1_result)
print("a2 与 x 和 epsilon 的关系（从隐私不等式 2 得到）:", privacy_inequality_2_result)
print("a2 与 x 和 epsilon 的关系（从隐私不等式 3 得到）:", privacy_inequality_3_result)
