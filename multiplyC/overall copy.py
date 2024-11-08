from sympy import symbols, Eq, solve, exp, Rational, simplify
# 1位2输出时，用C，-C
# 2位4输出时，用C,C*1/3,-C*1/3,-C
# 3位8输出时，用C,C*5/7,C*3/7,C*1/7,-C*1/7,-C*3/7,-C*5/7,-C
# 定义符号变量
x, epsilon = symbols('x epsilon')
C = (exp(epsilon) + 1) / (exp(epsilon) - 1)
qutibit = 4  # 可以更改 qutibit 值
outbit = 2 ** qutibit  # 计算输出总数

# 定义空列表存储概率和输出
probabilities = []
outputs = []
pa_i = []

# 生成前半部分的概率和输出
for i in range(1, outbit // 2 + 1):
    # 输出值公式
    output_value = (outbit - (2 * i - 1)) * C
    print("output_value",output_value)
    outputs.append(output_value)
    
    # 概率值公式
    abottom_i = 2 * (outbit // 2) * (outbit - (i *2) + 1)
    a_i = (exp(epsilon) - 1) / (abottom_i * (exp(epsilon) + 1))
    pa_i.append(a_i)
    p_i = x * a_i + Rational(1, outbit)
    print("abottom_i,a_i",abottom_i,a_i)
    probabilities.append(p_i)

# 添加后半部分：镜像前半部分
for i in range(outbit // 2):
    outputs.append(-outputs[outbit // 2-i-1])
    a_i = -pa_i[outbit // 2-i-1]
    p_i = x * a_i + Rational(1, outbit)
    probabilities.append(p_i)
    #probabilities.append(probabilities[i])

# 输出结果
print("输出值（outputs）:")
for idx, output in enumerate(outputs):
    print(f"输出值 {idx + 1}: {output}")

print("\n概率值公式（probabilities）:")
for idx, probability in enumerate(probabilities):
    print(f"p_{idx + 1}: {probability}")



# 计算期望值 E[X]
expectation = sum(prob * output for prob, output in zip(probabilities, outputs))

# 计算 E[X^2] 用于方差计算
expectation_X2 = sum(prob * (output**2) for prob, output in zip(probabilities, outputs))

# 计算方差 Var(X) = E[X^2] - (E[X])^2
variance = expectation_X2 - expectation**2

# 化简并打印结果
expectation_simplified = simplify(expectation)
variance_simplified = simplify(variance)

print("\n期望值 E[X]:", expectation_simplified)
print("\n方差 Var(X):", variance_simplified)