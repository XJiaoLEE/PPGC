import re
import matplotlib.pyplot as plt
from datetime import datetime
import os

# 定义日志文件路径列表
log_dir = "D:/论文/PPGC/Codes/train/FLlogs"
log_files = ["baseline_outbits2.log", "QSGD_outbits1.log", "PPGC_outbits1.log"]  # 在此添加多个日志文件名
log_file_paths = [os.path.join(log_dir, log_file) for log_file in log_files]

# 定义正则表达式以提取每轮结束的准确性信息
round_accuracy_pattern = r"\[(.*?)\] End of round (\d+), global model accuracy: ([0-9.]+)"

# 初始化颜色列表
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
plt.figure(figsize=(10, 6))

# 遍历每个日志文件
for idx, log_file_path in enumerate(log_file_paths):
    # 读取日志文件
    with open(log_file_path, 'r') as file:
        log_lines = file.readlines()

    # 提取每个轮次的准确性数据
    rounds = []

    for line in log_lines:
        # 匹配每轮结束的准确性
        match_round = re.search(round_accuracy_pattern, line)
        if match_round:
            round_num = int(match_round.group(2))
            accuracy = float(match_round.group(3))
            rounds.append((round_num, accuracy))

    # 提取数据以便绘图
    round_numbers = [round[0] for round in rounds]
    accuracies = [round[1] for round in rounds]

    # 绘制图形
    color = colors[idx % len(colors)]
    plt.plot(round_numbers, accuracies, marker='o', linestyle='-', color=color, label=f"{log_files[idx]}")
    # for i, round_num in enumerate(round_numbers):
    #     plt.text(round_numbers[i], accuracies[i], f"Round {round_num}", ha='right', color=color)

# 添加图例、标签和标题
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.title("Global Model Accuracy per Round")
plt.legend()
plt.grid(True)

# 保存和显示图片
png_filename = os.path.join(log_dir, "combined_accuracy_round_plot.png")
plt.savefig(png_filename)
plt.show()
print(f"Plot saved as {png_filename}")
