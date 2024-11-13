import re
import matplotlib.pyplot as plt
from datetime import datetime
import os

# 定义日志文件路径列表
log_dir = "D:/论文/PPGC/Codes/train/logs"
log_files = ["QSGD_outbits1.log", "baseline_outbits2.log"]  # 在此添加多个日志文件名
log_file_paths = [os.path.join(log_dir, log_file) for log_file in log_files]

# 定义正则表达式以提取开始时间和每个 epoch 的准确性信息
start_time_pattern = r"\[(.*?)\] Start training for the 1th step"
epoch_accuracy_pattern = r"\[(.*?)\] Epoch (\d+)/(\d+), Accuracy: ([0-9.]+)"

# 初始化颜色列表
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
plt.figure(figsize=(10, 6))

# 遍历每个日志文件
for idx, log_file_path in enumerate(log_file_paths):
    # 读取日志文件
    with open(log_file_path, 'r') as file:
        log_lines = file.readlines()

    # 提取开始时间和每个 epoch 的信息
    start_time = None
    epochs = []

    for line in log_lines:
        # 匹配开始时间
        if start_time is None:
            match_start = re.search(start_time_pattern, line)
            if match_start:
                start_time = datetime.strptime(match_start.group(1), "%Y-%m-%d %H:%M:%S")

        # 匹配每个 epoch 的准确性
        match_epoch = re.search(epoch_accuracy_pattern, line)
        if match_epoch:
            epoch_time = datetime.strptime(match_epoch.group(1), "%Y-%m-%d %H:%M:%S")
            epoch_num = int(match_epoch.group(2))
            total_epochs = int(match_epoch.group(3))
            accuracy = float(match_epoch.group(4))
            elapsed_time = (epoch_time - start_time).total_seconds() / 60  # 转换为分钟
            epochs.append((epoch_num, total_epochs, elapsed_time, accuracy))

    # 提取数据以便绘图
    epoch_numbers = [epoch[0] for epoch in epochs]
    elapsed_times = [epoch[2] for epoch in epochs]
    accuracies = [epoch[3] for epoch in epochs]

    # 绘制图形
    color = colors[idx % len(colors)]
    plt.plot(elapsed_times, accuracies, marker='o', linestyle='-', color=color, label=f"{log_files[idx]}")
    for i, epoch_num in enumerate(epoch_numbers):
        plt.text(elapsed_times[i], accuracies[i], f"Epoch {epoch_num}", ha='right', color=color)

# 添加图例、标签和标题
plt.xlabel("Time (minutes from start)")
plt.ylabel("Accuracy")
plt.title("Training Accuracy over Time for Multiple Logs")
plt.legend()
plt.grid(True)

# 保存和显示图片
png_filename = os.path.join(log_dir, "combined_accuracy_time_plot.png")
plt.savefig(png_filename)
plt.show()
print(f"Plot saved as {png_filename}")
