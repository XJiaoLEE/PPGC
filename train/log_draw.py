import re
import matplotlib.pyplot as plt
from datetime import datetime
import os

# 定义正则表达式以提取开始时间和每个 epoch 的准确性信息
start_time_pattern = r"\[(.*?)\] Start training for the 1th step"
epoch_accuracy_pattern = r"\[(.*?)\] Epoch (\d+)/(\d+), Accuracy: ([0-9.]+)"

# 读取日志文件
# 定义日志文件路径
log_dir = "D:/论文/PPGC/Codes/train/logs"
log_file_name = "QSGD_outbits1.log"
log_file_path = os.path.join(log_dir, log_file_name)
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
plt.figure(figsize=(10, 6))
plt.plot(elapsed_times, accuracies, marker='o', linestyle='-')
for i, epoch_num in enumerate(epoch_numbers):
    plt.text(elapsed_times[i], accuracies[i], f"Epoch {epoch_num}", ha='right')

plt.xlabel("Time (minutes from start)")
plt.ylabel("Accuracy")
plt.title("Training Accuracy over Time")
plt.grid(True)

# 保存图片
png_filename = os.path.splitext(log_file_path)[0] + ".png"
plt.savefig(png_filename)
print(f"Plot saved as {png_filename}")
