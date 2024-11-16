import paramiko
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import select
import os
import time

# 节点信息，包括远程主机的IP地址、用户名、密码和用户的XXXXX/PPGC目录
nodes = {
    "192.168.1.154": {"user": "d", "password": "d", "remote_directory": "/home/d/XXXXX/PPGC"},
    "192.168.1.169": {"user": "dd", "password": "dd", "remote_directory": "/home/dd/XXXXX/PPGC"},
    "192.168.1.157": {"user": "ddd", "password": "ddd", "remote_directory": "/home/ddd/XXXXX/PPGC"},
    "192.168.1.108": {"user": "dddd", "password": "dddd", "remote_directory": "/home/dddd/XXXXX/PPGC"},
    "192.168.1.107": {"user": "ddddd", "password": "ddddd", "remote_directory": "/home/ddddd/XXXXX/PPGC"},
    "192.168.1.232": {"user": "dddddd", "password": "dddddd", "remote_directory": "/home/dddddd/XXXXX/PPGC"},
    "192.168.1.199": {"user": "ddddddd", "password": "ddddddd", "remote_directory": "/home/ddddddd/XXXXX/PPGC"},
    "192.168.1.248": {"user": "dddddddd", "password": "dddddddd", "remote_directory": "/home/dddddddd/XXXXX/PPGC"}
}

# 当前主机的 rank 和 world_size
current_host_rank = 1  # 当前主机的 rank
world_size = len(nodes)

# 动态命令模板，用户可以在运行时修改它
command_template = "cd {remote_directory} && git pull origin main && make run_MNIST_PPGC_master world_size={world_size} rank={rank}"

# 检查并杀死占用指定端口的进程
def kill_process_on_port(port):
    try:
        result = subprocess.run(["lsof", "-t", f"-i:{port}"], capture_output=True, text=True)
        if result.stdout:
            pid = result.stdout.strip()
            print(f"端口 {port} 被占用，正在杀死进程 {pid}...")
            os.kill(int(pid), 9)
            print(f"进程 {pid} 已被杀死。")
        else:
            print(f"端口 {port} 未被占用。")
    except Exception as e:
        print(f"检查或杀死进程时出错: {e}")

# 执行 SSH 命令的函数，不设置超时时间并实时读取输出，包括 tqdm 进度条
def ssh_execute_command(hostname, username, password, command):
    try:
        # 创建SSH客户端
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # 连接远程主机
        print(f"正在连接到 {hostname}...")
        client.connect(hostname=hostname, username=username, password=password)
        
        # 执行命令
        print(f"在 {hostname} 上执行命令: {command}")
        stdin, stdout, stderr = client.exec_command(command)

        # 捕获并实时显示命令的输出（包括 tqdm 进度条）
        while True:
            # 使用 select 来检测 stdout 是否有内容可读取
            if stdout.channel.recv_ready():
                rl, wl, xl = select.select([stdout.channel], [], [], 0.0)
                if len(rl) > 0:
                    # 使用 recv 而不是 readline 来捕获 tqdm 的进度条
                    output = stdout.channel.recv(1024).decode('utf-8')
                    print(f"[{hostname}] {output}", end='')

            if stderr.channel.recv_stderr_ready():
                rl, wl, xl = select.select([stderr.channel], [], [], 0.0)
                if len(rl) > 0:
                    error_output = stderr.channel.recv_stderr(1024).decode('utf-8')
                    print(f"[{hostname} ERROR] {error_output}", end='')

            # 如果命令执行完成，跳出循环
            if stdout.channel.exit_status_ready():
                break

        # 确保读取所有的剩余数据
        stdout.channel.recv_exit_status()
        
    except Exception as e:
        print(f"在主机 {hostname} 执行命令时出错: {e}")
    finally:
        # 关闭SSH连接
        client.close()
        
# 并行执行远程和本地命令
def run_commands_in_parallel():
    # 检查并杀死占用 20008 端口的进程
    kill_process_on_port(20008)
    
    with ThreadPoolExecutor() as executor:
        futures = []
        # rank=0的节点优先运行
        rank = 0
        rank0_hostname = "192.168.1.248"
        rank0_credentials = nodes[rank0_hostname]
        username = rank0_credentials['user']
        password = rank0_credentials['password']
        remote_directory = rank0_credentials['remote_directory']
        command = command_template.format(remote_directory=remote_directory, world_size=world_size, rank=rank)
        futures.append(executor.submit(ssh_execute_command, rank0_hostname, username, password, command))

        # 确保 rank=0 的节点先启动一小段时间
        time.sleep(5)

        # 为其他远程主机添加任务
        rank = 1
        for hostname, credentials in nodes.items():
            if hostname == rank0_hostname:
                continue
            username = credentials['user']
            password = credentials['password']
            remote_directory = credentials['remote_directory']
            
            # 根据模板生成远程主机要执行的命令
            command = command_template.format(remote_directory=remote_directory, world_size=world_size, rank=rank)
            
            # 提交远程任务
            futures.append(executor.submit(ssh_execute_command, hostname, username, password, command))
            
            rank += 1  # 每次循环后增加 rank 值

        # 等待所有任务完成
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"任务执行失败: {e}")

# 并行执行本地和远程命令
run_commands_in_parallel()

print("分布式计算命令在所有主机上执行完成。")