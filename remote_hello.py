import paramiko
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed

# 节点信息，包括远程主机的IP地址、用户名、密码和用户的NetSenseML目录
nodes = {
    "192.168.1.154": {"user": "d", "password": "d"},
    "192.168.1.169": {"user": "dd", "password": "dd"},
    "192.168.1.157": {"user": "ddd", "password": "ddd"},
    "192.168.1.108": {"user": "dddd", "password": "dddd"},
    "192.168.1.107": {"user": "ddddd", "password": "ddddd"},
    "192.168.1.232": {"user": "dddddd", "password": "dddddd"},
    "192.168.1.199": {"user": "ddddddd", "password": "ddddddd"},
    "192.168.1.248": {"user": "dddddddd", "password": "dddddddd"}
}

# 执行 SSH 命令的函数，用于输出 "Hello"
def ssh_execute_hello(hostname, username, password):
    try:
        # 创建SSH客户端
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # 连接远程主机
        print(f"正在连接到 {hostname}...")
        client.connect(hostname=hostname, username=username, password=password)
        
        # 执行命令
        command = f'echo "Hello from {hostname}"'
        stdin, stdout, stderr = client.exec_command(command)

        # 捕获并打印命令的输出
        output = stdout.read().decode('utf-8').strip()
        print(f"[{hostname}] {output}")
        
    except Exception as e:
        print(f"在主机 {hostname} 执行命令时出错: {e}")
    finally:
        # 关闭SSH连接
        client.close()
        
# 并行执行远程命令
def run_hello_in_parallel():
    with ThreadPoolExecutor() as executor:
        futures = []
        # 为每个远程主机添加任务
        for hostname, credentials in nodes.items():
            username = credentials['user']
            password = credentials['password']
            
            # 提交远程任务
            futures.append(executor.submit(ssh_execute_hello, hostname, username, password))

        # 等待所有任务完成
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"任务执行失败: {e}")

# 运行测试
if __name__ == "__main__":
    run_hello_in_parallel()
    print("所有主机都已输出 Hello。")
