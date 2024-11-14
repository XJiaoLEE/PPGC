import os
import torch.distributed as dist

# 设置环境变量
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '12345'

try:
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        rank=0,
        world_size=1
    )
    print("Distributed initialized successfully!")
except Exception as e:
    print(f"Error during initialization: {e}")
