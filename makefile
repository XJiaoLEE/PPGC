# Makefile to run distributed training script across multiple machines

run_MNIST_baseline_master:
	@echo "Starting master node for distributed training with $(WORLD_SIZE) nodes..."
	python3 train\FL_MNIST.py --world_size=$(world_size) --rank=$(rank) --dist_url=tcp://10.120.35.3:20008 --mechanism=baseline --out_bits=1
#python FL_MNIST.py --world_size=1 --rank=0 --dist_url=tcp://10.7.161.177:23456 --mechanism=baseline --out_bits=1



# PYTHON = python
# SCRIPT = FL_MNIST.py
# WORLD_SIZE = 1
# DIST_URL = tcp://10.7.161.177:23456
# DIST_BACKEND = nccl
# MECHANISM = baseline
# OUT_BITS = 2

# run_MNIST_baseline_master:
# 	@echo "Starting master node for distributed training with $(WORLD_SIZE) nodes..."
# 	$(PYTHON) $(SCRIPT) --world_size=$(WORLD_SIZE) --rank=0 --dist_url=$(DIST_URL) --dist_backend=$(DIST_BACKEND) --mechanism=$(MECHANISM) --out_bits=$(OUT_BITS)

# run_worker:
# 	@echo "Starting worker node for distributed training..."
# 	$(PYTHON) $(SCRIPT) --world_size=$(WORLD_SIZE) --rank=$(RANK) --dist_url=$(DIST_URL) --dist_backend=$(DIST_BACKEND) --mechanism=$(MECHANISM) --out_bits=$(OUT_BITS)

# .PHONY: run_master run_worker
