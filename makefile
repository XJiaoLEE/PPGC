# Makefile to run distributed training script across multiple machines

run_MNIST_BASELINE_master:
	@echo "Starting master node for distributed training with $(WORLD_SIZE) nodes..."
	python3 train/FL_MNIST.py --world_size=$(world_size) --rank=$(rank) --dist_url=tcp://192.168.1.248:20008 --mechanism=baseline --out_bits=1


run_MNIST_QSGD_master:
	@echo "Starting master node for distributed training with $(WORLD_SIZE) nodes..."
	python3 train/FL_MNIST.py --world_size=$(world_size) --rank=$(rank) --dist_url=tcp://192.168.1.248:20008 --mechanism=QSGD --out_bits=1


run_MNIST_PPGC_master:
	@echo "Starting master node for distributed training with $(WORLD_SIZE) nodes..."
	python3 train/FL_MNIST.py --world_size=$(world_size) --rank=$(rank) --dist_url=tcp://192.168.1.248:20008 --mechanism=PPGC --out_bits=1

run_MNIST_ONEBIT_master:
	@echo "Starting master node for distributed training with $(WORLD_SIZE) nodes..."
	python3 train/FL_MNIST.py --world_size=$(world_size) --rank=$(rank) --dist_url=tcp://192.168.1.248:20008 --mechanism=ONEBIT --out_bits=1

run_MNIST_RAPPOR_master:
	@echo "Starting master node for distributed training with $(WORLD_SIZE) nodes..."
	python3 train/FL_MNIST.py --world_size=$(world_size) --rank=$(rank) --dist_url=tcp://192.168.1.248:20008 --mechanism=RAPPOR --out_bits=1


run_MNIST_BASELINE_master_large:
	@echo "Starting master node for distributed training with $(WORLD_SIZE) nodes..."
	python3 train/FL_MNIST_large.py --world_size=$(world_size) --rank=$(rank) --dist_url=tcp://192.168.1.248:20008 --mechanism=baseline --out_bits=1


run_MNIST_QSGD_master_large:
	@echo "Starting master node for distributed training with $(WORLD_SIZE) nodes..."
	python3 train/FL_MNIST_large.py --world_size=$(world_size) --rank=$(rank) --dist_url=tcp://192.168.1.248:20008 --mechanism=QSGD --out_bits=1


run_MNIST_PPGC_master_large:
	@echo "Starting master node for distributed training with $(WORLD_SIZE) nodes..."
	python3 train/FL_MNIST_large.py --world_size=$(world_size) --rank=$(rank) --dist_url=tcp://192.168.1.248:20008 --mechanism=PPGC --out_bits=1

run_MNIST_ONEBIT_master_large:
	@echo "Starting master node for distributed training with $(WORLD_SIZE) nodes..."
	python3 train/FL_MNIST_large.py --world_size=$(world_size) --rank=$(rank) --dist_url=tcp://192.168.1.248:20008 --mechanism=ONEBIT --out_bits=1

run_MNIST_RAPPOR_master_large:
	@echo "Starting master node for distributed training with $(WORLD_SIZE) nodes..."
	python3 train/FL_MNIST_large.py --world_size=$(world_size) --rank=$(rank) --dist_url=tcp://192.168.1.248:20008 --mechanism=RAPPOR --out_bits=1


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
