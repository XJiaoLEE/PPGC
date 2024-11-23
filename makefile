# Makefile to run distributed training script across multiple machines

# 定义 epsilon 变量，如果未定义则使用默认值 0
EPSILON ?= 0

# 生成 epsilon 参数的字符串，如果 EPSILON 为 0，则为空
EPSILON_ARG := $(if $(filter 0,$(EPSILON)),, --epsilon=$(EPSILON))

# 定义 sparsification 变量，如果未定义则使用默认值 0
sparsification ?= 0

# 生成 sparsification 参数的字符串，如果 sparsification 为 0，则为空
sparsification_ARG := $(if $(filter 0,$(sparsification)),, --sparsification=$(sparsification))

# 定义 dataset 变量，如果未定义则使用默认值 'MNIST'
dataset ?= MNIST

# 生成 dataset 参数的字符串，如果 dataset 为 'MNIST'，则为空，否则为传入的值
dataset_ARG := $(if $(filter MNIST,$(dataset)),, --dataset=$(dataset))


run_MNIST_BASELINE_master:
	@echo "Starting master node for distributed training with $(WORLD_SIZE) nodes..."
	python3 train/FL_MNIST.py --world_size=$(world_size) --rank=$(rank) --dist_url=tcp://192.168.1.248:20008 --mechanism=baseline --out_bits=1 $(EPSILON_ARG)

run_MNIST_QSGD_master:
	@echo "Starting master node for distributed training with $(WORLD_SIZE) nodes..."
	python3 train/FL_MNIST.py --world_size=$(world_size) --rank=$(rank) --dist_url=tcp://192.168.1.248:20008 --mechanism=QSGD --out_bits=1 $(EPSILON_ARG)

run_MNIST_PPGC_master:
	@echo "Starting master node for distributed training with $(WORLD_SIZE) nodes..."
	python3 train/FL_MNIST.py --world_size=$(world_size) --rank=$(rank) --dist_url=tcp://192.168.1.248:20008 --mechanism=PPGC --out_bits=1 $(EPSILON_ARG)

run_MNIST_ONEBIT_master:
	@echo "Starting master node for distributed training with $(WORLD_SIZE) nodes..."
	python3 train/FL_MNIST.py --world_size=$(world_size) --rank=$(rank) --dist_url=tcp://192.168.1.248:20008 --mechanism=ONEBIT --out_bits=1 $(EPSILON_ARG)

run_MNIST_RAPPOR_master:
	@echo "Starting master node for distributed training with $(WORLD_SIZE) nodes..."
	python3 train/FL_MNIST.py --world_size=$(world_size) --rank=$(rank) --dist_url=tcp://192.168.1.248:20008 --mechanism=RAPPOR --out_bits=1 $(EPSILON_ARG)

run_MNIST:
	@echo "Starting master node for distributed training with $(WORLD_SIZE) nodes..."
	python3 train/FL_MNIST_large.py --world_size=$(world_size) --rank=$(rank) --dist_url=tcp://192.168.1.248:20008 --mechanism=$(mechanism) --out_bits=1 $(EPSILON_ARG) $(sparsification_ARG)

run_large:
	@echo "Starting master node for distributed training with $(WORLD_SIZE) nodes..."
	python3 train/FL_large_train.py --world_size=$(world_size) --rank=$(rank) --dist_url=tcp://192.168.1.248:20008 --mechanism=$(mechanism) --out_bits=1 $(EPSILON_ARG) $(sparsification_ARG) $(dataset_ARG)

# run_MNIST_BASELINE_master_large:
# 	@echo "Starting master node for distributed training with $(WORLD_SIZE) nodes..."
# 	python3 train/FL_MNIST_large.py --world_size=$(world_size) --rank=$(rank) --dist_url=tcp://192.168.1.248:20008 --mechanism=baseline --out_bits=1 $(EPSILON_ARG)

# run_MNIST_QSGD_master_large:
# 	@echo "Starting master node for distributed training with $(WORLD_SIZE) nodes..."
# 	python3 train/FL_MNIST_large.py --world_size=$(world_size) --rank=$(rank) --dist_url=tcp://192.168.1.248:20008 --mechanism=QSGD --out_bits=1 $(EPSILON_ARG)

# run_MNIST_PPGC_master_large:
# 	@echo "Starting master node for distributed training with $(WORLD_SIZE) nodes..."
# 	python3 train/FL_MNIST_large.py --world_size=$(world_size) --rank=$(rank) --dist_url=tcp://192.168.1.248:20008 --mechanism=PPGC --out_bits=1 $(EPSILON_ARG)

# run_MNIST_ONEBIT_master_large:
# 	@echo "Starting master node for distributed training with $(WORLD_SIZE) nodes..."
# 	python3 train/FL_MNIST_large.py --world_size=$(world_size) --rank=$(rank) --dist_url=tcp://192.168.1.248:20008 --mechanism=ONEBIT --out_bits=1 $(EPSILON_ARG)

# run_MNIST_RAPPOR_master_large:
# 	@echo "Starting master node for distributed training with $(WORLD_SIZE) nodes..."
# 	python3 train/FL_MNIST_large.py --world_size=$(world_size) --rank=$(rank) --dist_url=tcp://192.168.1.248:20008 --mechanism=RAPPOR --out_bits=1 $(EPSILON_ARG)

# run_MNIST_TernGrad_master_large:
# 	@echo "Starting master node for distributed training with $(WORLD_SIZE) nodes..."
# 	python3 train/FL_MNIST_large.py --world_size=$(world_size) --rank=$(rank) --dist_url=tcp://192.168.1.248:20008 --mechanism=TernGrad --out_bits=1 $(EPSILON_ARG)


