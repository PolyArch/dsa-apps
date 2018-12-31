
NVCC=nvcc
ARCH?=sm_52

MKL_PATH=/home/jian.weng/intel/mkl/

CUDA_PATH?=/usr/local/cuda
CUDA_LIB64=$(CUDA_PATH)/lib64
CUDNN_PATH?=/usr/local/cudnn
NCCL_PATH?=/usr/local/nccl
MPI_PATH?=/usr/local/openmpi
MPI_INCLUDE_PATH?=$(MPI_PATH)/include
BIN_DIR?=bin
MKDIR=mkdir -p
NVCC = $(CUDA_PATH)/bin/nvcc
#BLAS
BLAS_LIBRARY?=cublas
BLAS_PATH?=$(CUDA_LIB64)
#CONV
CONV_LIBRARY?=cudnn
CONV_PATH?=$

.PHONY=all gemm clean

all: gemm

main.exe: main.cu
	$(NVCC) $< -o $@ -I$(CUDA_PATH)/include -L $(BLAS_PATH) -l$(BLAS_LIBRARY) -L $(CUDA_LIB64) -lcurand $(NVCC_ARCH_ARGS) -std=c++11

mkl.exe: mkl.cc
	g++ $< -o $@ -I$(MKL_PATH)/include -L$(MKL_PATH)/lib64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -ldl -lm -std=c++11

clean:
	rm -rf *.exe

