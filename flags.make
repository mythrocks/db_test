# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

CUDF_DIR := /home/dbaranec/projects/cudf/cpp

# compile CUDA with /usr/local/cuda/bin/nvcc
CUDA_FLAGS =  -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 --expt-extended-lambda --expt-relaxed-constexpr -Werror cross-execution-space-call -Xcompiler -Wall,-Werror -Xcompiler=-fPIE   -DJITIFY_USE_CACHE -DCUDF_VERSION=0.11.0 -std=c++14

CUDA_DEFINES = -DARROW_METADATA_V4

CUDF_INCLUDES = -I$(CUDF_DIR)/build/googletest/install/include -I$(CUDF_DIR)/build/include -I$(CUDF_DIR)/include -I$(CUDF_DIR) -I$(CUDF_DIR)/src -I$(CUDF_DIR)/thirdparty/cub -I$(CUDF_DIR)/thirdparty/jitify -I$(CUDF_DIR)/thirdparty/dlpack/include -I$(CUDF_DIR)/thirdparty/libcudacxx/include -I$(CUDF_DIR)/build/arrow/install/include -I$(CUDF_DIR)/build/arrow/build/flatbuffers_ep-prefix/src/flatbuffers_ep-install/include -I/home/dbaranec/miniconda3/envs/cudf_dev/include
