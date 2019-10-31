# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# compile CUDA with /usr/local/cuda/bin/nvcc
CUDA_FLAGS =  -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 --expt-extended-lambda --expt-relaxed-constexpr -Werror cross-execution-space-call -Xcompiler -Wall,-Werror -Xcompiler=-fPIE   -DJITIFY_USE_CACHE -DCUDF_VERSION=0.11.0 -std=c++14

CUDA_DEFINES = -DARROW_METADATA_V4

CUDA_INCLUDES = -I/home/dbaranec/projects/cudf/cpp/build/googletest/install/include -I/home/dbaranec/projects/cudf/cpp/build/include -I/home/dbaranec/projects/cudf/cpp/include -I/home/dbaranec/projects/cudf/cpp -I/home/dbaranec/projects/cudf/cpp/src -I/home/dbaranec/projects/cudf/cpp/thirdparty/cub -I/home/dbaranec/projects/cudf/cpp/thirdparty/jitify -I/home/dbaranec/projects/cudf/cpp/thirdparty/dlpack/include -I/home/dbaranec/projects/cudf/cpp/thirdparty/libcudacxx/include -I/home/dbaranec/projects/cudf/cpp/build/arrow/install/include -I/home/dbaranec/projects/cudf/cpp/build/arrow/build/flatbuffers_ep-prefix/src/flatbuffers_ep-install/include -I/home/dbaranec/miniconda3/envs/cudf_dev/include 

