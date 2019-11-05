include flags.make
# include link.txt

CUDF_DIR = /home/dbaranec/projects/cudf/cpp

# 3 steps.
# - build cudf
# - compile source to .o
# - link
# the compile and link are seperate because nvcc doesn't support -rpath. -rpath lets us hardcode paths to the .sos we want to load
# at startup. 
all:	
	cd ~/projects/cudf/cpp/build && $(MAKE) -f ~/projects/cudf/cpp/build/Makefile cudf
	nvcc $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -c db_test.cu -o db_test.o -g
	/usr/bin/c++   -Werror -O3 -DNDEBUG  -Wl,--disable-new-dtags db_test.o  -o DB_TEST  -L/usr/local/cuda/targets/x86_64-linux/lib/stubs  -L/usr/local/cuda/targets/x86_64-linux/lib  -L$(CUDF_DIR)/build/lib  -L$(CUDF_DIR)/build  -L$(CUDF_DIR)/build/arrow/build/flatbuffers_ep-prefix/src/flatbuffers_ep-install/lib  -L$(CUDF_DIR)/build/googletest/install/lib  -L/home/dbaranec/miniconda3/envs/cudf_dev/lib/librmm.so -Wl,-rpath,/usr/local/cuda/targets/x86_64-linux/lib/stubs:$(CUDF_DIR)/build:/home/dbaranec/miniconda3/envs/cudf_dev/lib:/usr/local/cuda/targets/x86_64-linux/lib:$(CUDF_DIR)/build/lib:$(CUDF_DIR)/build/arrow/build/flatbuffers_ep-prefix/src/flatbuffers_ep-install/lib:$(CUDF_DIR)/build/googletest/install/lib:/home/dbaranec/miniconda3/envs/cudf_dev/lib/librmm.so -lgmock -lgtest -lgmock_main -lgtest_main -lpthread $(CUDF_DIR)/build/libcudf.so $(CUDF_DIR)/build/tests/libcudftestutil.a /home/dbaranec/miniconda3/envs/cudf_dev/lib/libnvToolsExt.so -lNVCategory -lNVStrings -lnvToolsExt /home/dbaranec/miniconda3/envs/cudf_dev/lib/librmm.so -Wl,--whole-archive $(CUDF_DIR)/build/arrow/install/lib/libarrow_cuda.a -Wl,--no-whole-archive $(CUDF_DIR)/build/arrow/install/lib/libarrow.a -lnvrtc -lcudart -lcuda /home/dbaranec/miniconda3/envs/cudf_dev/lib/libz.so /home/dbaranec/miniconda3/envs/cudf_dev/lib/libboost_filesystem.so -lcudadevrt -lcudart_static -lrt -lpthread -ldl 
