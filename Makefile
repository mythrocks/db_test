include flags.make

cudf:
	cd $(CUDF_DIR)/build && $(MAKE) -f $(CUDF_DIR)/build/Makefile nvstrings
	cd $(CUDF_DIR)/build && $(MAKE) -f $(CUDF_DIR)/build/Makefile cudf
	cd $(CUDF_DIR)/build && $(MAKE) -f $(CUDF_DIR)/build/Makefile cudftestutil

db_test.o: db_test.cu
	nvcc $(CUDA_DEFINES) $(CUDF_INCLUDES) $(CUDA_FLAGS) -c db_test.cu -g

db_test_utils.o: db_test_utils.cu
	nvcc $(CUDA_DEFINES) $(CUDF_INCLUDES) $(CUDA_FLAGS) -c db_test_utils.cu -g

db_test_archive.o: db_test_archive.cu
	nvcc $(CUDA_DEFINES) $(CUDF_INCLUDES) $(CUDA_FLAGS) -c db_test_archive.cu -g

db_test:	cudf db_test.o db_test_utils.o db_test_archive.o
	echo $(CUDF_DIR)
	/usr/bin/c++   -Werror -O3 -DNDEBUG  -Wl,--disable-new-dtags db_test.o db_test_utils.o db_test_archive.o -o DB_TEST  -L/usr/local/cuda/targets/x86_64-linux/lib/stubs  -L/usr/local/cuda/targets/x86_64-linux/lib  -L$(CUDF_DIR)/build/lib  -L$(CUDF_DIR)/build  -L$(CUDF_DIR)/build/arrow/build/flatbuffers_ep-prefix/src/flatbuffers_ep-install/lib  -L$(CUDF_DIR)/build/googletest/install/lib  -L/home/dbaranec/miniconda3/envs/cudf_dev/lib/librmm.so -Wl,-rpath,/usr/local/cuda/targets/x86_64-linux/lib/stubs:$(CUDF_DIR)/build:/home/dbaranec/miniconda3/envs/cudf_dev/lib:/usr/local/cuda/targets/x86_64-linux/lib:$(CUDF_DIR)/build/lib:$(CUDF_DIR)/build/arrow/build/flatbuffers_ep-prefix/src/flatbuffers_ep-install/lib:$(CUDF_DIR)/build/googletest/install/lib:/home/dbaranec/miniconda3/envs/cudf_dev/lib/librmm.so -lpthread $(CUDF_DIR)/build/tests/libcudftestutil.a $(CUDF_DIR)/build/libcudf.so /home/dbaranec/miniconda3/envs/cudf_dev/lib/libnvToolsExt.so -lNVCategory -lNVStrings -lnvToolsExt /home/dbaranec/miniconda3/envs/cudf_dev/lib/librmm.so -Wl,--whole-archive $(CUDF_DIR)/build/arrow/install/lib/libarrow_cuda.a -Wl,--no-whole-archive $(CUDF_DIR)/build/arrow/install/lib/libarrow.a $(CUDF_DIR)/build/googletest/install/lib/libgtest.a -lnvrtc -lcudart -lcuda /home/dbaranec/miniconda3/envs/cudf_dev/lib/libz.so /home/dbaranec/miniconda3/envs/cudf_dev/lib/libboost_filesystem.so -lcudadevrt -lcudart_static -lrt -lpthread -ldl

rmm:
	cd $(RMM_DIR)/build && $(MAKE) -f $(RMM_DIR)/build/Makefile rmm

db_test_rmm.o: db_test_rmm.cu
	nvcc $(CUDA_DEFINES) $(CUDF_INCLUDES) $(CUDA_FLAGS) -c db_test_rmm.cu -g

db_test_rmm: db_test_rmm.o rmm
	/usr/bin/c++   -Werror -O3 -DNDEBUG  -Wl,--disable-new-dtags db_test_rmm.o -o DB_TEST_RMM -L/usr/local/cuda/targets/x86_64-linux/lib/stubs  -L/usr/local/cuda/targets/x86_64-linux/lib -L$(RMM_DIR)/build -Wl,-rpath,/usr/local/cuda/targets/x86_64-linux/lib/stubs:.:../cuda-linux64-mixed-rel-nightly/lib64:$(RMM_DIR)/build:/usr/local/cuda/targets/x86_64-linux/lib -lrmm -lpthread -Wl,--whole-archive -Wl,--no-whole-archive -lnvrtc -lcudart -lcuda -lcudadevrt -lcudart_static -lrt -lpthread -ldl
