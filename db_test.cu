#include <stdio.h>
#include <time.h>
#include <unistd.h>

#include <cudf/cudf.h>
#include <tests/utilities/legacy/column_wrapper.cuh>
#include <bitmask/legacy/bit_mask.cuh>

using namespace std;

// ------------------
//
// Defines
//
// ------------------

#define UNREFERENCED(_x)    do { (void)(_x); } while(0)

struct scoped_timer {
    timespec m_start;
    char m_name[64];

    scoped_timer(){}
    scoped_timer(const char *_name) 
    {         
        strcpy(m_name, _name);
        clock_gettime(CLOCK_MONOTONIC, &m_start);
    }
    ~scoped_timer()
    {
        timespec end;
        clock_gettime(CLOCK_MONOTONIC, &end);    
        long total = ((1000000000 * end.tv_sec) + end.tv_nsec) - 
                     ((1000000000 * m_start.tv_sec) + m_start.tv_nsec);      
        printf("%s : %.2f us\n", m_name, (float)total / (float)1000000.0f);
    }
};


// ------------------
//
// Internal functions
//
// ------------------

// there's some "do stuff the first time" issues that cause bogus timings.
// this function just flushes all that junk out
static void clear_baffles()
{
    // doing an alloc, a memcpy and a free seems to do the trick
    void *gpu_data;
    rmmError_t err = RMM_ALLOC(&gpu_data, 8 * 1024 * 1024, 0);    
    uint cpu_data[64] = { 0 };
    cudaMemcpy(gpu_data, cpu_data, sizeof(cpu_data), cudaMemcpyHostToDevice);    
    RMM_FREE(gpu_data, 0);

    // can't hurt if there's some weird async stuff happening
    sleep(1);
}

/*
// sort a column directly using thrust::sort
static void sort_column_basic()
{   
    int idx;

    // some source data.
    int num_rows = 16;
    float cpu_data[16] = { 5, 8, 10, 11, 2, 3, 1, 15, 12, 7, 6, 13, 9, 4, 0, 14 };
    printf("Unsorted: ");
    for(idx=0; idx<num_rows; idx++){        
        printf(idx < num_rows ? "%.2f, " : "%.2f", cpu_data[idx]);
    }    
    printf("\n");

    int data_size = num_rows * sizeof(float);  

    // allocate device memory for the floats
    float *gpu_data = nullptr;        
    rmmError_t err = RMM_ALLOC(&gpu_data, data_size, 0);    

    // copy cpu data over        
    cudaError_t mem_err = cudaMemcpy(gpu_data, cpu_data, data_size, cudaMemcpyHostToDevice);    

    // setup the column struct. validity mask is null indicating "everything is valid"
    //gdf_column gpu_column;
    //gdf_column_view(&gpu_column, gpu_data, nullptr, num_rows, GDF_FLOAT32);

    // sort
    thrust::device_ptr<float> dv(gpu_data);
    thrust::sort(dv, dv + num_rows, thrust::less<float>());

    // grab the data back
    cudaMemcpy(cpu_data, gpu_data, data_size, cudaMemcpyDeviceToHost);        

    printf("Sorted: ");
    for(idx=0; idx<num_rows; idx++){        
        printf(idx < num_rows ? "%.2f, " : "%.2f", cpu_data[idx]);
    }    
    printf("\n\n");

    RMM_FREE(gpu_data, 0);    
}
*/

#include <bitmask/legacy/legacy_bitmask.hpp>
#include <bitmask/legacy/bit_mask.cuh>
template <typename Type>
__global__
void normalize_nans_and_zeros (cudf::size_type size,
                               const Type* __restrict__ in_data,
                               const bit_mask::bit_mask_t* __restrict__ in_valid,                               
                               Type* __restrict__ out_data)
{
  int tid = threadIdx.x;
  int blkid = blockIdx.x;
  int blksz = blockDim.x;
  int gridsz = gridDim.x;

  int start = tid + blkid * blksz;
  int step = blksz * gridsz;

  for (int i=start; i<size; i+=step) {      
    /*
    if(!bit_mask::is_valid(in_valid, i)){
      continue;
    } 
    */   
    if(std::isnan(in_data[i])){
      out_data[i] = std::numeric_limits<Type>::quiet_NaN();
    } else if(in_data[i] == -0.0){
      out_data[i] = 99.0;
    } else {
      out_data[i] = in_data[i];
    }
  }
}

void ntest()
{    
    int idx;

    int num_els = 8;
    float cpu_data[8] = { -0.0f, 0.0f / 0.0f, 0.0f, 1.0f, -1.0f, 54.3f, -1.0f / 0.0f, -0.0f  };
    
    printf("Start: ");
    for(idx=0; idx<num_els; idx++){        
        printf(idx < num_els ? "%.2f, " : "%.2f", cpu_data[idx]);
    }    
    printf("\n");

    int data_size = num_els * sizeof(float);  

    // allocate device memory for the floats
    float *gpu_src = nullptr;            
    rmmError_t err = RMM_ALLOC(&gpu_src, data_size, 0);
    float *gpu_dst = nullptr;        
    RMM_ALLOC(&gpu_dst, data_size, 0);

    // copy cpu data over        
    cudaError_t mem_err = cudaMemcpy(gpu_src, cpu_data, data_size, cudaMemcpyHostToDevice); 

    // process
    normalize_nans_and_zeros<<<256, 256>>>(num_els, gpu_src, nullptr, gpu_dst);

    // grab the data back
    cudaMemcpy(cpu_data, gpu_dst, data_size, cudaMemcpyDeviceToHost); 

    printf("End: ");
    for(idx=0; idx<num_els; idx++){        
        printf(idx < num_els ? "%.2f, " : "%.2f", cpu_data[idx]);
    }    
    printf("\n");
}

int main()
{        
    float whee[10] = { -0.0f, 0.0f / 0.0f, 0.0f, 1.0f, -1.0f, 54.3f };
    float x = 0.0f / 0.0f;
    int yay = 10;
    yay++;

    printf("WHEEEEEEEEEEEEEEEEEEEEE\n");

    // init stuff
    cuInit(0);    
    rmmOptions_t rmm{};
    rmm.allocation_mode = CudaDefaultAllocation;
    rmm.initial_pool_size = 16 * 1024 * 1024;
    rmm.enable_logging = false;
    rmmInitialize(&rmm);    

    // there's some "do stuff the first time" issues that cause bogus timings.
    // this function just flushes all that junk out
    clear_baffles();

    ntest();

    // do a sort using just thrust::sort
    /*
    {
        scoped_timer t("column test basic");
        sort_column_basic();
    }
    */

    // shut stuff down
    rmmFinalize();
}