#include <cuda.h>
#include <cub/cub.cuh>
#include <rmm/rmm.h>
// #include <rmm/thrust_rmm_allocator.h>

__global__ void int_doubler(int *dst, int *src, int num_els)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid < num_els){
    dst[tid] = src[tid] * 2;
  }    
}

void mem_tester()
{
  int num_els = 16;

  // reference
  {
    printf("REFERENCE TEST : ");
    int *dst, *src;
    cudaMalloc(&src, sizeof(int) * num_els);
    int val = 2;
    for(int idx=0; idx<num_els; idx++){      
      cudaMemcpy(src + idx, &val, sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaMalloc(&dst, sizeof(int) * num_els);

    int_doubler<<<num_els, 1>>>(dst, src, num_els);
    int chk[16];
    cudaMemcpy(chk, dst, sizeof(int) * num_els, cudaMemcpyDeviceToHost);
    for(int idx=0; idx<num_els; idx++){
      if(idx < num_els-1){
        printf("%d, ", chk[idx]);
      } else {
        printf("%d\n", chk[idx]);
      }
    }
  }

  // rmm
  {
    printf("RMM TEST : ");
    int *dst, *src;
    RMM_ALLOC(&src, sizeof(int) * num_els, 0);      
    int val = 2;
    for(int idx=0; idx<num_els; idx++){      
      cudaMemcpy(src + idx, &val, sizeof(int), cudaMemcpyHostToDevice);
    }
    RMM_ALLOC(&dst, sizeof(int) * num_els, 0);    

    int_doubler<<<num_els, 1>>>(dst, src, num_els);
    int chk[16];
    cudaMemcpy(chk, dst, sizeof(int) * num_els, cudaMemcpyDeviceToHost);
    for(int idx=0; idx<num_els; idx++){
      if(idx < num_els-1){
        printf("%d, ", chk[idx]);
      } else {
        printf("%d\n", chk[idx]);
      }
    }  
  }  
}

int main()
{
   // init stuff
   cuInit(0);      

   rmmOptions_t options; // {PoolAllocation, 0, false};   
   rmmInitialize(&options);

   mem_tester();

    // shut stuff down
   rmmFinalize();
}