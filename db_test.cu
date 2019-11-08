#include <stdio.h>
#include <time.h>
#include <unistd.h>

#include <cudf/cudf.h>
#include <tests/utilities/legacy/column_wrapper.cuh>
#include <bitmask/legacy/bit_mask.cuh>
#include <bitmask/legacy/legacy_bitmask.hpp>
#include <bitmask/legacy/bit_mask.cuh>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/copying.hpp>
#include <cudf/replace.hpp>
#include <utilities/cuda_utils.hpp>

#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/column_utilities.hpp>

// ------------------
//
// Defines
//
// ------------------

#define BLOCK_SIZE           (256)

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

#if 0    // skeleton for working in cudf
namespace db_test {

using namespace cudf;
using namespace std;
using namespace rmm;
using namespace rmm::mr;

namespace {    // anonymous
   // functor and lambdas go here
}  // end anonymous namespace

namespace cudf {
namespace detail {
   // detail/internal versions of exposed functions go here
}  // namespace detail

   // externally exposed functions go here
}  // namespace cudf

} 
#endif   // skeleton for working in cudf

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

#if 0 // sort a column using thrust::sort
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
#endif // sort a column using thrust::sort

#if 0 // old normalize_nans_and_zeros kernel method. never got used.
namespace db_test {

using namespace cudf;
using namespace std;
using namespace rmm;
using namespace rmm::mr;

// old normalize_nans_and_zeros kernel method. never got used.
namespace {  // anonymous

/* --------------------------------------------------------------------------*/
/**
 * @brief Kernel that converts inputs from `in` to `out`  using the following
 *        rule:   Convert  -NaN  -> NaN
 *                Convert  -0.0  -> 0.0
 *
 * @param[in] column_device_view representing input data
 * @param[in] mutable_column_device_view representing output data. can be
 *            the same actual underlying buffer that in points to. 
 *
 * @returns
 */
/* ----------------------------------------------------------------------------*/
template <typename T>
__global__
void normalize_nans_and_zeros(column_device_view in, 
                              mutable_column_device_view out)
{
   int tid = threadIdx.x;
   int blkid = blockIdx.x;
   int blksz = blockDim.x;
   int gridsz = gridDim.x;

   int start = tid + blkid * blksz;
   int step = blksz * gridsz;

   // grid-stride
   for (int i=start; i<in.size(); i+=step) {
      if(!in.is_valid(i)){
         continue;
      }

      T el = in.element<T>(i);
      if(std::isnan(el)){
         out.element<T>(i) = std::numeric_limits<T>::quiet_NaN();
      } else if(el == (T)-0.0){
         out.element<T>(i) = (T)0.0;
      } else {
         out.element<T>(i) = el;
      }
   }
}                        

  /* --------------------------------------------------------------------------*/
  /**
   * @brief Functor called by the `type_dispatcher` in order to invoke and instantiate
   *        `normalize_nans_and_zeros` with the appropriate data types.
   */
  /* ----------------------------------------------------------------------------*/
struct normalize_nans_and_zeros_kernel_forwarder {
   // floats and doubles. what we really care about.
   template <typename T, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
   void operator()(  column_device_view in,
                     mutable_column_device_view out,
                     cudaStream_t stream)
   {
      cudf::util::cuda::grid_config_1d grid{in.size(), BLOCK_SIZE};
      normalize_nans_and_zeros<T><<<grid.num_blocks, BLOCK_SIZE, 0, stream>>>(in, out);
   }

   // if we get in here for anything but a float or double, that's a problem.
   template <typename T, std::enable_if_t<not std::is_floating_point<T>::value>* = nullptr>
   void operator()(  column_device_view in,
                     mutable_column_device_view out,
                     cudaStream_t stream)
   {
      CUDF_FAIL("Unexpected non floating-point type.");      
   }   
};

} // end anonymous namespace

namespace cudf {
namespace detail {

std::unique_ptr<column> normalize_nans_and_zeros( column_view input,                                                  
                                                  cudaStream_t stream,
                                                  rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource())
{   
    // to device. unique_ptr which gets automatically cleaned up when we leave
   auto device_in = column_device_view::create(input);
   
   // ultimately, the output.
   auto out = make_numeric_column(input.type(), input.size(), ALL_VALID, stream);
   // from device. unique_ptr which gets automatically cleaned up when we leave.
   auto device_out = mutable_column_device_view::create(*out);

   // invoke the actual kernel.  
   experimental::type_dispatcher(input.type(), 
                                 normalize_nans_and_zeros_kernel_forwarder{},
                                 *device_in,
                                 *device_out,
                                 stream);

   return out;                 
}                                                 

void normalize_nans_and_zeros(mutable_column_view in_out,
                              cudaStream_t stream)
{  
   // wrapping the in_out data in a column_view so we can call the same lower level code.
   // that we use for the non in-place version.
   column_view input = in_out;

   // to device. unique_ptr which gets automatically cleaned up when we leave
   auto device_in = column_device_view::create(input);

   // from device. unique_ptr which gets automatically cleaned up when we leave.   
   auto device_out = mutable_column_device_view::create(in_out);

    // invoke the actual kernel.  
   experimental::type_dispatcher(input.type(), 
                                 normalize_nans_and_zeros_kernel_forwarder{},
                                 *device_in,
                                 *device_out,
                                 stream);
} 

}  // namespace detail

/**
 * @brief Function that converts inputs from `input` using the following rule
 *        rule:   Convert  -NaN  -> NaN
 *                Convert  -0.0  -> 0.0
 *
 * @param[in] column_device_view representing input data
 * @param[in] device_memory_resource allocator for allocating output data 
 *
 * @returns new column
 */
std::unique_ptr<column> normalize_nans_and_zeros( column_view input,                                                                                                    
                                                  rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource())
{
   return detail::normalize_nans_and_zeros(input, 0, mr);;
}

/**
 * @brief Function that processes values in-place from `in_out` using the following rule
 *        rule:   Convert  -NaN  -> NaN
 *                Convert  -0.0  -> 0.0
 *
 * @param[in, out] mutable_column_view representing input data. data is processed in-place
 *
 * @returns new column
 */
void normalize_nans_and_zeros(mutable_column_view in_out)
{
   return detail::normalize_nans_and_zeros(in_out, 0);
}

} // namespace cudf

} // anonymous namespace

void ntest()
{
   float whee[10] = { 32.5f, -0.0f, 111.0f, -NAN, NAN, 1.0f, 0.0f, 54.3f };   
   int num_els = 8;

   uint32_t nan = *((uint32_t*)(&whee[1]));   

   printf("Before: ");
   for(int idx=0; idx<num_els; idx++){
      printf(idx < num_els ? "%.2f, " : "%.2f", whee[idx]);
   }
   printf("\n");

   // copy the data to a column (which is always on the device)
   auto test_data = cudf::make_numeric_column(cudf::data_type(cudf::FLOAT32), num_els, cudf::ALL_VALID, 0);      
   // there's an overloaded operator for this but I like to see what's
   // actually going on.
   auto view = test_data->mutable_view();
   cudaMemcpy(view.head(), whee, sizeof(float) * num_els, cudaMemcpyHostToDevice);

   // do it
   db_test::cudf::normalize_nans_and_zeros(view);

   // get the data back
   cudaMemcpy(whee, view.head(), sizeof(float) * num_els, cudaMemcpyDeviceToHost);
   
   uint32_t nan2 = *((uint32_t*)(&whee[1]));

   printf("After: ");
   for(int idx=0; idx<num_els; idx++){
      printf(idx < num_els ? "%.2f, " : "%.2f", whee[idx]);
   }
   printf("\n\n");
}
#endif   // old normalize_nans_and_zeros kernel method. never got used.

#if 0    // copy_if_else
//namespace db_test {

using namespace cudf;
using namespace std;
using namespace rmm;
using namespace rmm::mr;

namespace {

using namespace cudf; 

/* --------------------------------------------------------------------------*/
/**
* @brief Functor called by the `type_dispatcher` in order to perform a copy if/else
*        using a filter function to select from lhs/rhs columns.
*/
/* ----------------------------------------------------------------------------*/
struct copy_if_else_functor {
   template <typename T, typename Filter>
   void operator()(  Filter filter,
                     column_view const& lhs,
                     column_view const& rhs,
                     mutable_column_view& out,
                     cudaStream_t stream)
   {
      auto begin  = thrust::make_zip_iterator(thrust::make_tuple( thrust::make_counting_iterator(0),
                                                                  lhs.begin<T>(),
                                                                  rhs.begin<T>()));

      auto end  = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_counting_iterator(lhs.size()),
                                                               lhs.end<T>(),
                                                               rhs.end<T>()));
      
      thrust::transform(rmm::exec_policy(stream)->on(stream), begin, end, out.begin<T>(),
                        [filter] __device__ (thrust::tuple<size_type, T, T> i)
                        {
                           return filter(thrust::get<0>(i)) ? thrust::get<1>(i) : thrust::get<2>(i);
                        });
   } 
};

}  // end anonymous namespace

namespace cudf {
namespace detail {

/**
 * @brief   Returns a new column, where each element is selected from either @p lhs or 
 *          @p rhs based on the filter lambda. 
 * 
 * @p filter must be a functor or lambda with the following signature:
 * __device__ bool operator()(cudf::size_type i);
 * It should return true if element i of @p lhs should be selected, or false if element i of @p rhs should be selected. 
 *         
 * @throws cudf::logic_error if lhs and rhs are not of the same type
 * @throws cudf::logic_error if lhs and rhs are not of the same length 
 * @param[in] filter lambda. 
 * @param[in] left-hand column_view
 * @param[in] right-hand column_view
 * @param[in] mr resource for allocating device memory
 *
 * @returns new column with the selected elements
 */
template<typename Filter>
unique_ptr<column> copy_if_else( Filter filter, column_view const& lhs, column_view const& rhs,
                                 rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource(),
                                 cudaStream_t stream = 0)
{
   // output
   std::unique_ptr<column> out = experimental::allocate_like(lhs, lhs.size(), experimental::mask_allocation_policy::RETAIN, mr);
   auto mutable_view = out->mutable_view();
   
   cudf::experimental::type_dispatcher(lhs.type(), 
                                       copy_if_else_functor{},
                                       filter,
                                       lhs,
                                       rhs,
                                       mutable_view,
                                       stream);

   return out;
}

}  // namespace detail
}  // namespace cudf

namespace cudf {
namespace detail {

struct pfunk {
    column_device_view bool_mask_device;

    __device__ bool operator()(int i) const
    {
       return bool_mask_device.element<cudf::experimental::bool8>(i);
    }
};

unique_ptr<column> copy_if_else( column_view const& boolean_mask, column_view const& lhs, column_view const& rhs,
                                 rmm::mr::device_memory_resource *mr,
                                 cudaStream_t stream)
{
   CUDF_EXPECTS(lhs.type() == rhs.type(), "Both columns must be of the same type");
   CUDF_EXPECTS(lhs.size() == rhs.size(), "Both columns must be of the size");
   CUDF_EXPECTS(boolean_mask.type() == data_type(BOOL8), "Boolean mask column must be of type BOOL8");   
   CUDF_EXPECTS(boolean_mask.size() == lhs.size(), "Boolean mask column must be the same size as lhs and rhs columns");

   // filter in this case is a column made of bools
   auto bool_mask_device_ptr = column_device_view::create(boolean_mask);   
   column_device_view bool_mask_device = *bool_mask_device_ptr;   
   auto filter = [bool_mask_device] __device__ (cudf::size_type i) { return bool_mask_device.element<cudf::experimental::bool8>(i); }; 

   return copy_if_else(filter, lhs, rhs, mr, stream);
}

}  // namespace detail

/**
 * @brief   Returns a new column, where each element is selected from either @p lhs or 
 *          @p rhs based on the value of the corresponding element in @p boolean_mask
 *
 * Selects each element i in the output column from either @p rhs or @p lhs using the following rule:
 *          output[i] = (boolean_mask[i] == true) ? lhs[i] : rhs[i]
 *         
 * @throws cudf::logic_error if lhs and rhs are not of the same type
 * @throws cudf::logic_error if lhs and rhs are not of the same length
 * @throws cudf::logic_error if boolean mask is not of type bool8
 * @throws cudf::logic_error if boolean mask is not of the same length as lhs and rhs 
 * @param[in] column_view representing "left (true) / right (false)" boolean for each element
 * @param[in] left-hand column_view
 * @param[in] right-hand column_view
 * @param[in] mr resource for allocating device memory
 *
 * @returns new column with the selected elements
 */
unique_ptr<column> copy_if_else( column_view const& boolean_mask, column_view const& lhs, column_view const& rhs, 
                                 rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource())
{
   return detail::copy_if_else(boolean_mask, lhs, rhs, mr, 0);
}

}  // namespace cudf


// to keep names shorter
#define wrapper cudf::test::fixed_width_column_wrapper
using bool_wrapper = wrapper<cudf::experimental::bool8>;

template<typename T>
void copy_if_else_check(bool_wrapper const&  mask_w,
                        wrapper<T> const&    lhs_w,
                        wrapper<T> const&    rhs_w,
                        wrapper<T> const&    expected_w)
{
   // construct input views
   column mask(mask_w);
   column_view mask_v(mask);
   //
   column lhs(lhs_w);
   column_view lhs_v = lhs.view();
   //
   column rhs(rhs_w);
   column_view rhs_v = rhs.view();
   //
   column expected(expected_w);
   column_view expected_v = expected.view();

   // get the result
   auto out = cudf::copy_if_else(mask_v, lhs_v, rhs_v);
   column_view out_v = out->view();   

   T whee[64];
   cudaMemcpy(whee, out_v.head(), sizeof(T) * out_v.size(), cudaMemcpyDeviceToHost);

   // compare
   cudf::test::expect_columns_equal(out_v, expected_v);
}

void copy_if_else_test()
{
   {
      bool_wrapper   mask_w      { true, true, false, true, true }; 
      wrapper<int>   lhs_w       { 5, 5, 5, 5, 5 };
      wrapper<int>   rhs_w       { 6, 6, 6, 6, 6 };
      wrapper<int>   expected_w  { 5, 5, 6, 5, 5 };
      copy_if_else_check(mask_w, lhs_w, rhs_w, expected_w); 
   }

   {
      bool_wrapper   mask_w      { false, true, false, false, true };
      wrapper<double>lhs_w       { -10.0f, -10.0, -10.0, -10.0, -10.0 };
      wrapper<double>rhs_w       { 7.0, 7.0, 7.0, 7.0, 7.0 };
      wrapper<double>expected_w  { 7.0, -10.0, 7.0, 7.0, -10.0 };
      copy_if_else_check(mask_w, lhs_w, rhs_w, expected_w);
   }
}

#endif   // copy if else

//}  // db_test

int main()
{                
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

    // shut stuff down
   rmmFinalize();
}
