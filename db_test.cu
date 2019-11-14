#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <iostream>

#include <cudf/cudf.h>
#include <tests/utilities/legacy/column_wrapper.cuh>
#include <bitmask/legacy/bit_mask.cuh>
#include <bitmask/legacy/legacy_bitmask.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/copying.hpp>
#include <cudf/replace.hpp>
#include <utilities/cuda_utils.hpp>
// #include <cudf/detail/copy_if_else.cuh>
#include <cudf/legacy/rolling.hpp>

#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/column_utilities.hpp>

#include <cub/cub.cuh>


// ------------------
//
// Defines
//
// ------------------

#define UNREFERENCED(_x)    do { (void)(_x); } while(0)

using namespace cudf;

// to keep names shorter
#define wrapper cudf::test::fixed_width_column_wrapper
using bool_wrapper = wrapper<cudf::experimental::bool8>;

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

template <typename cType>
struct printy {   
   cType *cv;

   template <typename T>
   void operator()()
   {
      print_column<T>();
   }
   
   template <typename T, std::enable_if_t<! (std::is_floating_point<T>::value ||
                                             std::is_integral<T>::value) >* = nullptr>
   void print_column()
   {
      CUDF_FAIL("I can't print this");
   }   
   
   template <typename T, std::enable_if_t<  (std::is_floating_point<T>::value ||
                                             std::is_integral<T>::value) >* = nullptr>   
   void print_column() const
   {      
      int idx;

      std::cout << "-----------------------------\n";

      // print values
      T *values = (T*)alloca(sizeof(T) * cv->size());      
      cudaMemcpy(values, cv->head(), sizeof(T) * cv->size(), cudaMemcpyDeviceToHost);      
      for(idx=0; idx<cv->size(); idx++){
         std::cout << values[idx];
         if(idx < cv->size() - 1){
            std::cout << ", ";            
         }
      }
      std::cout << "\n";

      // print validity mask
      if(cv->nullable()){
         int mask_size = ((cv->size() + 31) / 32) * sizeof(cudf::bitmask_type);
         cudf::bitmask_type *validity = (cudf::bitmask_type*)alloca(mask_size);
         cudaMemcpy(validity, cv->null_mask(), mask_size, cudaMemcpyDeviceToHost);      

         for(idx=0; idx<cv->size(); idx++){
            std::cout << (validity[idx / 32] & (1 << (idx % 32)) ? "1" : "0");
            if(idx < cv->size() - 1){
               std::cout << ", ";
            }
         }
         std::cout << "\nNull count: " << cv->null_count() << "\n";
      }

      std::cout << "-----------------------------\n";
   }
};
template <typename T> void print_column(T&c) { cudf::experimental::type_dispatcher(c.type(), printy<T>{&c}); }

struct gdf_printy {
   gdf_column const &c;

   template <typename T>
   void operator()()
   {
      print_column<T>();
   }
   
   template <typename T, std::enable_if_t<! (std::is_floating_point<T>::value ||
                                             std::is_integral<T>::value) >* = nullptr>
   void print_column()
   {
      CUDF_FAIL("I can't print this");
   }   
   
   template <typename T, std::enable_if_t<  (std::is_floating_point<T>::value ||
                                             std::is_integral<T>::value) >* = nullptr>   
   void print_column() const
   {      
      int idx;

      std::cout << "-----------------------------\n";

      // print values
      T *values = (T*)alloca(sizeof(T) * c.size);
      cudaMemcpy(values, c.data, sizeof(T) * c.size, cudaMemcpyDeviceToHost);      
      for(idx=0; idx<c.size; idx++){
         std::cout << values[idx];
         if(idx < c.size - 1){
            std::cout << ", ";            
         }
      }
      std::cout << "\n";

      // print validity mask
      if(c.valid != nullptr){
         int mask_size = ((c.size + 7) / 8) * sizeof(gdf_valid_type);
         gdf_valid_type *validity = (gdf_valid_type*)alloca(mask_size);
         cudaMemcpy(validity, c.valid, mask_size, cudaMemcpyDeviceToHost);      

         for(idx=0; idx<c.size; idx++){
            std::cout << (validity[idx / 8] & (1 << (idx % 8)) ? "1" : "0");
            if(idx < c.size - 1){
               std::cout << ", ";
            }
         }
         std::cout << "\nNull count: " << c.null_count << "\n";
      }

      std::cout << "-----------------------------\n";
   }
};
void print_gdf_column(gdf_column const& c) { cudf::experimental::type_dispatcher(cudf::data_type((cudf::type_id)c.dtype), gdf_printy{c}); }

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

// ------------------
//
// Work
//
// ------------------


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

/*

template <typename T, typename Filter, bool has_validity>
__global__
void copy_if_else_kernel(  column_device_view const lhs,
                           column_device_view const rhs,
                           SelectIter s_iter,
                           mutable_column_device_view out,
                           cudf::size_type * __restrict__ const null_count)
{   
   const cudf::size_type tid = threadIdx.x + blockIdx.x * blockDim.x;   
   const int w_id = tid / warp_size;
   // begin/end indices for the column data
   cudf::size_type begin = 0;
   cudf::size_type end = lhs.size();   
   // warp indices.  since 1 warp == 32 threads == sizeof(bit_mask_t) * 8,
   // each warp will process one (32 bit) of the validity mask via 
   // __ballot_sync()
   cudf::size_type w_begin = cudf::util::detail::bit_container_index<bit_mask::bit_mask_t>(begin);
   cudf::size_type w_end = cudf::util::detail::bit_container_index<bit_mask::bit_mask_t>(end);

   // lane id within the current warp
   const int w_lane_id = threadIdx.x % warp_size;

   // store a null count for each warp in the block
   constexpr cudf::size_type b_max_warps = 32;
   __shared__ uint32_t b_warp_null_count[b_max_warps];
   // initialize count to 0. we have to do this because the WarpReduce
   // at the end will end up summing all values, even ones which we never 
   // visit.
   if(has_validity){   
      if(threadIdx.x < b_max_warps){
         b_warp_null_count[threadIdx.x] = 0;
      }   
      __syncthreads();   
   }   

   // current warp.
   cudf::size_type w_cur = w_begin + w_id;         
   // process each grid
   while(w_cur <= w_end){
      // absolute element index
      cudf::size_type index = (w_cur * warp_size) + w_lane_id;
      bool in_range = (index >= begin && index < end);

      bool valid = true;
      if(has_validity){
         valid = in_range && filter(index) ? lhs.is_valid(index) : rhs.is_valid(index);
      }

      // do the copy if-else, but only if this element is valid in the column to be copied 
      if(in_range && valid){ 
         out.element<T>(index) = filter(index) ? lhs.element<T>(index) : rhs.element<T>(index);
      }
      
      // update validity
      if(has_validity){
         // get mask indicating which threads in the warp are actually in range
         int w_active_mask = __ballot_sync(0xFFFFFFFF, in_range);      
         // the final validity mask for this warp
         int w_mask = __ballot_sync(w_active_mask, valid);
         // only one guy in the warp needs to update the mask and count
         if(w_lane_id == 0){
            out.set_mask_word(w_cur, w_mask);
            cudf::size_type b_warp_cur = threadIdx.x / warp_size;
            b_warp_null_count[b_warp_cur] = __popc(~(w_mask | ~w_active_mask));
         }
      }      

      // next grid
      w_cur += blockDim.x * gridDim.x;
   }

   if(has_validity){
      __syncthreads();
      // first warp uses a WarpReduce to sum the null counts from all warps
      // within the block
      if(threadIdx.x < b_max_warps){
         // every thread collectively sums all the null counts using a WarpReduce      
         uint32_t w_null_count = b_warp_null_count[threadIdx.x];
         __shared__ typename cub::WarpReduce<uint32_t>::TempStorage temp_storage;
         uint32_t b_null_count = cub::WarpReduce<uint32_t>(temp_storage).Sum(w_null_count);

         // only one thread in the warp needs to do the actual store
         if(w_lane_id == 0){
            // using an atomic here because there are multiple blocks doing this work
            atomicAdd(null_count, b_null_count);
         }
      }
   }
}
*/
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
   //auto lhs_dv = column_device_view::create(lhs_v);   
   //
   column rhs(rhs_w);
   column_view rhs_v = rhs.view();
   //auto rhs_dv = column_device_view::create(rhs_v);
   //
   column expected(expected_w);
   column_view expected_v = expected.view();
    
   /*
   std::unique_ptr<column> out = experimental::allocate_like(rhs, rhs.size(), experimental::mask_allocation_policy::RETAIN);
   column_view out_v(out->view());
   auto out_dv = mutable_column_device_view::create(out->mutable_view());      
   cudf::util::cuda::grid_config_1d grid{lhs.size(), 256};
   cudf::size_type *null_count = nullptr;
   RMM_ALLOC(&null_count, sizeof(cudf::size_type), 0);
   cudf::size_type null_count_out = 0;
   cudaMemcpy(null_count, &null_count_out, sizeof(cudf::size_type), cudaMemcpyHostToDevice);   
   copy_if_else_kernel<T, cudf::experimental::bool8 const*, true><<<1, 256, 0, 0>>>(
      *lhs_dv,
      *rhs_dv,
      mask_v.begin<cudf::experimental::bool8>(),
      *out_dv,
      null_count); 
   cudaMemcpy(&null_count_out, null_count, sizeof(cudf::size_type), cudaMemcpyDeviceToHost);
   */
   auto out = cudf::experimental::copy_if_else(lhs_v, rhs_v, mask_v);
   column_view out_v(*out);

   cudf::test::expect_columns_equal(out_v, expected_v);

   print_column(out_v);
   print_column(expected_v);   
}

void copy_if_else_test()
{      
   using T = int;

   // short one. < 1 warp/bitmask length   
      int num_els = 4;

      bool mask[]    = { 1, 0, 0, 0 };
      bool_wrapper mask_w(mask, mask + num_els);

      T lhs[]        = { 5, 5, 5, 5 }; 
      bool lhs_v[]   = { 1, 1, 1, 1 };
      wrapper<T> lhs_w(lhs, lhs + num_els, lhs_v);

      T rhs[]        = { 6, 6, 6, 6 };
      bool rhs_v[]   = { 1, 0, 0, 0 };
      wrapper<T> rhs_w(rhs, rhs + num_els, rhs_v);
      
      T expected[]   = { 5, 6, 6, 6 };
      bool exp_v[]   = { 1, 0, 0, 0 };
      wrapper<T> expected_w(expected, expected + num_els, exp_v);

      auto out = cudf::experimental::copy_if_else(lhs_w, rhs_w, mask_w);      
      cudf::test::expect_columns_equal(out->view(), expected_w);   

   column_view out_v(*out);
   print_column(out_v);
   column expected_c(expected_w);
   column_view expected_v = expected_c.view();
   print_column(expected_v);
}

#endif   // copy if else

#if 0
void rolling_window_test()
{   
   gdf_col_pointer col = create_gdf_column<int>(std::vector<int>{5, 0, 7, 0, 8},
                                                std::vector<cudf::valid_type>{ 1<<0 | 1<<2 | 1<<4 });

   gdf_col_pointer expected = create_gdf_column<int>(std::vector<int>{0, 12, 0, 15, 0},
                                                std::vector<cudf::valid_type>{ 1<<1 | 1<<3 });

   print_gdf_column(*col);   
   
   gdf_column *out = cudf::rolling_window(*col, 2, 2, 1, GDF_SUM, nullptr, nullptr, nullptr);   

   print_gdf_column(*expected);
   print_gdf_column(*out);   

   //expect_columns_are_equal(col, expected, 

   /*
template <typename ColumnType>
gdf_col_pointer create_gdf_column(std::vector<ColumnType> const & host_vector,
                                  std::vector<cudf::valid_type> const & valid_vector = std::vector<cudf::valid_type>())

   
     void testWindowStaticWithNulls() {
    WindowOptions v0 = WindowOptions.builder().windowSize(2).minPeriods(2).forwardWindow(0).
        aggType(AggregateOp.SUM).build();
    try (ColumnVector v1 = ColumnVector.fromBoxedInts(0, 1, 2, null, 4);
        ColumnVector expected = ColumnVector.fromBoxedInts(null, 1, 3, null, null);
        ColumnVector result = v1.rollingWindow(v0)) {        
        assertColumnsAreEqual(expected, result);
    }
    */
}
#endif

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

   // copy_if_else_test();
   // rolling_window_test();

    // shut stuff down
   rmmFinalize();
}
