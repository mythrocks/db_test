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
// #include <utilities/cuda_utils.hpp>
// #include <cudf/detail/copy_if_else.cuh>
#include <cudf/legacy/rolling.hpp>

#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/column_utilities.hpp>

#include <cub/cub.cuh>

#include <rmm/device_scalar.hpp>

#include "nvstrings/NVStrings.h"

#include <cudf/scalar/scalar.hpp>

#include <cudf/detail/copy_if_else.cuh>


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

         int actual_null_count = 0;
         for(idx=0; idx<cv->size(); idx++){
            bool is_null = !(validity[idx / 32] & (1 << (idx % 32)));
            if(is_null){
               actual_null_count++;
            }
            std::cout << (is_null ? "0" : "1");
            if(idx < cv->size() - 1){
               std::cout << ", ";
            }
         }
         std::cout << "\nReported null count: " << cv->null_count();
         std::cout << "\nActual null count: " << actual_null_count << "\n";
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

         int actual_null_count = 0;
         for(idx=0; idx<c.size; idx++){
            bool is_null = !(validity[idx / 8] & (1 << (idx % 8)));
            if(is_null){
               actual_null_count++;
            }
            std::cout << (is_null ? "0" : "1");
            if(idx < c.size - 1){
               std::cout << ", ";
            }
         }         
         std::cout << "\nReported null count: " << c.null_count;
         std::cout << "\nActual null count: " << actual_null_count << "\n";
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
   /*
   // make sure we span at least 2 warps      
   int num_els = 64;
   
   bool mask[]    = { 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                     1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };   
   bool_wrapper mask_w(mask, mask + num_els);

   T lhs[]        = { 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 
                     5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5 };   
   bool lvalid[] = { 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                     0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0 };
   wrapper<T> lhs_w(lhs, lhs + num_els, lvalid);

   T rhs[]        = { 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                     6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6 };
   bool rvalid[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
   wrapper<T> rhs_w(rhs, rhs + num_els, rvalid);
   
   T expected[]   = { 5, 6, 5, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 
                     5, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5 };   
   bool valid_e[] = { 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                     0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0 };
   wrapper<T> expected_w(expected, expected + num_els, valid_e);
   */    
}
#endif   // copy if else


#if 0    // timestamp_parse_test
void timestamp_parse_test()
{
   std::vector<const char*> hstrs{"1776"};
   NVStrings *strs = NVStrings::create_from_array(hstrs.data(), hstrs.size());
   
   rmm::device_scalar<unsigned long> results(hstrs.size(),0);
   // strs->timestamp2long("%Y-%m-%d %H:%M:%S.%f %z", NVStrings::ms, results.data());
   strs->timestamp2long("%Y-%m", NVStrings::ms, results.data());

   unsigned long res = results.value();

   NVStrings *back = NVStrings::long2timestamp(results.data(), 1, NVStrings::ms, "%Y-%m-%d %H:%M:%S.%f %z");
   back->print();

   int whee = 10;
   whee++;
}

#endif   // timestamp_parse_test

/*
std::unique_ptr<column> _copy_if_else( cudf::scalar const& lhs, column_view const& rhs, column_view const& boolean_mask,
                                      rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource(),
                                      cudaStream_t stream = 0)
{
   CUDF_EXPECTS(lhs.type() == rhs.type(), "Both columns must be of the same type");   
   CUDF_EXPECTS(not boolean_mask.has_nulls(), "Boolean mask must not contain null values.");
   CUDF_EXPECTS(boolean_mask.type() == data_type(BOOL8), "Boolean mask column must be of type BOOL8");   
   CUDF_EXPECTS(boolean_mask.size() == rhs.size(), "Boolean mask column must be the same size as lhs and rhs columns");   

   auto bool_mask_device_p = column_device_view::create(boolean_mask);
   column_device_view bool_mask_device = *bool_mask_device_p;
   auto filter = [bool_mask_device] __device__ (cudf::size_type i) { return bool_mask_device.element<cudf::experimental::bool8>(i); };

   return cudf::experimental::detail::copy_if_else(lhs, rhs, filter, mr, stream);
}
*/

void copy_if_else_scalar_test()
{   
   using T = int;

   /*
   // short one. < 1 warp/bitmask length   
   int num_els = 5;

   bool mask[]    = { 1, 0, 0, 1, 0 };
   bool_wrapper mask_w(mask, mask + num_els);

   T lhs[]        = { 99, 5, 5, 99, 5 };
   bool lhs_v[]   = { 1, 1, 1, 1, 1 };  
   // wrapper<T> lhs_w(lhs, lhs + num_els, lhs_v);   
   cudf::numeric_scalar<T> lhs_w(88);   

   T rhs[]        = { 6, 6, 6, 6, 6 };  
   bool rhs_v[]   = { 1, 1, 1, 1, 0 };  
   // wrapper<T> rhs_w(rhs, rhs + num_els, rhs_v);
   // column_view rhs_c(rhs_w);
   cudf::numeric_scalar<T> rhs_w(77);   
      
   T expected[]        = { 99, 6, 6, 99, 6 };  
   bool expected_v[]   = { 1, 1, 1, 1, 0 };
   wrapper<T> expected_w(expected, expected + num_els, expected_v);   
      
   auto out = cudf::experimental::copy_if_else(lhs_w, rhs_w, mask_w);
   column_view out_v = out->view();
   print_column(out_v);
   cudf::test::expect_columns_equal(out->view(), expected_w);
   */
  /*
   int num_els = 4;

   bool mask[]    = { 1, 0, 0, 1 };
   bool_wrapper mask_w(mask, mask + num_els);

   cudf::numeric_scalar<T> lhs_w(5);

   cudf::numeric_scalar<T> rhs_w(6);
   
   T expected[]   = { 5, 6, 6, 5 };   
   wrapper<T> expected_w(expected, expected + num_els);

   auto out = cudf::experimental::copy_if_else(lhs_w, rhs_w, mask_w);
   column_view out_v = out->view();
   print_column(out_v);
   cudf::test::expect_columns_equal(out_v, expected_w);   
   */

   {
      int num_els = 4;

      bool mask[]    = { 1, 0, 1, 1 };
      bool_wrapper mask_w(mask, mask + num_els);

      T lhs[]        = { 5, 5, 5, 5 }; 
      bool lhs_m[]   = { 1, 1, 1, 0 };
      wrapper<T> lhs_w(lhs, lhs + num_els, lhs_m);

      T rhs[]        = { 6, 6, 6, 6 };
      bool rhs_m[]   = { 1, 0, 1, 1 };
      wrapper<T> rhs_w(rhs, rhs + num_els, rhs_m);      

      T expected[]   = { 5, 6, 5, 5 };
      bool exp_m[]   = { 1, 0, 1, 0 };
      wrapper<T> expected_w(expected, expected + num_els, exp_m);

      auto out = cudf::experimental::copy_if_else(lhs_w, rhs_w, mask_w);     
      column_view out_v = out->view();
      print_column(out_v);
      cudf::test::expect_columns_equal(out->view(), expected_w);  
   }
   
   {
      int num_els = 4;

      bool mask[]    = { 1, 0, 1, 1 };
      bool_wrapper mask_w(mask, mask + num_els);

      T lhs[]        = { 5, 5, 5, 5 }; 
      bool lhs_m[]   = { 1, 1, 1, 0 };
      wrapper<T> lhs_w(lhs, lhs + num_els, lhs_m);

      T rhs[]        = { 6, 6, 6, 6 };
      wrapper<T> rhs_w(rhs, rhs + num_els);      

      T expected[]   = { 5, 6, 5, 5 };
      bool exp_m[]   = { 1, 1, 1, 0 };
      wrapper<T> expected_w(expected, expected + num_els, exp_m);

      auto out = cudf::experimental::copy_if_else(lhs_w, rhs_w, mask_w);     
      column_view out_v = out->view();
      print_column(out_v);
      cudf::test::expect_columns_equal(out->view(), expected_w);  
   }
   {
      int num_els = 4;

      bool mask[]    = { 1, 0, 1, 1 };
      bool_wrapper mask_w(mask, mask + num_els);

      T lhs[]        = { 5, 5, 5, 5 };       
      wrapper<T> lhs_w(lhs, lhs + num_els);

      T rhs[]        = { 6, 6, 6, 6 };
      bool rhs_m[]   = { 1, 0, 1, 1 };
      wrapper<T> rhs_w(rhs, rhs + num_els, rhs_m);      

      T expected[]   = { 5, 6, 5, 5 };
      bool exp_m[]   = { 1, 0, 1, 1 };
      wrapper<T> expected_w(expected, expected + num_els, exp_m);

      auto out = cudf::experimental::copy_if_else(lhs_w, rhs_w, mask_w);     
      column_view out_v = out->view();
      print_column(out_v);
      cudf::test::expect_columns_equal(out->view(), expected_w);  
   }
   
   {
      int num_els = 4;

      bool mask[]    = { 1, 0, 1, 1 };
      bool_wrapper mask_w(mask, mask + num_els);

      T lhs[]        = { 5, 5, 5, 5 };       
      wrapper<T> lhs_w(lhs, lhs + num_els);

      T rhs[]        = { 6, 6, 6, 6 };
      wrapper<T> rhs_w(rhs, rhs + num_els);      

      T expected[]   = { 5, 6, 5, 5 };      
      wrapper<T> expected_w(expected, expected + num_els);

      auto out = cudf::experimental::copy_if_else(lhs_w, rhs_w, mask_w);     
      column_view out_v = out->view();
      print_column(out_v);
      cudf::test::expect_columns_equal(out->view(), expected_w);  
   }
}

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
   // timestamp_parse_test();
   // column_equal_test();
   copy_if_else_scalar_test();

    // shut stuff down
   rmmFinalize();
}
