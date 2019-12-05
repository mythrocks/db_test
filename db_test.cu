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
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
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
#include <cudf/detail/utilities/integer_utils.hpp>


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

void print_nullmask(column_view const& cv, bool draw_separators)
{
   int mask_size = ((cv.size() + 31) / 32) * sizeof(cudf::bitmask_type);
   cudf::bitmask_type *validity = (cudf::bitmask_type*)alloca(mask_size);
   cudaMemcpy(validity, cv.null_mask(), mask_size, cudaMemcpyDeviceToHost);

   std::cout << "V: ";

   int actual_null_count = 0;
   for(int idx=0; idx<cv.size(); idx++){
      // the validity mask doesn't have offset baked into it so we have to shift it
      // ourselves
      int v_index = idx + cv.offset();      

      bool is_null = !(validity[v_index / 32] & (1 << (v_index % 32)));
      if(is_null){
         actual_null_count++;
      }
      std::cout << (is_null ? "0" : "1");
      if(idx < cv.size() - 1){
         std::cout << ", ";
      }
   }
   if(draw_separators){
      std::cout << "\nReported null count: " << cv.null_count();
      std::cout << "\nActual null count: " << actual_null_count << "\n";
   } else {
      std::cout << "\n";
   }
}

template<typename T>
struct printy_impl {
   static void print_column(column_view const& cv, bool draw_separators)
   {            
      int idx;

      if(draw_separators){
         std::cout << "-----------------------------\n";
      }

      std::cout << "D: ";

      // print values
      T *values = (T*)alloca(sizeof(T) * cv.size());            
      cudaMemcpy(values, cv.begin<T>(), sizeof(T) * cv.size(), cudaMemcpyDeviceToHost);      
      for(idx=0; idx<cv.size(); idx++){
         std::cout << values[idx];
         if(idx < cv.size() - 1){
            std::cout << ", ";            
         }
      }
      std::cout << "\n";

      // print validity mask
      if(cv.nullable()){         
         print_nullmask(cv, draw_separators);
      }

      if(draw_separators){
         std::cout << "-----------------------------\n";
      }
   }
};

template<>
struct printy_impl<string_view> {
   static void print_column(column_view const& _cv, bool draw_separators)
   {    
      strings_column_view cv(_cv);      

      if(draw_separators){
         std::cout << "-----------------------------\n";
      }

      cudf::strings::print(cv, 0, -1, -1, ",");
      std::cout << "\n";

      // print validity mask
      if(_cv.nullable()){
         print_nullmask(_cv, draw_separators);
      }

      if(draw_separators){
         std::cout << "-----------------------------\n";
      }
   }
};

template<>
struct printy_impl<timestamp_D> {
   static void print_column(column_view const& cv, bool draw_separators){}
};
template<>
struct printy_impl<timestamp_s> {
   static void print_column(column_view const& cv, bool draw_separators){}
};
template<>
struct printy_impl<timestamp_ms> {
   static void print_column(column_view const& cv, bool draw_separators){}
};
template<>
struct printy_impl<timestamp_us> {
   static void print_column(column_view const& cv, bool draw_separators){}
};
template<>
struct printy_impl<timestamp_ns> {
   static void print_column(column_view const& cv, bool draw_separators){}
};
struct printy {
   template<typename T>
   void operator()(column_view const& c, bool draw_separators)
   {      
      printy_impl<T>::print_column(c, draw_separators);
   }
};

void print_column(column_view const& c, bool draw_separators = true) 
{ 
   cudf::experimental::type_dispatcher(c.type(), printy{}, c, draw_separators); 
}
// template <typename cT> void print_column(cT&c, bool draw_seperators = true) { cudf::experimental::type_dispatcher(c.type(), printy<cT>{&c}, draw_seperators); }
void print_table(table_view const &tv)
{ 
   std::cout << "-----------------------------\n";
   int idx;
   for(idx=0; idx<tv.num_columns(); idx++){
      print_column(tv.column(idx), false);
      if(idx < tv.num_columns() - 1){
         std::cout << "-\n";
      }
   }
   std::cout << "-----------------------------\n";
}

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

#if 0 // copy_if_else
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

#endif // copy_if_else

/*
__global__
void split_kernel(table_device_view const input, mutable_column_device_view &out)
{
   size_type row = blockIdx.x * blockDim.x + threadIdx.x;
   size_type column = blockIdx.y * blockDim.y + threadIdx.y;

   size_type stride_x = blockDim.x * gridDim.x;
   size_type stride_y = blockDim.y * gridDim.y;
}
*/

/*
class split_result

void table_split_test()
{    
   int num_els = 4;

   short c0[]        = { 3, 3, 3, 3 };       
   wrapper<int> c0_w(c0, c0 + num_els);

   float c1[]        = { 4, 4, 4, 4 };
   wrapper<float> c1_w(c1, c1 + num_els);
   
   std::vector<std::unique_ptr<cudf::column>> cols;
   cols.push_back(c0_w.release());
   cols.push_back(c1_w.release());
   cudf::experimental::table t{std::move(cols)};
   print_table(t);

   int whee = 10;
   whee++;
}
*/

/*
struct serialized_subtable_blob {  
   // CPU address. managed entirely by this class.
   // offsets for subcolumns. probably some packed format like:  
   // (uint) num_columns      
   // (for each column)
   //    (uint) absolute column offset
   //    (uint) absolute column validity mask offset
   void *get_header();
   size_t get_header_size();

   // GPU address. points into "all_data"
   void *get_ptr();
   size_t get_size();      
};
struct serialized_subtables {
   std::vector<serialized_subtable_blob>     subtables;
   std::unique_ptr<rmm::device_buffer>       all_data;  
};

// serialize all subtables (sender)
serialized_subtables serialize_subtables(cudf::experimental::table_view const& input, std::vector<size_type> const& splits);

// deserialize a single subtable (receiver)
table deserialize_subtable(void *header, void *data);

// example sender
void sender(table_view &input, std::vector<size_type> const& splits)
{
   // make subtables
   serialized_subtables out = memory_aligned_split(input, splits);

   // send to remote machines.  
   for(int idx=0; idx<out.subtables.size(); idx++){
      serialized_subtable_blob &sub = out.subtables[idx];
      sendify(sub.get_header(), sub.get_header_size(), sub.get_ptr(), sub.get_size());
   }  
}

// example receiver
// since this memory is completely gamed, you will be responsible for calling table::release() on
// anything that comes out of here. 
table receiver(void *header, size_t header_size, void *data, size_t data_size)
{
   // don't really need header_size and data_size here since the header should have everything it needs.
   return deserialize_subtable(header, data);
}
*/



std::unique_ptr<column> make_strings(std::vector<const char*> _strings)
{   
   cudf::test::strings_column_wrapper strings( _strings.begin(), _strings.end(),
        thrust::make_transform_iterator( _strings.begin(), [] (auto str) { return str!=nullptr; }));
      
   return strings.release();
}

struct contiguous_split_result {
   cudf::table_view                    table;
   std::unique_ptr<rmm::device_buffer> all_data;
};

template<typename T>
struct size_functor_impl {
   void operator()(size_type *tp)
   {
      *tp = sizeof(T);
   }
};

template<>
struct size_functor_impl<string_view> {
   void operator()(size_type *tp){};
};

struct size_functor {
   template<typename T>
   void operator()(size_type *tp)
   {
      size_functor_impl<T> sizer{};
      sizer(tp);
   }
};

typedef std::vector<cudf::column_view> subcolumns;
static constexpr int split_align = 8;

std::vector<std::unique_ptr<rmm::device_buffer>> build_output_buffers_reference( std::vector<subcolumns> const& split_table, 
                                                                                 size_type *type_sizes,
                                                                                 rmm::mr::device_memory_resource* mr,
                                                                                 cudaStream_t stream)
{ 
   size_t num_out_tables = split_table[0].size();
   std::vector<std::unique_ptr<rmm::device_buffer>> result;

   // output packing for a given table will be:
   // (C0)(V0)(C1)(V1)
   // where Cx = column x and Vx = validity mask x
   // padding to split_align boundaries between each buffer
   for(size_t t_idx=0; t_idx<num_out_tables; t_idx++){  
      size_t subtable_size = 0;

      for(size_t c_idx=0; c_idx<split_table.size(); c_idx++){         
         column_view const& subcol = split_table[c_idx][t_idx];
         
         // special case for strings. we'll handle this eventually
         if(subcol.type().id() == cudf::STRING){
            continue;
         }
         size_t data_size = cudf::util::div_rounding_up_safe(subcol.size() * type_sizes[subcol.type().id()], split_align) * split_align;
         subtable_size += data_size;
         if(subcol.nullable()){
            size_t valid_size = cudf::bitmask_allocation_size_bytes(subcol.size(), split_align);
            subtable_size += valid_size;
         }
      }

      // allocate
      result.push_back(std::make_unique<rmm::device_buffer>(rmm::device_buffer{subtable_size, stream, mr}));
   }
   
   return result;
}

std::vector<contiguous_split_result> perform_split_reference(  std::vector<subcolumns> const& split_table,
                                                               size_type *type_sizes,
                                                               std::vector<std::unique_ptr<rmm::device_buffer>>& output_buffers)
{    
   size_t num_out_tables = split_table[0].size();
   std::vector<contiguous_split_result> result;

   // output packing for a given table will be:
   // (C0)(V0)(C1)(V1)
   // where Cx = column x and Vx = validity mask x
   // padding to split_align boundaries between each buffer
   for(size_t t_idx=0; t_idx<num_out_tables; t_idx++){
      // destination column_views and backing data
      std::vector<column_view> dst_cols;
      char *dst = static_cast<char*>(output_buffers[t_idx]->data());      
            
      for(size_t c_idx=0; c_idx<split_table.size(); c_idx++){         
         column_view const& subcol = split_table[c_idx][t_idx];
         
         // special case for strings. we'll handle this eventually
         if(subcol.type().id() == cudf::STRING){
            continue;
         }                  

         // data size         
         size_t data_size = cudf::util::div_rounding_up_safe(subcol.size() * type_sizes[subcol.type().id()], split_align) * split_align;

         // see if we've got validity
         bitmask_type *dst_validity = nullptr;
         size_t validity_size = 0;
         if(subcol.nullable()){
            dst_validity = reinterpret_cast<bitmask_type*>(dst + data_size);
            validity_size = cudf::bitmask_allocation_size_bytes(subcol.size(), split_align);
         }         
         
         // copy the data
         mutable_column_view new_col{subcol.type(), subcol.size(), dst, dst_validity};
         cudf::experimental::copy_range(subcol, new_col, 0, subcol.size(), 0);         

         // push the final column_view to the output
         dst_cols.push_back(new_col);         

         // increment dst buffers
         dst += (data_size + validity_size);                 
      }

      // construct the final table view
      result.push_back(contiguous_split_result{table_view{dst_cols}, std::move(output_buffers[t_idx])});
   }
      
   return result;
}

// do it all the simple way on the 
// void split_reference(cudf::experimental::table_view const& input, std::vector<size_type> const& splits)
std::vector<contiguous_split_result> contiguous_split_reference(  cudf::table_view const& input,
                                                                  std::vector<size_type> const& splits,
                                                                  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                                                  cudaStream_t stream = 0)
{   
   // generate per-type size info
   size_type type_sizes[cudf::NUM_TYPE_IDS] = { 0 };
   for(int idx=cudf::EMPTY+1; idx<NUM_TYPE_IDS; idx++){      
      if(idx == cudf::CATEGORY){ continue; }
      cudf::experimental::type_dispatcher(cudf::data_type((type_id)idx), size_functor{}, &type_sizes[idx]);
   }
   
   // slice each column into a set of sub-columns
   std::vector<subcolumns> split_table;
   for(auto iter = input.begin(); iter != input.end(); iter++){   
      split_table.push_back(cudf::experimental::slice(*iter, splits));
   }   

   // compute total output sizes. doing this as a seperate step because it's likely
   // that a kernel-y way of doing this will be two passes and the first step might be reusable
   std::vector<std::unique_ptr<rmm::device_buffer>> output_buffers = build_output_buffers_reference(split_table, type_sizes, mr, stream);

   // output data
   return perform_split_reference(split_table, type_sizes, output_buffers);
}

std::vector<contiguous_split_result> contiguous_split(cudf::table_view const& input,
                                                      std::vector<size_type> const& splits,
                                                      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                                      cudaStream_t stream = 0)
{
   return contiguous_split_reference(input, splits);
}

void split_test()
{
   /*
   std::vector<std::unique_ptr<column>> columns;
   int c0d[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
   wrapper<int> c0(c0d, c0d + 10);
   columns.push_back(c0.release());

   short c1d[] = { 20, 21, 22, 23, 24, 25, 26, 27, 28, 29 };
   wrapper<short> c1(c1d, c1d + 10);
   columns.push_back(c1.release());

   double c2d[] = { 30, 31, 32, 33, 34, 35, 36, 37, 38, 39 };
   wrapper<double> c2(c2d, c2d + 10);
   columns.push_back(c2.release());
   
   cudf::experimental::table t(std::move(columns));
   print_table(t.view());

   std::vector<size_type> splits { 0, 5, 5, 10 };

   auto out = contiguous_split(t.view(), splits);
   
   for(size_t idx=0; idx<out.size(); idx++){
      print_table(out[idx].table);
   }
   */

   std::vector<std::unique_ptr<column>> columns;
   int c0d[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
   bool c0v[] ={ 1, 1, 1, 1, 0, 0, 1, 1, 1, 1 };
   wrapper<int> c0(c0d, c0d + 10, c0v);
   columns.push_back(c0.release());

   short c1d[] = { 20, 21, 22, 23, 24, 25, 26, 27, 28, 29 };
   bool c1v[] ={ 1, 1, 1, 0, 1, 1, 0, 1, 1, 1 };
   wrapper<short> c1(c1d, c1d + 10, c1v);
   columns.push_back(c1.release());

   double c2d[] = { 30, 31, 32, 33, 34, 35, 36, 37, 38, 39 };
   bool c2v[] ={ 1, 1, 0, 1, 1, 1, 1, 0, 1, 1 };
   wrapper<double> c2(c2d, c2d + 10, c2v);
   columns.push_back(c2.release());
   
   cudf::experimental::table t(std::move(columns));
   print_table(t.view());

   std::vector<size_type> splits { 0, 5, 5, 10 };

   auto out = contiguous_split(t.view(), splits);
   
   for(size_t idx=0; idx<out.size(); idx++){
      print_table(out[idx].table);
   }

   int whee = 10;
   whee++;

   /*
   int num_els = 3;
   int c0[] = { 0, 1, 2 };
   wrapper<int> c0_w(c0, c0 + num_els);
   std::vector<size_type> splits { 0, 2, 1, 3 };
      
   auto out = cudf::experimental::slice(c0_w, splits);

   for(size_t idx=0; idx<out.size(); idx++){
      print_column(out[idx]);      
   }
   
   auto c = make_strings( {"abcdefghijklmnopqrstuvwxyz",
                           "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                           "00000000000000000000000000"});
   column_view cv = c->view();
   std::vector<size_type> ssplits { 0, 2, 1, 3 };
   auto sout = cudf::experimental::slice(cv, ssplits);
   
   for(size_t idx=0; idx<sout.size(); idx++){
      print_column(sout[idx]);      
   } 

   int whee = 10;
   whee++;
   */  
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
   // copy_if_else_scalar_test();
   split_test();   

    // shut stuff down
   rmmFinalize();
}

