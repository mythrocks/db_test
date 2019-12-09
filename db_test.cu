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
#include <cudf/detail/utilities/cuda.cuh>

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

struct scope_timer_manual {
   timespec m_start;
   timespec m_end;
   char m_name[64];
   bool silent;

   static thread_local int indent;
   
   scope_timer_manual()
   {
      silent = true;
      //indent++;
   }
   scope_timer_manual(const char *_name) 
   {         
      //indent++;
      silent = false;
      strcpy(m_name, _name);      
   }    
   ~scope_timer_manual()
   {      
      //indent--;
   }

   void start()
   {
      clock_gettime(CLOCK_MONOTONIC, &m_start);
   }

   void end()
   {      
      clock_gettime(CLOCK_MONOTONIC, &m_end);          
      if(!silent){
         for(int idx=0; idx<indent; idx++){
            printf("   ");
         }
         printf("%s : %.2f ms\n", m_name, total_time_ms());
      }
   }

   float total_time_ms()
   {
      return (float)total_time_raw() / (float)1000000.0f;
   }
   size_t total_time_raw()
   {
      return ((1000000000 * m_end.tv_sec) + m_end.tv_nsec) - 
               ((1000000000 * m_start.tv_sec) + m_start.tv_nsec);
   }   
};
thread_local int scope_timer_manual::indent = 0;

struct scope_timer : public scope_timer_manual {    
    scope_timer() : scope_timer_manual() {}    
    scope_timer(const char *_name) : scope_timer_manual(_name)
    {         
        start();
    }
    ~scope_timer()
    {
        end();
    }
};

void print_nullmask(column_view const& cv, bool draw_separators, int max_els)
{   
   int num_els = max_els > 0 ? min(max_els, cv.size()) : cv.size();

   int mask_size = ((num_els + 31) / 32) * sizeof(cudf::bitmask_type);
   cudf::bitmask_type *validity = (cudf::bitmask_type*)alloca(mask_size);   
   cudaMemcpy(validity, (cv.null_mask() + cudf::word_index(cv.offset())), mask_size, cudaMemcpyDeviceToHost);

   std::cout << "V: ";

   int actual_null_count = 0;
   for(int idx=0; idx<num_els; idx++){
      // take intra-word offset into account
      int v_index = idx + (cv.offset() % 32);

      bool is_null = !(validity[v_index / 32] & (1 << (v_index % 32)));
      if(is_null){
         actual_null_count++;
      }
      std::cout << (is_null ? "0" : "1");
      if(idx < num_els - 1){
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
   static void print_column(column_view const& cv, bool draw_separators, int max_els)
   {            
      int idx;      

      if(draw_separators){
         std::cout << "-----------------------------\n";
      }

      std::cout << "D: ";

      int num_els = max_els > 0 ? min(max_els, cv.size()) : cv.size();

      // print values
      T *values = (T*)alloca(sizeof(T) * num_els);            
      cudaMemcpy(values, cv.begin<T>(), sizeof(T) * num_els, cudaMemcpyDeviceToHost);      
      for(idx=0; idx<num_els; idx++){
         std::cout << values[idx];
         if(idx < num_els - 1){
            std::cout << ", ";            
         }
      }
      std::cout << "\n";

      // print validity mask
      if(cv.nullable()){         
         print_nullmask(cv, draw_separators, max_els);
      }

      if(draw_separators){
         std::cout << "-----------------------------\n";
      }
   }
};

template<>
struct printy_impl<string_view> {
   static void print_column(column_view const& _cv, bool draw_separators, int max_els)
   {    
      strings_column_view cv(_cv);      

      if(draw_separators){
         std::cout << "-----------------------------\n";
      }

      cudf::strings::print(cv, 0, -1, -1, ",");
      std::cout << "\n";

      // print validity mask
      if(_cv.nullable()){
         print_nullmask(_cv, draw_separators, max_els);
      }

      if(draw_separators){
         std::cout << "-----------------------------\n";
      }
   }
};

template<>
struct printy_impl<timestamp_D> {
   static void print_column(column_view const& cv, bool draw_separators, int max_els){}
};
template<>
struct printy_impl<timestamp_s> {
   static void print_column(column_view const& cv, bool draw_separators, int max_els){}
};
template<>
struct printy_impl<timestamp_ms> {
   static void print_column(column_view const& cv, bool draw_separators, int max_els){}
};
template<>
struct printy_impl<timestamp_us> {
   static void print_column(column_view const& cv, bool draw_separators, int max_els){}
};
template<>
struct printy_impl<timestamp_ns> {
   static void print_column(column_view const& cv, bool draw_separators, int max_els){}
};
struct printy {
   template<typename T>
   void operator()(column_view const& c, bool draw_separators, int max_els)
   {      
      printy_impl<T>::print_column(c, draw_separators, max_els);
   }
};

void print_column(column_view const& c, bool draw_separators = true, int max_els = 0) 
{ 
   cudf::experimental::type_dispatcher(c.type(), printy{}, c, draw_separators, max_els); 
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

std::unique_ptr<column> make_strings(std::vector<const char*> _strings)
{   
   cudf::test::strings_column_wrapper strings( _strings.begin(), _strings.end(),
        thrust::make_transform_iterator( _strings.begin(), [] (auto str) { return str!=nullptr; }));
      
   return strings.release();
}


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

/*
struct contiguous_split_result {
   cudf::table_view                    table;
   std::unique_ptr<rmm::device_buffer> all_data;
};

template<typename T>
struct column_buf_size_functor_impl {
   void operator()(size_type *tp)
   {
      *tp = sizeof(T);
   }
};

template<>
struct column_buf_size_functor_impl<string_view> {
   void operator()(size_type *tp){};
};

struct column_buf_size_functor {
   template<typename T>
   void operator()(size_type *tp)
   {
      column_buf_size_functor_impl<T> sizer{};
      sizer(tp);
   }
};

typedef std::vector<cudf::column_view> subcolumns;
static constexpr int split_align = 8;

template<

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
*/

/*
void string_column_alloc_sizes(string_column_view const &scv, size_t& offsets_size, size_t& chars_size, size_t& validity_size)
{   
   CUDF_EXPECTS(scv.parent().size() == 0, "Found a string column with a non-zero length parent column");

   // making an assumption about offsets always being of type size_type. maybe a bad idea.      
   running_data_size += cudf::util::div_rounding_up_safe(sc.offsets().size() * sizeof(size_type), split_align) * split_align;
   running_data_size += cudf::util::div_rounding_up_safe(static_cast<size_t>(sc.chars().size()), split_align) * split_align;

   if(c.nullable()){
      running_validity_size += cudf::bitmask_allocation_size_bytes(c.size(), split_align);         
   }
}
*/


/*
      size_t offsets_size, chars_size, validity_size;
      string_column_alloc_sizes(strings_column_view{c}, offsets_size, chars_size, validity_size);
      running_data_size += (offset_size + chars_size);
      running_validity_size += validity_size;
      */

     /*
      strings_column_view sc(c);
      CUDF_EXPECTS(sc.parent().size() == 0, "Found a string column with a non-zero length parent column");

      // there's some unnecessary recomputation of sizes happening here, but it really shouldn't affect much.
      size_t offsets_size, chars_size, validity_size;      
      string_column_alloc_sizes(sc, offsets_size, chars_size, validity_size);      
      size_t data_size = offsets_size + chars_size;
      
      // outgoing pointers
      rmm::device_vector<size_type> offsets(static_cast<size_type*>(dst), static_cast<size_type*>(dst + offsets_size));
      rmm::device_vector<size_type> chars(static_cast<size_type*>(dst + offsets_size), static_cast<size_type*>(dst + data_size));            
      bitmask_type* validity = validity_size == 0 ? nullptr : reinterpret_cast<bitmask_type*>(dst + data_size);

      // increment working buffer
      dst += (data_size + validity_size);

      auto string_column = 

      mutable_column_view mcv{in.type(), in.size(), data, validity};      
      
      // this is a double-dispatch, but I'm using it because returning a column_view through this
      // functor is causing some odd compiler errors.
      cudf::experimental::copy_range(in, mcv, 0, in.size(), 0);
      out_cols.push_back(mcv); 
      */

template <size_type block_size, typename T, bool has_validity>
__launch_bounds__(block_size)
__global__
void copy_in_place_kernel( column_device_view const in,
                           mutable_column_device_view out
                           // PERFORMANCE TEST 2, 3, 4
                           /*,size_type * __restrict__ const valid_count*/)
{
   const size_type tid = threadIdx.x + blockIdx.x * block_size;
   const int warp_id = tid / cudf::experimental::detail::warp_size;
   const size_type warps_per_grid = gridDim.x * block_size / cudf::experimental::detail::warp_size;   

   // PERFORMANCE TEST 4
   /*
   if(threadIdx.x == 0){
      atomicCAS(valid_count, 0, 0);      
   }
   */

   // begin/end indices for the column data
   size_type begin = 0;
   size_type end = in.size();
   // warp indices.  since 1 warp == 32 threads == sizeof(bit_mask_t) * 8,
   // each warp will process one (32 bit) of the validity mask via
   // __ballot_sync()
   size_type warp_begin = cudf::word_index(begin);
   size_type warp_end = cudf::word_index(end-1);      

   // lane id within the current warp   
   const int lane_id = threadIdx.x % cudf::experimental::detail::warp_size;

   // PERFORMANCE TEST 2, 3, 4   
   // constexpr size_type leader_lane{0};
   // size_type warp_valid_count{0};

   // current warp.
   size_type warp_cur = warp_begin + warp_id;   
   size_type index = tid;
   while(warp_cur <= warp_end){
      bool in_range = (index >= begin && index < end);
            
      bool valid = true;
      if(has_validity){
         valid = in_range && in.is_valid(index);
      }
      if(in_range){
         out.element<T>(index) = in.element<T>(index);
      }
      
      // update validity      
      if(has_validity){
         // the final validity mask for this warp 
         int warp_mask = __ballot_sync(0xFFFF'FFFF, valid && in_range);
         // only one guy in the warp needs to update the mask and count
         if(lane_id == 0){            
            out.set_mask_word(warp_cur, warp_mask);
            
            // PERFORMANCE TEST 2, 3, 4   
            // warp_valid_count += __popc(warp_mask);
         }
      }            

      // next grid
      warp_cur += warps_per_grid;
      index += block_size * gridDim.x;
   }
   
   // PERFORMANCE TEST 2, 3, 4 
   /*
   if(has_validity){
      // sum all null counts across all warps
      size_type block_valid_count = cudf::experimental::detail::single_lane_block_sum_reduce<block_size, leader_lane>(warp_valid_count);
      // block_valid_count will only be valid on thread 0
      if(threadIdx.x == 0){
         // using an atomic here because there are multiple blocks doing this work
         atomicAdd(valid_count, block_valid_count);
      }
   } 
   */
}

static constexpr size_t split_align = 8;

template<typename T>
struct column_buf_size_functor_impl {
   void operator()(column_view const& c, size_t& running_data_size, size_t& running_validity_size)
   {
      running_data_size += cudf::util::div_rounding_up_safe(c.size() * sizeof(T), split_align) * split_align;      
      if(c.nullable()){
         running_validity_size += cudf::bitmask_allocation_size_bytes(c.size(), split_align);         
      }
   }
};

// TODO
template<>
struct column_buf_size_functor_impl<string_view> {
   void operator()(column_view const& c, size_t& running_data_size, size_t& running_validity_size){};
};

struct column_buf_size_functor {
   template<typename T>
   void operator()(column_view const& c, size_t& running_data_size, size_t& running_validity_size)
   {
      column_buf_size_functor_impl<T> sizer{};
      sizer(c, running_data_size, running_validity_size);
   }
};

template<typename T>
struct column_copy_functor_impl {
   void operator()(column_view const& in, char *&dst, std::vector<column_view>& out_cols
      // PERFORMANCE TEST 4
      /*, rmm::device_scalar<cudf::size_type> &valid_count*/)
   {      
      // there's some unnecessary recomputation of sizes happening here, but it really shouldn't affect much.
      size_t data_size = 0;
      size_t validity_size = 0;      
      column_buf_size_functor_impl<T>{}(in, data_size, validity_size);

      // outgoing pointers
      char* data = dst;
      bitmask_type* validity = validity_size == 0 ? nullptr : reinterpret_cast<bitmask_type*>(dst + data_size);

      // increment working buffer
      dst += (data_size + validity_size);      

      // custom copy kernel (which should probably just be an in-place copy() function in cudf.
      cudf::size_type num_els = cudf::util::round_up_safe(in.size(), cudf::experimental::detail::warp_size);
      constexpr int block_size = 256;
      cudf::experimental::detail::grid_1d grid{num_els, block_size, 1};      
      
      // so there's a significant performance issue that comes up. our incoming column_view objects
      // are the result of a slice.  because of this, they have an UNKNOWN_NULL_COUNT.  because of that,
      // calling column_device_view::create() will cause a recompute of the count, which ends up being
      // -extremely- slow because a.) the typical use case here will involve -huge- numbers of calls and
      // b.) the count recompute involves tons of device allocs and memcopies, which sort of nullifies
      // the entire point of contiguous_split.
      // so to get around this, I am manually constructing a fake-ish view here where the null
      // count is arbitrarily bashed to 0.
      //      
      // I ran 5 performance tests here, all on 6 gigs of input data, 512 columns split 256 ways, for a total of
      // 128k calls to this function.
      //
      // 1. 500 ms    : no valdity information.
      // 2. 10,000 ms  : validify information, leaving UNKNOWN_NULL_COUNTS in place. the time difference 
      //    here is in the null_count() recomputation that happens in column_device_view::create() and the time
      //    spent allocating/reading from the device scalar to get the resulting null count
      // 3. 3,600 ms  : validity information, faking 0 input null count,  allocating a device scalar on the spot, 
      //    recomputing null count in the copy_in_place_kernel and reading it back.
      // 4. 2,700 ms  : validity information, faking 0 input null count, keeping a global device scalar, 
      //    recomputing null count in the copy_in_place_kernel and reading it back.
      // 5. 500 ms    : validity information, faking 0 input null count, setting output null count to UNKNOWN_NULL_COUNT. the
      //    implication here of course is that someone else might end up paying this price later on down the road.
      //
      // Summary : nothing super surprising.  anything that causes memory allocation or copying between host and device
      //           is super slow and becomes extremely noticeable at scale. best bet here seems to be to go with case 5 and 
      //           let someone else pay the cost for lazily evaluating null counts down the road.
      //
      //

      // see performance note above about null counts.
      column_view          in_hacked{  in.type(), in.size(), in.head<T>(), 
                                       in.null_mask(), in.null_mask() == nullptr ? UNKNOWN_NULL_COUNT : 0,
                                       in.offset() };
      mutable_column_view  mcv{in.type(), in.size(), data, 
                               validity, validity == nullptr ? UNKNOWN_NULL_COUNT : 0 };      
      if(in.nullable()){         
         // PERFORMANCE TEST 2, 3
         // rmm::device_scalar<cudf::size_type> valid_count{0, 0, rmm::mr::get_default_resource()};
         copy_in_place_kernel<block_size, T, true><<<grid.num_blocks, block_size, 0, 0>>>(
                           *column_device_view::create(in_hacked), 
                           *mutable_column_device_view::create(mcv)
                           // PERFORMANCE TEST 2, 3, 4
                           //, valid_count.data()
                           );
         // PERFORMANCE TEST 2, 3, 4
         // mcv.set_null_count(in.size() - valid_count.value());                  
      } else
       {
         copy_in_place_kernel<block_size, T, false><<<grid.num_blocks, block_size, 0, 0>>>(
                           *column_device_view::create(in_hacked), 
                           *mutable_column_device_view::create(mcv)
                           // PERFORMANCE TEST 2, 3, 4
                           /*, nullptr*/);
      }
      mcv.set_null_count(cudf::UNKNOWN_NULL_COUNT);

      out_cols.push_back(mcv);
   }
};

// TODO
template<>
struct column_copy_functor_impl<string_view> {
   void operator()(column_view const& in, char *&dst, std::vector<column_view>& out_cols
                  // PERFORMANCE TEST 4
                  /*, rmm::device_scalar<cudf::size_type> &valid_count*/)
   {       
   };
};

struct column_copy_functor {
   template<typename T>   
   void operator()(column_view const& in, char *&dst, std::vector<column_view>& out_cols
                  // PERFORMANCE TEST 4
                  /*, rmm::device_scalar<cudf::size_type> &valid_count*/)
   {
      column_copy_functor_impl<T> fn{};      
      fn(in, dst, out_cols 
         // PERFORMANCE TEST 4
         /* , valid_count*/);
   }
};

struct contiguous_split_result {
   cudf::table_view                    table;
   std::unique_ptr<rmm::device_buffer> all_data;
};
typedef std::vector<cudf::column_view> subcolumns;

std::vector<contiguous_split_result> contiguous_split(cudf::table_view const& input,
                                                      std::vector<size_type> const& splits,
                                                      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                                      cudaStream_t stream = 0)
{    
   // slice each column into a set of sub-columns   
   std::vector<subcolumns> split_table;
   split_table.reserve(input.num_columns());
   for(int idx=0; idx<input.num_columns(); idx++){
      split_table.push_back(cudf::experimental::slice(input.column(idx), splits));
   }
   size_t num_out_tables = split_table[0].size();   

   std::vector<contiguous_split_result> result;

   // DEBUG --------------------------
      float total_alloc_time = 0.0f;
      int total_allocs = 0;
      float total_copy_time = 0.0f;
      int total_copies = 0;
   // DEBUG --------------------------

   // PERFORMANCE TEST 4
   // rmm::device_scalar<cudf::size_type> valid_count{0, 0, rmm::mr::get_default_resource()};

   // output packing for a given table will be:
   // (C0)(V0)(C1)(V1)
   // where Cx = column x and Vx = validity mask x
   // padding to split_align boundaries between each buffer
   for(size_t t_idx=0; t_idx<num_out_tables; t_idx++){  
      size_t subtable_data_size = 0;
      size_t subtable_validity_size = 0;

      // compute sizes
      size_t sts = split_table.size();
      for(size_t c_idx=0; c_idx<split_table.size(); c_idx++){
         column_view const& subcol = split_table[c_idx][t_idx];         
         cudf::experimental::type_dispatcher(subcol.type(), column_buf_size_functor{}, subcol, subtable_data_size, subtable_validity_size);
      }
      
      // DEBUG --------------------------
         scope_timer_manual alloc_time;
         alloc_time.start();      
      // DEBUG --------------------------
      // allocate the blob
      auto device_buf = std::make_unique<rmm::device_buffer>(rmm::device_buffer{subtable_data_size + subtable_validity_size, stream, mr});
      char* buf = static_cast<char*>(device_buf->data());       
      // DEBUG --------------------------
         cudaDeviceSynchronize();
         alloc_time.end();
         total_alloc_time += alloc_time.total_time_ms();
         total_allocs++;
      // DEBUG --------------------------
            
      // DEBUG --------------------------
         scope_timer_manual copy_time;
         copy_time.start();
      // DEBUG --------------------------
      // create columns for the subtables
      std::vector<column_view> out_cols;
      out_cols.reserve(split_table.size());
      for(size_t c_idx=0; c_idx<split_table.size(); c_idx++){
         // copy
         column_view const& subcol = split_table[c_idx][t_idx];                  
         cudf::experimental::type_dispatcher(subcol.type(), column_copy_functor{}, subcol, buf, out_cols 
            // PERFORMANCE TEST 4
            /*, valid_count*/);
      }     
      // DEBUG --------------------------
         cudaDeviceSynchronize();
         copy_time.end();
         total_copy_time += copy_time.total_time_ms();
         total_copies += split_table.size();
      // DEBUG --------------------------
   
      result.push_back(contiguous_split_result{cudf::table_view{out_cols}, std::move(device_buf)});
   }
 
   // DEBUG --------------------------
      printf("  alloc time : %.2f (%d allocs)\n", total_alloc_time, total_allocs);
      printf("  copy time : %.2f (%d copies)\n", total_copy_time, total_copies);
   // DEBUG --------------------------

   // DEBUG --------------------------
         cudaDeviceSynchronize();
   // DEBUG --------------------------
   
   return result;
}


void verify_split_results( cudf::experimental::table const& src_table, 
                           std::vector<contiguous_split_result> const &dst_tables,
                           std::vector<size_type> const& splits,
                           int verbosity = 0)
{     
   table_view src_v(src_table.view());

   // printf("Verification : \n");
   // printf("%d, %d\n", src_v.num_columns(), (int)splits.size());   

   int col_count = 0;
   for(size_t c_idx = 0; c_idx<(size_t)src_v.num_columns(); c_idx++){
      for(size_t s_idx=0; s_idx<splits.size(); s_idx+=2){         
         // grab the subpiece of the src table
         auto src_subcol = cudf::experimental::slice(src_v.column(c_idx), std::vector<size_type>{splits[s_idx], splits[s_idx+1]});         
         
         // make sure it's the same as the output subtable's equivalent column
         size_t subtable_index = s_idx/2;         
         cudf::test::expect_columns_equal(src_subcol[0], dst_tables[subtable_index].table.column(c_idx), true);
         if(verbosity > 0 && (col_count % verbosity == 0)){
            printf("----------------------------\n");            
            print_column(src_subcol[0], false, 20);
            print_column(dst_tables[subtable_index].table.column(c_idx), false, 20);
            printf("----------------------------\n");
         }
         col_count++;        
      }
   }   
}

float frand()
{
   return (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * 65535.0f;
}

void single_split_test( int64_t total_desired_bytes, 
                        int64_t num_cols,                     
                        int64_t num_rows,
                        int64_t num_splits,
                        bool include_validity)
{
   printf("total data size : %.2f GB\n", (float)total_desired_bytes / (float)(1024 * 1024 * 1024));
   
   srand(31337);

   // generate input columns and table   
   std::vector<std::unique_ptr<column>> columns(num_cols);
   scope_timer_manual src_table_gen("src table gen");
   src_table_gen.start();
   std::vector<bool> all_valid(num_rows);
   if(include_validity){
      for(int idx=0; idx<num_rows; idx++){
         all_valid[idx] = true;
      }
   }   
   std::vector<int> icol(num_rows);
   std::vector<float> fcol(num_rows);
   for(int idx=0; idx<num_cols; idx++){
      if(idx % 2 == 0){                  
         for(int e_idx=0; e_idx<num_rows; e_idx++){
            icol[e_idx] = rand();
         }
         if(include_validity){            
            wrapper<int> cw(icol.begin(), icol.end(), all_valid.begin());            
            columns[idx] = cw.release();
            columns[idx]->set_null_count(0);
         } else {
            wrapper<int> cw(icol.begin(), icol.end());
            columns[idx] = cw.release();
            columns[idx]->set_null_count(0);
         }         
      } else {         
         for(int e_idx=0; e_idx<num_rows; e_idx++){
            fcol[e_idx] = frand();
         }
         if(include_validity){
            wrapper<float> cw(fcol.begin(), fcol.end(), all_valid.begin());
            columns[idx] = cw.release();
            columns[idx]->set_null_count(0);
         } else {
            wrapper<float> cw(fcol.begin(), fcol.end());
            columns[idx] = cw.release();
            columns[idx]->set_null_count(0);
         }         
      }
   }
   cudf::experimental::table src_table(std::move(columns));
   src_table_gen.end();
   printf("# columns : %d\n", (int)num_cols);

   // generate splits   
   int split_stride = num_rows / num_splits;
   std::vector<size_type> splits;  
   scope_timer_manual split_gen("split gen");
   split_gen.start();
   for(int idx=0; idx<num_rows; idx+=split_stride){
      splits.push_back(idx);
      splits.push_back(min(idx + split_stride, (int)num_rows));
   }
   split_gen.end();

   printf("# splits : %d\n", (int)splits.size() / 2);
   /*
   printf("splits : ");
   for(size_t idx=0; idx<splits.size(); idx+=2){
      printf("(%d, %d) ", splits[idx], splits[idx+1]);
   }
   */

   // do the split
   scope_timer_manual split_time("contiguous_split total");
   split_time.start();
   auto dst_tables = contiguous_split(src_table.view(), splits);
   cudaDeviceSynchronize();
   split_time.end();

   scope_timer_manual verify_time("verify_split_results");
   verify_time.start();
   verify_split_results(src_table, dst_tables, splits);
   verify_time.end();

   scope_timer_manual free_time("free buffers");
   free_time.start();   
   for(size_t idx=0; idx<dst_tables.size(); idx++){
      rmm::device_buffer *buf = dst_tables[idx].all_data.release();
      delete buf;   
   }
   cudaDeviceSynchronize();
   free_time.end();   
}

void large_split_tests()
{      
   // single_split_test does ints and floats only
   int el_size = 4;
      
   /*
   {
      // Tesla T4, 16 GB (all times in milliseconds)
      // total data size : 2.00 GB
      // src table gen : 8442.80 ms
      // # columns : 512
      // split gen : 0.00 ms
      // # splits : 256
      //    alloc time : 77.27 (256 allocs)     <------
      //    copy time : 436.92 (131072 copies)  <------
      // contiguous_split total : 524.31 ms     <------
      // verify_split_results : 6763.76 ms
      // free buffers : 0.18 ms                 <------
 
      // pick some numbers
      int64_t total_desired_bytes = (int64_t)2 * (1024 * 1024 * 1024);
      int64_t num_cols = 512;      
      int64_t num_rows = total_desired_bytes / (num_cols * el_size);
      int64_t num_splits = num_cols / 2;
      single_split_test(total_desired_bytes, num_cols, num_rows, num_splits, false);
   }
   */  
   
   /*
   {
      // Tesla T4, 16 GB (all times in milliseconds)
      // total data size : 2.00 GB
      // src table gen : 9383.00 ms
      // # columns : 512
      // split gen : 0.00 ms
      // # splits : 256
      //    alloc time : 43.93 (256 allocs)     <------
      //    copy time : 413.77 (131072 copies)  <------
      // contiguous_split total : 469.21 ms     <------
      // verify_split_results : 11387.72 ms
      // free buffers : 0.20 ms                 <------
 
      // pick some numbers
      int64_t total_desired_bytes = (int64_t)2 * (1024 * 1024 * 1024);
      int64_t num_cols = 512;      
      int64_t num_rows = total_desired_bytes / (num_cols * el_size);
      int64_t num_splits = num_cols / 2;
      single_split_test(total_desired_bytes, num_cols, num_rows, num_splits, true);
   }
   */  

   /*      
   {
      // Tesla T4, 16 GB (all times in milliseconds)
      // total data size : 4.00 GB
      // src table gen : 16917.02 ms
      // # columns : 512
      // split gen : 0.00 ms
      // # splits : 256
      //    alloc time : 79.27 (256 allocs)     <------
      //    copy time : 454.59 (131072 copies)  <------
      // contiguous_split total : 541.54 ms     <------
      // verify_split_results : 6777.47 ms
      // free buffers : 0.18 ms                 <------
 
      // pick some numbers
      int64_t total_desired_bytes = (int64_t)4 * (1024 * 1024 * 1024);
      int64_t num_cols = 512;      
      int64_t num_rows = total_desired_bytes / (num_cols * el_size);
      int64_t num_splits = num_cols / 2;
      single_split_test(total_desired_bytes, num_cols, num_rows, num_splits, false);
   } 
   */        

   /* 
   {
      // Tesla T4, 16 GB (all times in milliseconds)
      // total data size : 4.00 GB
      // src table gen : 18649.68 ms
      // # columns : 512
      // split gen : 0.00 ms
      // # splits : 256
      //    alloc time : 47.73 (256 allocs)     <------
      //    copy time : 446.58 (131072 copies)  <------
      // contiguous_split total : 503.26 ms     <------
      // verify_split_results : 11802.98 ms
      // free buffers : 0.26 ms                 <------
 
      // pick some numbers
      int64_t total_desired_bytes = (int64_t)4 * (1024 * 1024 * 1024);
      int64_t num_cols = 512;      
      int64_t num_rows = total_desired_bytes / (num_cols * el_size);
      int64_t num_splits = num_cols / 2;
      single_split_test(total_desired_bytes, num_cols, num_rows, num_splits, true);
   }   
   */
      
   {
      // Tesla T4, 16 GB
      // total data size : 6.00 GB
      // src table gen : 25230.81 ms
      // # columns : 512
      // split gen : 0.00 ms
      // # splits : 256
      //    alloc time : 48.01 (256 allocs)     <------
      //    copy time : 416.30 (131072 copies)  <------
      // contiguous_split total : 471.48 ms     <------
      // verify_split_results : 53921.47 ms
      // free buffers : 0.20 ms                 <------

      // pick some numbers
      int64_t total_desired_bytes = (int64_t)6 * (1024 * 1024 * 1024);
      int64_t num_cols = 512;      
      int64_t num_rows = total_desired_bytes / (num_cols * el_size);
      int64_t num_splits = num_cols / 2;
      single_split_test(total_desired_bytes, num_cols, num_rows, num_splits, false);
   } 
   
   /*
   {
      // Tesla T4, 16 GB
      // total data size : 6.00 GB
      // src table gen : 27897.44 ms
      // # columns : 512
      // split gen : 0.00 ms
      // # splits : 256
      //    alloc time : 61.25 (256 allocs)     <------
      //    copy time : 447.05 (131072 copies)  <------
      // contiguous_split total : 517.05 ms     <------
      // verify_split_results : 13794.44 ms
      // free buffers : 0.20 ms                 <------
 
      // pick some numbers
      int64_t total_desired_bytes = (int64_t)6 * (1024 * 1024 * 1024);
      int64_t num_cols = 512;      
      int64_t num_rows = total_desired_bytes / (num_cols * el_size);
      int64_t num_splits = num_cols / 2;
      single_split_test(total_desired_bytes, num_cols, num_rows, num_splits, true);
   }*/ 
   
   /*
   {
      // Tesla T4, 16 GB
      // total data size : 6.00 GB
      // src table gen : 28402.29 ms
      // # columns : 10
      // split gen : 0.01 ms
      // # splits : 257
      //    alloc time : 45.74 (257 allocs)     <------
      //    copy time : 70.60 (2570 copies)     <------
      // contiguous_split total : 116.76 ms     <------
      // verify_split_results : 1962.77 ms
      // free buffers : 0.24 ms                 <------
 
      // pick some numbers
      int64_t total_desired_bytes = (int64_t)6 * (1024 * 1024 * 1024);
      int64_t num_cols = 10;
      int64_t num_rows = total_desired_bytes / (num_cols * el_size);
      int64_t num_splits = 256;
      single_split_test(total_desired_bytes, num_cols, num_rows, num_splits, false);
   }
   */

   /*
    {
      // Tesla T4, 16 GB
      // total data size : 6.00 GB
      // src table gen : 30930.70 ms
      // # columns : 10
      // split gen : 0.00 ms
      // # splits : 257
      //    alloc time : 42.77 (257 allocs)     <------
      //    copy time : 72.51 (2570 copies)     <------
      // contiguous_split total : 115.61 ms     <------
      // verify_split_results : 2088.58 ms
      // free buffers : 0.25 ms                 <------
 
      // pick some numbers
      int64_t total_desired_bytes = (int64_t)6 * (1024 * 1024 * 1024);
      int64_t num_cols = 10;
      int64_t num_rows = total_desired_bytes / (num_cols * el_size);
      int64_t num_splits = 256;
      single_split_test(total_desired_bytes, num_cols, num_rows, num_splits, true);
   }
   */
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
   
   verify_split_results(t, out, splits, true);           

   /*
   for(size_t idx=0; idx<out.size(); idx++){
      print_table(out[idx].table);
   }
   */

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
   int whee = 10;
   whee++;
   */

   /*
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
   rmm.allocation_mode = PoolAllocation; // CudaDefaultAllocation
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
   large_split_tests();
   // split_test();   

    // shut stuff down
   rmmFinalize();
}

