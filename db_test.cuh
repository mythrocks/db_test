#pragma once

#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <iostream>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/copying.hpp>
#include <cudf/replace.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf/detail/utilities/cuda.cuh>

#include <rmm/rmm_api.h>

#include <cub/cub.cuh>

#include <rmm/device_scalar.hpp>

#include "nvstrings/NVStrings.h"

#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>

#include <cudf/detail/copy_if_else.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>

// #include <cudf/lists/list_view.cuh>

#include <arrow/io/api.h>

#include <cudf/binaryop.hpp>

#include <cudf/filling.hpp>

#include <fstream>

#include <cudf/detail/gather.hpp>
#include <cudf/dictionary/encode.hpp>
#include <rmm/rmm_api.h>
#include <simt/chrono>

#define wrapper cudf::test::fixed_width_column_wrapper
using float_wrapper = wrapper<float>;
using float64_wrapper = wrapper<double>;
using int_wrapper = wrapper<int>;
using int8_wrapper = wrapper<int8_t>;
using int64_wrapper = wrapper<int64_t>;

struct scope_timer_manual {
   timespec m_start;
   timespec m_end;
   char m_name[64];
   bool silent;

   static thread_local int indent;
   
   scope_timer_manual()
   {
      silent = true;
      indent++;
   }
   scope_timer_manual(const char *_name) 
   {         
      indent++;
      silent = false;
      strcpy(m_name, _name);      
   }    
   ~scope_timer_manual()
   {      
      indent--;
   }

   void start()
   {
      int err = clock_gettime(CLOCK_MONOTONIC, &m_start);
      CUDF_EXPECTS(err == 0, "clock_gettime() failed");
   }

   void end()
   {            
      int err = clock_gettime(CLOCK_MONOTONIC, &m_end);          
      CUDF_EXPECTS(err == 0, "clock_gettime() failed");
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
   float total_time_ns()
   {
      return (float)total_time_raw() / (float)1000.0f;
   }
   size_t total_time_raw()
   {
      return ((1000000000 * m_end.tv_sec) + m_end.tv_nsec) - 
               ((1000000000 * m_start.tv_sec) + m_start.tv_nsec);
   }   
};

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

#define UNREFERENCED(_x)    do { (void)(_x); } while(0)

void print_column(cudf::column_view const& c, bool draw_separators = true, int max_els = 0);
void print_table(cudf::table_view const &tv);

/*
template<typename T>
std::unique_ptr<cudf::table> create_random_fixed_table(cudf::size_type num_columns, cudf::size_type num_rows, bool include_validity)
{       
   auto valids = cudf::test::make_counting_transform_iterator(0, 
      [](auto i) { 
        return i % 2 == 0 ? true : false; 
      }
    );
   std::vector<cudf::test::fixed_width_column_wrapper<T>> src_cols(num_columns);
   for(int idx=0; idx<num_columns; idx++){
      auto rand_elements = cudf::test::make_counting_transform_iterator(0, [](T i){return rand();});
      if(include_validity){
         src_cols[idx] = cudf::test::fixed_width_column_wrapper<T>(rand_elements, rand_elements + num_rows, valids);
      } else {
         src_cols[idx] = cudf::test::fixed_width_column_wrapper<T>(rand_elements, rand_elements + num_rows);
      }
   }      
   std::vector<std::unique_ptr<cudf::column>> columns(num_columns);
   std::transform(src_cols.begin(), src_cols.end(), columns.begin(), [](cudf::test::fixed_width_column_wrapper<T> &in){   
      auto ret = in.release();
      ret->has_nulls();
      return ret;
   });
   return std::make_unique<cudf::table>(std::move(columns));   
}
*/

template<typename T>
std::unique_ptr<cudf::table> create_random_fixed_table(cudf::size_type num_columns, cudf::size_type num_rows, bool include_validity)
{       
   auto valids = cudf::test::make_counting_transform_iterator(0, 
      [](auto i) { 
        return i % 2 == 0 ? true : false; 
      }
    );
   std::vector<cudf::test::fixed_width_column_wrapper<T>> src_cols(num_columns);
   for(int idx=0; idx<num_columns; idx++){
      auto rand_elements = cudf::test::make_counting_transform_iterator(0, [](T i){return rand();});
      if(include_validity){
         src_cols[idx] = cudf::test::fixed_width_column_wrapper<T>(rand_elements, rand_elements + num_rows, valids);
      } else {
         src_cols[idx] = cudf::test::fixed_width_column_wrapper<T>(rand_elements, rand_elements + num_rows);
      }
   }      
   std::vector<std::unique_ptr<cudf::column>> columns(num_columns);
   std::transform(src_cols.begin(), src_cols.end(), columns.begin(), [](cudf::test::fixed_width_column_wrapper<T> &in){   
      auto ret = in.release();
      ret->has_nulls();
      return ret;
   });
   return std::make_unique<cudf::table>(std::move(columns));   
}