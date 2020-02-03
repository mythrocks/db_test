#pragma once

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
#include <cudf/legacy/rolling.hpp>

#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <cudf/detail/utilities/cuda.cuh>

#include <cub/cub.cuh>

#include <rmm/device_scalar.hpp>

#include "nvstrings/NVStrings.h"

#include <cudf/scalar/scalar.hpp>

#include <cudf/detail/copy_if_else.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>

#include <cudf/legacy/io_readers.hpp>
#include <tests/io/legacy/io_test_utils.hpp>
#include <cudf/io/functions.hpp>

#include <arrow/io/api.h>

#include <fstream>

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
void print_gdf_column(gdf_column const& c);

std::unique_ptr<cudf::experimental::table> create_random_int_table(cudf::size_type num_columns, cudf::size_type num_rows, bool include_validity);