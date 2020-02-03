#include "db_test.cuh"

using namespace cudf;

thread_local int scope_timer_manual::indent = 0;

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

template<typename T, typename TDisplay = T>
struct printy_impl {
   //template<typename TDisplay = T>
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
         std::cout << static_cast<TDisplay>(values[idx]);
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
struct printy_impl<int8_t> {
   static void print_column(column_view const& _cv, bool draw_separators, int max_els)
   {
      printy_impl<int8_t, int> printy;
      printy.print_column(_cv, draw_separators, max_els);
   }
};

template<>
struct printy_impl<string_view> {
   static void print_column(column_view const& _cv, bool draw_separators, int max_els)
   {    
      // strings_column_view cv(_cv);      

      if(draw_separators){
         std::cout << "-----------------------------\n";
      }

      // cudf::strings::print(cv, 0, -1, -1, ",");
      cudf::test::print(_cv);
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

void print_column(column_view const& c, bool draw_separators, int max_els) 
{ 
   cudf::experimental::type_dispatcher(c.type(), printy{}, c, draw_separators, max_els); 
}
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

std::unique_ptr<cudf::experimental::table> create_random_int_table(cudf::size_type num_columns, cudf::size_type num_rows, bool include_validity)
{       
   auto valids = cudf::test::make_counting_transform_iterator(0, [](auto i) { return true; });
   std::vector<cudf::test::fixed_width_column_wrapper<int>> src_cols(num_columns);
   for(int idx=0; idx<num_columns; idx++){
      auto rand_elements = cudf::test::make_counting_transform_iterator(0, [](int i){return rand();});
      if(include_validity){
         src_cols[idx] = cudf::test::fixed_width_column_wrapper<int>(rand_elements, rand_elements + num_rows, valids);
      } else {
         src_cols[idx] = cudf::test::fixed_width_column_wrapper<int>(rand_elements, rand_elements + num_rows);
      }
   }      
   std::vector<std::unique_ptr<cudf::column>> columns(num_columns);
   std::transform(src_cols.begin(), src_cols.end(), columns.begin(), [](cudf::test::fixed_width_column_wrapper<int> &in){   
      auto ret = in.release();
      ret->set_null_count(0);
      return ret;
   });
   return std::make_unique<cudf::experimental::table>(std::move(columns));   
}