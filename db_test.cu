#include "db_test.cuh"

#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include <tests/copying/slice_tests.cuh>


using namespace cudf;

void print_names(std::vector<cudf::io::column_name_info> const &schema_info, std::string const& indent = "")
{
  for(size_t idx=0; idx<schema_info.size(); idx++){
    printf("%s%s\n", indent.c_str(), schema_info[idx].name.c_str());
    print_names(schema_info[idx].children, indent + "   ");
  }
}

namespace dave {
namespace test {

/*
bool compare(cudf::lists_column_view const& lhs, cudf::lists::column_view const& rhs)
{
  auto d_lhs = cudf::column_device_view::create(lhs.parent());
  auto d_rhs = cudf::column_device_view::create(rhs.parent());

  auto iter = thrust::make_counting_iterator(0);
  thrust::copy_if(thrust::device,
                  iter, iter + lhs.size(),
                  [lhs = *d_lhs, rhs = *d_rhs] __device__ (size_type index){
                    while(1){                      
                      // validity mismatch
                      if(lhs.is_valid() != rhs.is_valid()){
                        return true;
                      }

                      // if the row is null, we're done
                      if(!lhs.is_valid()){
                        return false;
                      }

                      // make sure offsets  match
                      auto lhs_offsets = lhs.child(cudf::lists_column_view::offsets_column_index);
                      size_type lhs_shift = lhs_offsets.head<size_type>(lhs.offset());                      
                      auto rhs_offsets = rhs.child(cudf::lists_column_view::offsets_column_index);
                      size_type rhs_shift = rhs_offsets.head<size_type>(rhs.offset());

                      size_type start = lhs_offsets.data<size_type>(index) - lhs_shift;
                      size_type end = lhs_offsets.data<size_type>(index) - lhs_shift;

                      // offset mismatch
                      if((lhs_offsets.data<size_type>(index) - lhs_shift != rhs_offsets.data<size_type>(index) - rhs_shift) ||
                          (lhs_offsets.data<size_type>(index+1) - lhs_shift != rhs_offsets.data<size_type>(index+1) - rhs_shift)){
                        return true;
                      }

                      // if this is another list, continue
                      if(lhs.child(cudf::lists_column_view::child_column_index(
                    }

                    // match
                    return false;
                  });
}
void compare_test()
{
  std::vector<bool> valids { true, true, true, false };

  cudf::test::lists_column_wrapper<int> a {{ {{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}, {{9, 10}}, {LCW{}} }, valids.begin()};
  cudf::test::print(a);

  cudf::test::lists_column_wrapper<int> b {{ {{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}, {{9, 10}}, {LCW{}} }, valids.begin()};
  auto bi = b.release();
  cudf::test::print(*bi);
  mutable_column_view mcv(*bi);
  auto b_mask = mcv.null_mask();
  bitmask_type bmv;
  cudaMemcpy(&bmv, b_mask, sizeof(bitmask_type), cudaMemcpyDeviceToHost);
  bmv &= ~(1<<1);  
  cudaMemcpy(b_mask, &bmv, sizeof(bitmask_type), cudaMemcpyHostToDevice);
  bi->set_null_count(cudf::UNKNOWN_NULL_COUNT);
  cudf::test::print(*bi);
}
*/

template<typename T>
std::unique_ptr<column> make_parquet_col(int skip_rows, int num_rows, int lists_per_row, int list_size, bool include_validity)
{
  auto valids = cudf::test::make_counting_transform_iterator( 0, [](auto i) { return i % 2 == 0 ? 1 : 0; });  

  // root list 
  int *row_offsets = new int[num_rows + 1];
  int row_offset_count = 0;
   {
    int offset = 0;
    for(int idx=0; idx<(num_rows)+1; idx++){
      row_offsets[row_offset_count] = offset;
      if(!include_validity || valids[idx]){
        offset += lists_per_row;
      }      
      row_offset_count++; 
    }
  }
  cudf::test::fixed_width_column_wrapper<int> offsets(row_offsets, row_offsets + row_offset_count);

  // cudf::test::print(offsets);
  printf("# rows : %d\n", static_cast<column_view>(offsets).size()-1);

  // child list
  int *child_row_offsets = new int[(num_rows * lists_per_row) + 1];
  int child_row_offset_count = 0;
  {
    int offset = 0;    
    for(int idx=0; idx<(num_rows * lists_per_row)+1; idx++){
      int row_index = idx / lists_per_row;
      if(include_validity && !valids[row_index]){
        continue;
      }

      child_row_offsets[child_row_offset_count] = offset;       
      offset += list_size;
      child_row_offset_count++;
    }
  }    
  cudf::test::fixed_width_column_wrapper<int> child_offsets(child_row_offsets, child_row_offsets + child_row_offset_count);
    
  // cudf::test::print(child_offsets);
  printf("# child rows : %d\n", static_cast<column_view>(child_offsets).size()-1);

  // child values
  T *child_values = new T[num_rows * lists_per_row * list_size];
  T first_child_value_index = skip_rows * lists_per_row * list_size;
  int child_value_count = 0;
  {
    for(int idx=0; idx<(num_rows * lists_per_row * list_size); idx++){
      int row_index = idx / lists_per_row;
      if(include_validity && !valids[row_index]){
        continue;
      }
      child_values[child_value_count] = child_value_count + first_child_value_index;
      child_value_count++;
    }
  }
/*
  T first_child_value_index = skip_rows * lists_per_row * list_size;  
  auto child_values = cudf::test::make_counting_transform_iterator(0, [list_size, first_child_value_index](T index){
    // return index % list_size;
    return index + first_child_value_index;
  });   
  auto child_data = include_validity ? 
    cudf::test::fixed_width_column_wrapper<T>(child_values, child_values + (num_rows * lists_per_row * list_size), valids) :
    cudf::test::fixed_width_column_wrapper<T>(child_values, child_values + (num_rows * lists_per_row * list_size));  
    */
  auto child_data = include_validity ? 
    cudf::test::fixed_width_column_wrapper<T>(child_values, child_values + child_value_count, valids) :
    cudf::test::fixed_width_column_wrapper<T>(child_values, child_values + child_value_count);  

  printf("# values : %d\n", static_cast<column_view>(child_data).size());

  // cudf::test::print(child_data);
  int child_offsets_size = static_cast<column_view>(child_offsets).size()-1;
  auto child = cudf::make_lists_column(child_offsets_size, child_offsets.release(), child_data.release(), 0, rmm::device_buffer{});
    
  int offsets_size = static_cast<column_view>(offsets).size()-1;
  return include_validity ?   
    cudf::make_lists_column(offsets_size, offsets.release(), std::move(child), UNKNOWN_NULL_COUNT, 
      cudf::test::detail::make_null_mask(valids, valids + offsets_size)) :
    cudf::make_lists_column(offsets_size, offsets.release(), std::move(child), 0, rmm::device_buffer{});
}

void parquet_big_list_test(bool include_validity)
{
  namespace cudf_io = cudf::io; 

  // auto expected = make_parquet_col(256, 80, 50);

  std::string filename = include_validity ? "lists_big2_valids.parquet" : "lists_big2.parquet";
  
  int max_rows = 256;  

  // read the whole file
  {
    int skip_rows = 0;
    int num_rows = max_rows;
    
    auto expected = make_parquet_col<int>(skip_rows, num_rows, 80, 50, include_validity);  
        
    cudf_io::parquet_reader_options read_args = cudf::io::parquet_reader_options::builder(cudf_io::source_info{filename});
    read_args.set_skip_rows(skip_rows);
    read_args.set_num_rows(num_rows);
    auto result = cudf_io::read_parquet(read_args);    
    //cudf::test::print(result.tbl->get_column(0));
    //printf("\n\n\n");    

    {
      column_view offsets_a = expected->child(0);
      column_view offsets_b = result.tbl->get_column(0).child(0);
      cudf::test::print(offsets_a);
      printf("\n\n\n");    
      cudf::test::print(offsets_b);
      cudf::test::expect_columns_equal(offsets_a, offsets_b);
      printf("offsets level 0 size : %d\n", offsets_a.size());
    }

    //cudf::test::print(result.tbl->get_column(0));

    {
      column_view offsets_a = expected->child(1).child(0);
      column_view offsets_b = result.tbl->get_column(0).child(1).child(0);
      //cudf::test::print(offsets_a);
      //printf("\n\n\n");    
      //cudf::test::print(offsets_b);
      cudf::test::expect_columns_equal(offsets_a, offsets_b);
      printf("offsets level 1 size : %d\n", offsets_b.size());
    }

    {
      column_view leaf_a = expected->child(1).child(1);  
      column_view leaf_b = result.tbl->get_column(0).child(1).child(1);        
      cudf::test::expect_columns_equal(leaf_a, leaf_b);
    }

    cudf::test::expect_columns_equal(*expected, result.tbl->get_column(0));  
    //cudf::test::print(*expected);
    //cudf::test::print(result.tbl->get_column(0));
  }
  
  // read one row at a time.
  {     
    int whee = 10;
    whee++;
    for(int idx=0; idx<max_rows; idx++){
      if(idx % 10 == 0){
        printf("Row : %d\n", idx);
      }
      int skip_rows = idx;
      int num_rows = 1;
      
      auto expected = make_parquet_col<int>(skip_rows, num_rows, 80, 50, include_validity);  

      cudf_io::parquet_reader_options read_args = cudf::io::parquet_reader_options::builder(cudf_io::source_info{filename});
      read_args.set_skip_rows(skip_rows);
      read_args.set_num_rows(num_rows);
      auto result = cudf_io::read_parquet(read_args);    
      //cudf::test::print(result.tbl->get_column(0));
      //printf("\n\n\n");    

      {
        column_view offsets_a = expected->child(0);
        column_view offsets_b = result.tbl->get_column(0).child(0);
        //cudf::test::print(offsets_a);
        //printf("\n\n\n");    
        //cudf::test::print(offsets_b);
        cudf::test::expect_columns_equal(offsets_a, offsets_b);
        printf("offsets level 0 size : %d\n", offsets_a.size());
      }

      {
        column_view offsets_a = expected->child(1).child(0);
        column_view offsets_b = result.tbl->get_column(0).child(1).child(0);
        cudf::test::expect_columns_equal(offsets_a, offsets_b);
        printf("offsets level 1 size : %d\n", offsets_b.size());
      }

      {
        column_view leaf_a = expected->child(1).child(1);  
        column_view leaf_b = result.tbl->get_column(0).child(1).child(1);  
        cudf::test::expect_columns_equal(leaf_a, leaf_b);
      }

      cudf::test::expect_columns_equal(*expected, result.tbl->get_column(0));  
      //cudf::test::print(*expected);
      //cudf::test::print(result.tbl->get_column(0));  
    }
  }

  // read the remaining rows after skip
  { 
    int whee = 10;

    for(int idx=0; idx<max_rows; idx++){
      if(idx % 10 == 0){
        printf("Row : %d\n", idx);
      }
      
      int skip_rows = idx;
      int num_rows = max_rows - skip_rows;

      auto expected = make_parquet_col<int>(skip_rows, num_rows, 80, 50, include_validity);  

      cudf_io::parquet_reader_options read_args = cudf::io::parquet_reader_options::builder(cudf_io::source_info{filename});
      read_args.set_skip_rows(skip_rows);
      read_args.set_num_rows(num_rows);
      auto result = cudf_io::read_parquet(read_args);    
      //cudf::test::print(result.tbl->get_column(0));
      //printf("\n\n\n");    

      {
        column_view offsets_a = expected->child(0);
        column_view offsets_b = result.tbl->get_column(0).child(0);
        //cudf::test::print(offsets_a);
        //printf("\n\n\n");    
        //cudf::test::print(offsets_b);
        cudf::test::expect_columns_equal(offsets_a, offsets_b);
        printf("offsets level 0 size : %d\n", offsets_a.size());
      }

      {
        column_view offsets_a = expected->child(1).child(0);
        column_view offsets_b = result.tbl->get_column(0).child(1).child(0);
        cudf::test::expect_columns_equal(offsets_a, offsets_b);
        printf("offsets level 1 size : %d\n", offsets_b.size());
      }

      {
        column_view leaf_a = expected->child(1).child(1);  
        column_view leaf_b = result.tbl->get_column(0).child(1).child(1);  
        cudf::test::expect_columns_equal(leaf_a, leaf_b);
      }

      cudf::test::expect_columns_equal(*expected, result.tbl->get_column(0));  
      //cudf::test::print(*expected);
      //cudf::test::print(result.tbl->get_column(0));  
    }
  }  

  // read 31 rows at a time
  {
    int whee = 10;
    whee++;    
    for(int skip_rows=0; skip_rows<max_rows; skip_rows += 31){
      /*
      if(idx % 10 == 0){
        printf("Row : %d\n", idx);
      }
      */
            
      int num_rows = min(max_rows - skip_rows, 31);

      printf("ROWS : %d, %d\n", skip_rows, num_rows);
      auto expected = make_parquet_col<int>(skip_rows, num_rows, 80, 50, include_validity);  

      cudf_io::parquet_reader_options read_args = cudf::io::parquet_reader_options::builder(cudf_io::source_info{filename});
      read_args.set_skip_rows(skip_rows);
      read_args.set_num_rows(num_rows);   
      auto result = cudf_io::read_parquet(read_args);    
      //cudf::test::print(result.tbl->get_column(0));
      //printf("\n\n\n");    

      {
        column_view offsets_a = expected->child(0);
        column_view offsets_b = result.tbl->get_column(0).child(0);
        //cudf::test::print(offsets_a);
        //printf("\n\n\n");    
        //cudf::test::print(offsets_b);
        cudf::test::expect_columns_equal(offsets_a, offsets_b);
        printf("offsets level 0 size : %d\n", offsets_a.size());
      }

      {
        column_view offsets_a = expected->child(1).child(0);
        column_view offsets_b = result.tbl->get_column(0).child(1).child(0);
        cudf::test::expect_columns_equal(offsets_a, offsets_b);
        printf("offsets level 1 size : %d\n", offsets_b.size());
      }

      {
        column_view leaf_a = expected->child(1).child(1);  
        column_view leaf_b = result.tbl->get_column(0).child(1).child(1);  
        cudf::test::expect_columns_equal(leaf_a, leaf_b);
      }

      cudf::test::expect_columns_equal(*expected, result.tbl->get_column(0));  
      //cudf::test::print(*expected);
      //cudf::test::print(result.tbl->get_column(0));  
    }
  }

  int whee = 10;
  whee++;
}

/*
required group list_int (LIST) {
    repeated group list {
      optional int32 element;
    }
  }
  optional group list_list_int (LIST) {
    repeated group list {
      optional group element (LIST) {
        repeated group list {
          optional int32 element;
        }
      }
    }
  }
  optional group list_list_list_int (LIST) {
    repeated group list {
      optional group element (LIST) {
        repeated group list {
          required group element (LIST) {
            repeated group list {
              optional int32 element;
            }
          }
        }
      }
    }
  }
  optional group list_str (LIST) {
    repeated group list {
      optional binary element (UTF8);
    }
  }
  optional group list_list_str (LIST) {
    repeated group list {
      optional group element (LIST) {
        repeated group list {
          optional binary element (UTF8);
        }
      }
    }
  }
*/

// Generates a vector of uniform random values of type T
template <typename T>
inline auto random_values(size_t size)
{
  std::vector<T> values(size);

  using T1 = T;
  using uniform_distribution =
    typename std::conditional_t<std::is_same<T1, bool>::value,
                                std::bernoulli_distribution,
                                std::conditional_t<std::is_floating_point<T1>::value,
                                                   std::uniform_real_distribution<T1>,
                                                   std::uniform_int_distribution<T1>>>;

  static constexpr auto seed = 0xf00d;
  static std::mt19937 engine{seed};
  static uniform_distribution dist{};
  std::generate_n(values.begin(), size, [&]() { return T{dist(engine)}; });

  return values;
}

template <typename T>
  using column_wrapper = typename std::conditional<std::is_same<T, cudf::string_view>::value,
                                                 cudf::test::strings_column_wrapper,
                                                 cudf::test::fixed_width_column_wrapper<T>>::type;


void parquet_test()
{  
  namespace cudf_io = cudf::io;  
  using TypeParam = int;

  {    
    auto valids = cudf::test::make_counting_transform_iterator( 0, [](auto i) { return i % 2 == 0 ? false : true; } );    
    cudf::test::fixed_width_column_wrapper<int> col{ {1, 2, 3, 4, 5, 6}, valids };

    std::vector<std::unique_ptr<column>> cols;
    cols.push_back(col.release());
    auto expected = std::make_unique<table>(std::move(cols));
    EXPECT_EQ(1, expected->num_columns());

    std::string filepath("SingleColumn.parquet");    
    cudf_io::parquet_writer_options out_args = cudf_io::parquet_writer_options::builder(cudf_io::sink_info{filepath}, expected->view());
    cudf_io::write_parquet(out_args);
  }
  
  {
    cudf_io::parquet_reader_options in_args = cudf_io::parquet_reader_options::builder(cudf_io::source_info{"SingleColumn.parquet"});
    auto result = cudf_io::read_parquet(in_args);    
    cudf::test::print(result.tbl->view().column(0));  

    //printf("Num rows : %d\n", result.tbl->view().column(0).size());

    //cudf::test::expect_tables_equal(expected->view(), result.tbl->view());    
  }


/*
 {
    auto sequence =
      cudf::test::make_counting_transform_iterator(0, [](auto i) { return TypeParam(i); });
    auto validity = cudf::test::make_counting_transform_iterator(0, [](auto i) { return true; });

    constexpr auto num_rows = 100000;
    column_wrapper<TypeParam> col(sequence, sequence + num_rows, validity);

    std::vector<std::unique_ptr<column>> cols;
    cols.push_back(col.release());
    auto expected = std::make_unique<table>(std::move(cols));
    EXPECT_EQ(1, expected->num_columns());

    std::string filepath("skippedrows.parquet");
    cudf_io::write_parquet_args out_args{cudf_io::sink_info{filepath}, expected->view()};
    cudf_io::write_parquet(out_args);

    cudf_io::read_parquet_args in_args{cudf_io::source_info{filepath}};
    auto result = cudf_io::read_parquet(in_args);

    cudf::test::print(result.tbl->view().column(0));

    cudf::test::expect_tables_equal(expected->view(), result.tbl->view());
  }
  */
 /*
 {    
    std::string filepath("row_group.parquet");  
    cudf_io::read_parquet_args in_args{cudf_io::source_info{filepath}};    
    in_args.skip_rows = 37;
    in_args.num_rows = 25;
    auto result = cudf_io::read_parquet(in_args);

    for(int idx=0; idx<result.tbl->view().num_columns(); idx++){
      cudf::test::print(result.tbl->view().column(idx));
    }

    int whee = 10;
    whee++;
 }
 */
}

void parquet_old_test()
{
  namespace cudf_io = cudf::io;
    
  using column         = cudf::column;
  using table          = cudf::table;
  using table_view     = cudf::table_view;
  using TypeParam = int;
  
  /*
  auto sequence =
    cudf::test::make_counting_transform_iterator(0, [](auto i) { return TypeParam(i); });
  auto validity = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i%2 == 0 ? false : true; });
  constexpr auto num_rows = 50;

  column_wrapper<TypeParam> col0(sequence, sequence + num_rows, validity);  
  std::vector<std::unique_ptr<column>> cols0;
  cols0.push_back(col0.release());
  auto tbl0 = std::make_unique<table>(std::move(cols0));  
    
  column_wrapper<TypeParam> col1(sequence, sequence + num_rows, validity);  
  std::vector<std::unique_ptr<column>> cols1;
  cols1.push_back(col1.release());
  auto tbl1 = std::make_unique<table>(std::move(cols1));  

  auto full_table = cudf::concatenate({*tbl0, *tbl1});

  std::string filepath("SingleColumn.parquet");
  cudf_io::write_parquet_chunked_args args{cudf_io::sink_info{filepath}};
  auto state = cudf_io::write_parquet_chunked_begin(args);
  cudf_io::write_parquet_chunked(*tbl0, state);
  cudf_io::write_parquet_chunked(*tbl1, state);
  cudf_io::write_parquet_chunked_end(state);
  
  //cudf_io::write_parquet_args out_args{cudf_io::sink_info{filepath}, expected->view()};
  //cudf_io::write_parquet(out_args);

  cudf_io::read_parquet_args in_args{cudf_io::source_info{filepath}};
  auto result = cudf_io::read_parquet(in_args);

  cudf::test::print(result.tbl->get_column(0));  
  printf("\n");
  cudf::test::print(full_table->get_column(0).view());
  */

  srand(31337);
  auto table1 = create_random_fixed_table<int>(5, 5, true);
  auto table2 = create_random_fixed_table<int>(5, 5, true);

  auto full_table = cudf::concatenate({*table1, *table2});

  std::string filepath("ChunkedSimple.parquet");
  cudf_io::chunked_parquet_writer_options args = cudf_io::chunked_parquet_writer_options::builder(cudf_io::sink_info{filepath});
  auto state = cudf_io::write_parquet_chunked_begin(args);  
  cudf_io::write_parquet_chunked(*table1, state);
  cudf_io::write_parquet_chunked(*table2, state);
  cudf_io::write_parquet_chunked_end(state);

  cudf_io::parquet_reader_options read_args = cudf_io::parquet_reader_options::builder(cudf_io::source_info{filepath});
  auto result = cudf_io::read_parquet(read_args);

  //expect_tables_equal(*result.tbl, *full_table);
  cudf::test::print(result.tbl->get_column(0));  
  printf("\n");
  cudf::test::print(full_table->get_column(0).view());
}
/*
template<typename T>
class empty_list : public cudf::test::lists_column_wrapper<T> {
public:
   empty_list() : cudf::test::lists_column_wrapper<T>(
     cudf::test::lists_column_wrapper<T>::list_type::EMPTY) {}
};
*/

void parquet_test_test()
{
  namespace cudf_io = cudf::io;
    
  using column         = cudf::column;
  using table          = cudf::table;
  using table_view     = cudf::table_view;
  using TypeParam = int;
  { 
    cudf_io::parquet_reader_options read_args = cudf_io::parquet_reader_options::builder(cudf_io::source_info{"test_parquet_reader_list_num_rows.parquet"});
    auto result = cudf_io::read_parquet(read_args);    
    cudf::test::print(result.tbl->get_column(0));
  }

  auto expected = make_parquet_col<int64_t>(0, 4, 10, 10, true);
  //cudf::test::print(*expected);
      
  // cudf_io::read_parquet_args read_args{cudf_io::source_info{"test_parquet_writer_list_large_table.parquet"}};  
  cudf_io::parquet_reader_options read_args = cudf_io::parquet_reader_options::builder(cudf_io::source_info{"test_parquet_reader_list_validity.parquet"});
  // cudf_io::read_parquet_args read_args{cudf_io::source_info{"lists_big2.parquet"}};
  auto result = cudf_io::read_parquet(read_args);    
  cudf::test::print(result.tbl->get_column(0));

{
      column_view offsets_a = expected->child(0);
      column_view offsets_b = result.tbl->get_column(0).child(0);
      cudf::test::print(offsets_a);
      printf("\n\n\n");    
      cudf::test::print(offsets_b);
      cudf::test::expect_columns_equal(offsets_a, offsets_b);
      printf("offsets level 0 size : %d\n", offsets_a.size());
    }

    //cudf::test::print(result.tbl->get_column(0));

    {
      column_view offsets_a = expected->child(1).child(0);
      column_view offsets_b = result.tbl->get_column(0).child(1).child(0);
      //cudf::test::print(offsets_a);
      //printf("\n\n\n");    
      //cudf::test::print(offsets_b);
      cudf::test::expect_columns_equal(offsets_a, offsets_b);
      printf("offsets level 1 size : %d\n", offsets_b.size());
    }

    {
      column_view leaf_a = expected->child(1).child(1);  
      column_view leaf_b = result.tbl->get_column(0).child(1).child(1);        
      cudf::test::print(leaf_a);
      printf("\n\n\n");    
      cudf::test::print(leaf_b);
      cudf::test::expect_columns_equal(leaf_a, leaf_b);
    }

  ///cudf::test::print(result.tbl->get_column(0));
  // cudf::test::expect_columns_equal(*expected, result.tbl->get_column(0));
  //cudf::test::print(result.tbl->get_column(1));
  //printf("Type : %d (nullable : %s)\n", result.tbl->get_column(2).type().id(), result.tbl->get_column(2).nullable() ? "yes" : "no");
  //cudf::test::print(result.tbl->get_column(2));
}

template<typename T>
class test_class {
public:
  test_class(){}  
};

}}

void parquet_struct_test()
{
  namespace cudf_io = cudf::io;

  /*
  auto first_name = cudf::test::strings_column_wrapper{"James", "Michael", "Robert", "Maria", "Jen"};
  auto middle_name = cudf::test::strings_column_wrapper{"", "Rose", "", "Anne", "Mary"};
  auto last_name = cudf::test::strings_column_wrapper{"Smith", "", "Williams", "Jones", "Brown"};
  auto struct_col = cudf::test::structs_column_wrapper{{first_name, middle_name, last_name}};
  cudf::test::print(struct_col);    

  {
    using LCW = cudf::test::lists_column_wrapper<int>;
    #define EL {LCW{}}

    bool valids1[] = { true, false };
    bool valids2[] = { true, false, true };
    bool valids3[] = { true, false, true, true, true };
    cudf::test::lists_column_wrapper<int> list{ {{{1, 2}, {3, 4}}, 
                                                 EL, 
                                                 {{{5, 6}, EL}, valids1},
                                                 {{LCW{1}}},
                                                 {LCW{5}, {{6, 0, 8}, valids2}}
                                                 }, valids3 };
    cudf::test::print(list);
  }
  */ 

  // cudf_io::read_parquet_args read_args{cudf_io::source_info{"parquet_struct_test_nested.parquet"}};  
  // cudf_io::read_parquet_args read_args{cudf_io::source_info{"parquet_struct_test_nested_nulls.parquet"}};    
  // cudf_io::read_parquet_args read_args{cudf_io::source_info{"ints_with_null.parquet"}};  
  // cudf_io::read_parquet_args read_args{cudf_io::source_info{"SingleColumnWithNulls.parquet"}};    
  // cudf_io::read_parquet_args read_args{cudf_io::source_info{"list_int.parquet"}};     

  {
    // cudf_io::parquet_reader_options read_args = cudf_io::parquet_reader_options::builder(cudf_io::source_info{"parquet_list_of_structs.parquet"});
    cudf_io::parquet_reader_options read_args = cudf_io::parquet_reader_options::builder(cudf_io::source_info{"meit1.parquet"});
    // read_args.set_columns({"tr_fm"});
    read_args.set_columns({"tr_fm"});
    auto result = cudf_io::read_parquet(read_args);
    print_names(result.metadata.schema_info);
    for(int idx=0; idx<result.tbl->view().num_columns(); idx++){
      cudf::test::print(result.tbl->view().column(idx));
    }
  } 
        
  {
    // cudf_io::parquet_reader_options read_args = cudf_io::parquet_reader_options::builder(cudf_io::source_info{"parquet_list_of_structs.parquet"});
    cudf_io::parquet_reader_options read_args = cudf_io::parquet_reader_options::builder(cudf_io::source_info{"list_int.parquet"});
    auto result = cudf_io::read_parquet(read_args);
    print_names(result.metadata.schema_info);
    for(int idx=0; idx<result.tbl->view().num_columns(); idx++){
      cudf::test::print(result.tbl->view().column(idx));
    }
  }  

  {
    cudf_io::parquet_reader_options read_args = cudf_io::parquet_reader_options::builder(cudf_io::source_info{"parquet_struct_test.parquet"});
    //read_args.set_columns({"name"});
    auto result = cudf_io::read_parquet(read_args);
    print_names(result.metadata.schema_info);
    for(int idx=0; idx<result.tbl->view().num_columns(); idx++){
      cudf::test::print(result.tbl->view().column(idx));
    }
  }    
 
  {
    cudf_io::parquet_reader_options read_args = cudf_io::parquet_reader_options::builder(cudf_io::source_info{"parquet_struct_test_nulls.parquet"});
    //read_args.set_columns({"name"});
    auto result = cudf_io::read_parquet(read_args);
    print_names(result.metadata.schema_info);
    for(int idx=0; idx<result.tbl->view().num_columns(); idx++){
      cudf::test::print(result.tbl->view().column(idx));
    }
  }

  {
    cudf_io::parquet_reader_options read_args = cudf_io::parquet_reader_options::builder(cudf_io::source_info{"parquet_struct_test_nulls2.parquet"});
    auto result = cudf_io::read_parquet(read_args);
    print_names(result.metadata.schema_info);
    for(int idx=0; idx<result.tbl->view().num_columns(); idx++){
      cudf::test::print(result.tbl->view().column(idx));
    }
  }  
  {
    cudf_io::parquet_reader_options read_args = cudf_io::parquet_reader_options::builder(cudf_io::source_info{"parquet_struct_test_nested.parquet"}); 
    auto result = cudf_io::read_parquet(read_args);
    print_names(result.metadata.schema_info);
    for(int idx=0; idx<result.tbl->view().num_columns(); idx++){
      cudf::test::print(result.tbl->view().column(idx));
    }
  }    
  { 
    cudf_io::parquet_reader_options read_args = cudf_io::parquet_reader_options::builder(cudf_io::source_info{"parquet_list_of_structs.parquet"}); 
    auto result = cudf_io::read_parquet(read_args);
    print_names(result.metadata.schema_info);
    for(int idx=0; idx<result.tbl->view().num_columns(); idx++){
      cudf::test::print(result.tbl->view().column(idx));
    }
  }

 { 
    cudf_io::parquet_reader_options read_args = cudf_io::parquet_reader_options::builder(cudf_io::source_info{"parquet_map.parquet"}); 
    auto result = cudf_io::read_parquet(read_args);
    print_names(result.metadata.schema_info);
    for(int idx=0; idx<result.tbl->view().num_columns(); idx++){
      cudf::test::print(result.tbl->view().column(idx));
    }
  }
  {     
    cudf_io::parquet_reader_options read_args = cudf_io::parquet_reader_options::builder(cudf_io::source_info{"parquet_struct_test_nested_nulls3.parquet"});     
    auto result = cudf_io::read_parquet(read_args);
    print_names(result.metadata.schema_info);
    for(int idx=0; idx<result.tbl->view().num_columns(); idx++){
      cudf::test::print(result.tbl->view().column(idx));
    }
  }  
  {     
    cudf_io::parquet_reader_options read_args = cudf_io::parquet_reader_options::builder(cudf_io::source_info{"parquet_complex.parquet"});     
    auto result = cudf_io::read_parquet(read_args);
    print_names(result.metadata.schema_info);
    for(int idx=0; idx<result.tbl->view().num_columns(); idx++){
      cudf::test::print(result.tbl->view().column(idx));
    }
  }   
}

// { {0, 1, 2, 3, 4, 5}, {-10, -20, -30, -40, -50, -60, -70, -80, -90, -100}, {}, {NULL, 4, NULL, -100, NULL, 7, NULL}, {NULL}, NULL }
void parquet_list_test()
{
  namespace cudf_io = cudf::io;
  
  using LCW = cudf::test::lists_column_wrapper<int>;
  using SLCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  #define EL {LCW{}}

  
  /*
  {
    std::vector<bool> v{true, false, true, true, true};
    auto valids = cudf::test::make_counting_transform_iterator( 0, [&v](auto i) { return v[i]; } );

    cudf::test::fixed_width_column_wrapper<int> col{ {10, 11, 12, 13, 14}, valids };

    std::vector<std::unique_ptr<column>> cols;
    cols.push_back(col.release());  
    auto expected = std::make_unique<table>(std::move(cols));
      
    cudf_io::write_parquet_args args{cudf_io::sink_info{"ints_with_null.parquet"}, *expected};
    cudf_io::write_parquet(args);

    cudf_io::read_parquet_args read_args{cudf_io::source_info{"ints_with_null.parquet"}};
    auto result = cudf_io::read_parquet(read_args);

    cudf::test::print(expected->get_column(0));
    cudf::test::print(result.tbl->get_column(0));
  }
  */

  int skip_rows = 0;

     
  // list_int, column 0     
  auto valids = cudf::test::make_counting_transform_iterator( 0, [](auto i) { return i % 2 == 0 ? false : true; } );    
  std::vector<bool> v2{true,true, true, true, true, false};
  auto valids2 = cudf::test::make_counting_transform_iterator( 0, [&v2, skip_rows](auto i) { return v2[i + skip_rows]; } );   
  cudf::test::lists_column_wrapper<int> list0 { {
                                                {0, 1, 2, 3, 4, 5}, 
                                                {-10, -20, -30, -40, -50, -60, -70, -80, -90, -100}, 
                                                EL,
                                                {{99, 4, 99, -100, 99, 7, 99}, valids}, 
                                                {{99}, valids}, 
                                                EL
                                                }, valids2 };

  // list_list_int, column 0 (same as above)

  #define EL {LCW{}}
  #define ESL {SLCW{}}

  // list_list_int, column 1
  std::vector<bool> _v1{false};
  auto _valids1 = cudf::test::make_counting_transform_iterator( 0, [&_v1](auto i) { return _v1[i]; } );
  std::vector<bool> _v2{false, true, true, true, true, true};
  auto _valids2 = cudf::test::make_counting_transform_iterator( 0, [&_v2, skip_rows](auto i) { return _v2[i + skip_rows]; } );
  std::vector<bool> _v3{true, true, true, false, true};
  auto _valids3 = cudf::test::make_counting_transform_iterator( 0, [&_v3](auto i) { return _v3[i]; } );
  std::vector<bool> _v4{false, false, false};
  auto _valids4 = cudf::test::make_counting_transform_iterator( 0, [&_v4](auto i) { return _v4[i]; } );
  std::vector<bool> _v5{true, true, false, true, false, true};
  auto _valids5 = cudf::test::make_counting_transform_iterator( 0, [&_v5](auto i) { return _v5[i]; } );
  std::vector<bool> _v6{false, true, true, true};
  auto _valids6 = cudf::test::make_counting_transform_iterator( 0, [&_v6](auto i) { return _v6[i]; } );
  cudf::test::lists_column_wrapper<int> list1{{ 
                                                {EL},
                                                {{EL}, _valids1},
                                                { {{99}, _valids1} },                                                  
                                                { { {1}, EL, EL, {{-10, -20, -30, 99, -50}, _valids3}, EL, {{99, 99, 99}, _valids4} }, _valids5 },
                                                {{1, 2, 3}, {10, 7, 5, 3, 1, 7}},
                                                { { EL, {-10}, EL, {-60, 50, 20} }, _valids6 }
                                              }, _valids2};               
  // list_list_list_int, column 2
  std::vector<bool> __v1{false};
  auto __valids1 = cudf::test::make_counting_transform_iterator( 0, [&__v1](auto i) { return __v1[i]; } );    
  std::vector<bool> __v2{true, false};    
  auto __valids2 = cudf::test::make_counting_transform_iterator( 0, [&__v2](auto i) { return __v2[i]; } );
  std::vector<bool> __v3{true, true, false};    
  auto __valids3 = cudf::test::make_counting_transform_iterator( 0, [&__v3](auto i) { return __v3[i]; } );
  std::vector<bool> __v4{true, true, true, false};    
  auto __valids4 = cudf::test::make_counting_transform_iterator( 0, [&__v4](auto i) { return __v4[i]; } );
  std::vector<bool> __v5{false, true, true, true};
  auto __valids5 = cudf::test::make_counting_transform_iterator( 0, [&__v5](auto i) { return __v5[i]; } );
  std::vector<bool> __v6{false, true, true, true, true, false};
  auto __valids6 = cudf::test::make_counting_transform_iterator( 0, [&__v6](auto i) { return __v6[i]; } );
  std::vector<bool> __v7{true, true, true, true, true, true, true, false, false};
  auto __valids7 = cudf::test::make_counting_transform_iterator( 0, [&__v7](auto i) { return __v7[i]; } );
  std::vector<bool> __v8{true, true, false, true, true};
  auto __valids8 = cudf::test::make_counting_transform_iterator( 0, [&__v8](auto i) { return __v8[i]; } );
  std::vector<bool> __v9{true, false, true, true, false, true};
  auto __valids9 = cudf::test::make_counting_transform_iterator( 0, [&__v9](auto i) { return __v9[i]; } );
  std::vector<bool> __v10{false, true};
  auto __valids10 = cudf::test::make_counting_transform_iterator( 0, [&__v10](auto i) { return __v10[i]; } );    
   
  cudf::test::lists_column_wrapper<int> list2{                                                 
                                                { { {{99}, __valids1}} },
                                                { { {EL}, __valids1 } },                                                
                                                { {{EL}}, __valids1 },
                                                
                                                {  {EL, {{1, 99}, __valids2}},
                                                    {EL},
                                                    {{EL, {1}, {99}}, __valids3},
                                                    {{{99}, EL, {2}, {{3, 1, 7, 99}, __valids4}}, __valids5} },
                                                                                                
                                                {{ {{EL, EL, {{99, 1}, __valids10}, {3, 2, 1}, EL, EL}, __valids6},
                                                    EL,
                                                    EL,
                                                    EL,
                                                    EL,
                                                    {{EL, EL, EL, {{3, 2, 1, 0, 3, 3, 3, 99, 99}, __valids7}, EL}, __valids8}},
                                                                                                                __valids9},
                                                { {EL, {{1, 99}, __valids2}},
                                                  {EL},
                                                  {{EL, {1}, EL}, __valids3},
                                                  {{EL, EL, {2}, {{3, 1, 7, 99}, __valids4}}, __valids5},
                                                  {{1, 2, 3}, {10, 7, 5, 3, 1, 7}} }
                                            };
                                            
                                            
                                            /*                
  // list_list_list_int (w/required field, column 2
  std::vector<bool> __v1{false};
  auto __valids1 = cudf::test::make_counting_transform_iterator( 0, [&__v1](auto i) { return __v1[i]; } );    
  std::vector<bool> __v2{true, false};    
  auto __valids2 = cudf::test::make_counting_transform_iterator( 0, [&__v2](auto i) { return __v2[i]; } );
  std::vector<bool> __v3{true, true, false};    
  auto __valids3 = cudf::test::make_counting_transform_iterator( 0, [&__v3](auto i) { return __v3[i]; } );
  std::vector<bool> __v4{true, true, true, false};    
  auto __valids4 = cudf::test::make_counting_transform_iterator( 0, [&__v4](auto i) { return __v4[i]; } );
  std::vector<bool> __v5{false, true, true, true};
  auto __valids5 = cudf::test::make_counting_transform_iterator( 0, [&__v5](auto i) { return __v5[i]; } );
  std::vector<bool> __v6{false, true, true, true, true, false};
  auto __valids6 = cudf::test::make_counting_transform_iterator( 0, [&__v6](auto i) { return __v6[i]; } );
  std::vector<bool> __v7{true, true, true, true, true, true, true, false, false};
  auto __valids7 = cudf::test::make_counting_transform_iterator( 0, [&__v7](auto i) { return __v7[i]; } );
  std::vector<bool> __v8{true, true, false, true, true};
  auto __valids8 = cudf::test::make_counting_transform_iterator( 0, [&__v8](auto i) { return __v8[i]; } );
  std::vector<bool> __v9{true, false, true, true, false, true};
  auto __valids9 = cudf::test::make_counting_transform_iterator( 0, [&__v9](auto i) { return __v9[i]; } );
  std::vector<bool> __v10{false, true};
  auto __valids10 = cudf::test::make_counting_transform_iterator( 0, [&__v10](auto i) { return __v10[i]; } );    
  cudf::test::lists_column_wrapper<int> list2{                                                 
                                                {{{ {99}, __valids1 }}}, 
                                                
                                                {EL},

                                                { {EL}, __valids1 },

                                                { {EL, {{1, 99}, __valids2}},
                                                  {EL},
                                                  {EL, {1}, EL},
                                                  {EL, EL, {2}, {{3, 1, 7, 99}, __valids4}} },
                                                
                                                { {{EL, EL, {{99, 1}, __valids10}, {3, 2, 1}, EL, EL},
                                                   {99},
                                                   EL,
                                                   EL,
                                                   {99},
                                                   {EL, EL, {0}, {{3, 2, 1, 0, 3, 3, 3, 99, 99}, __valids7}, EL}},
                                                                                                                __valids9},
                                                                                                                
                                                { {EL, {{1, 99}, __valids2}},
                                                  {EL},
                                                  {EL, {1}, {0}},
                                                  {{0}, EL, {2}, {{3, 1, 7, 99}, __valids4}},
                                                  {{1, 2, 3}, {10, 7, 5, 3, 1, 7}} }
                                            };
                                            */
                                            

  // list_string, column 3 
  std::vector<bool> ___v1{false};
  auto ___valids1 = cudf::test::make_counting_transform_iterator( 0, [&___v1](auto i) { return ___v1[i]; } );    
  std::vector<bool> ___v2{true, false, true, true, true, true};
  auto ___valids2 = cudf::test::make_counting_transform_iterator( 0, [&___v2](auto i) { return ___v2[i]; } );
  std::vector<bool> ___v3{false, true, false, true, true, false, true, true, true};
  auto ___valids3 = cudf::test::make_counting_transform_iterator( 0, [&___v3](auto i) { return ___v3[i]; } );
  std::vector<bool> ___v4{false, true, false, true, true, false, true, true, true, true, false, true, true, true, true};
  auto ___valids4 = cudf::test::make_counting_transform_iterator( 0, [&___v4](auto i) { return ___v4[i]; } );
  std::vector<bool> ___v5{false, true, true, true, true, true };
  auto ___valids5 = cudf::test::make_counting_transform_iterator( 0, [&___v5, skip_rows](auto i) { return ___v5[i + skip_rows]; } );
  cudf::test::lists_column_wrapper<cudf::string_view> list3 { {
                                                        ESL,
                                                        {{"99"}, ___valids1},
                                                        SLCW{""},
                                                        {{"", "99", "abc", "123", "xyz", "pdq"}, ___valids2},
                                                        {{"99", "", "99", "", "", "99", "fizgig", "abracadabra", "wut"}, ___valids3},
                                                        {{"99", "", "99", "", "", "99", "fizgig", "abracadabra", "wut", "", "99", "abc", "123", "xyz", "pdq"}, ___valids4}
                                                        }, ___valids5};

  // list_list_string, column 4  
  std::vector<bool> ____v1{false};
  auto ____valids1 = cudf::test::make_counting_transform_iterator( 0, [&____v1](auto i) { return ____v1[i]; } );    
  std::vector<bool> ____v2{true, false, true, true, true, true};
  auto ____valids2 = cudf::test::make_counting_transform_iterator( 0, [&____v2](auto i) { return ____v2[i]; } );
  std::vector<bool> ____v3{false, true, true, true, true, false, true};
  auto ____valids3 = cudf::test::make_counting_transform_iterator( 0, [&____v3](auto i) { return ____v3[i]; } );
  std::vector<bool> ____v4{true, false, true, true};
  auto ____valids4 = cudf::test::make_counting_transform_iterator( 0, [&____v4](auto i) { return ____v4[i]; } );
  std::vector<bool> ____v5{false, true, false, true, true, false, true, true, true};
  auto ____valids5 = cudf::test::make_counting_transform_iterator( 0, [&____v5](auto i) { return ____v5[i]; } );
  std::vector<bool> ____v6{true, false, true, true, true, true};
  auto ____valids6 = cudf::test::make_counting_transform_iterator( 0, [&____v6](auto i) { return ____v6[i]; } );
  std::vector<bool> ____v7{false, true, true,false, true, true};
  auto ____valids7 = cudf::test::make_counting_transform_iterator( 0, [____v7](auto i) { return ____v7[i]; } );
  std::vector<bool> ____v8{false, true, true, true, true, false, true};
  auto ____valids8 = cudf::test::make_counting_transform_iterator( 0, [&____v8](auto i) { return ____v8[i]; } );
  std::vector<bool> ____v9{false, true, true, false, true, true, true, false, true, true};
  auto ____valids9 = cudf::test::make_counting_transform_iterator( 0, [&____v9](auto i) { return ____v9[i]; } );
  std::vector<bool> ____v10{false, true, true, true, true, true};
  auto ____valids10 = cudf::test::make_counting_transform_iterator( 0, [&____v10, skip_rows](auto i) { return ____v10[i + skip_rows]; } );
  cudf::test::lists_column_wrapper<cudf::string_view> list4 { {
                                                        ESL,
                                                        {{ESL}, ____valids1},
                                                        {{{"99"}, ____valids1}},                                                        
                                                        {{
                                                          ESL, 
                                                          {"99"}, 
                                                          {{"", "99", "abc", "123", "xyz", "pdq"}, ____valids2},
                                                          {{"99", "", "woot", "foo", "bar", "99", ""}, ____valids3},
                                                        }, ____valids4},
                                                        {{
                                                          {"99"},
                                                          ESL,
                                                          ESL,
                                                          {"99"},
                                                          {{"99", "", "99", "", "", "99", "fizgig", "abracadabra", "wut"}, ____valids5},
                                                          {{"", "99", "abc", "123", "xyz", "pdq"}, ____valids6}
                                                        }, ____valids7},                                                        
                                                        {{
                                                          {"99"},
                                                          ESL,
                                                          ESL,
                                                          {"99"},
                                                          {{"99", "", "99", "", "", "99", "fizgig", "abracadabra", "wut"}, ____valids5},
                                                          {{"", "99", "abc", "123", "xyz", "pdq"}, ____valids6},
                                                          ESL,
                                                          {"99"},
                                                          {{"", "99", "abc", "123", "xyz", "pdq"}, ____valids6},
                                                          {{"99", "", "woot", "foo", "bar", "99", ""}, ____valids8}
                                                        }, ____valids9}
                                                        
                                                        }, ____valids10};  

  //test::print(list0);  
  //test::print(list1);  
  //test::print(list2);  
  //test::print(list3);  
  //test::print(list4);
   
   //cudf_io::read_parquet_args read_args{cudf_io::source_info{"nullstrings.parquet"}};   
   // cudf_io::parquet_reader_options read_args = cudf::io::parquet_reader_options::builder(cudf_io::source_info{"list_int.parquet"});    
   // cudf_io::read_parquet_args read_args{cudf_io::source_info{"ListColumn.parquet"}};
   //read_args.skip_rows = 1;
   //read_args.num_rows = 1;   
   cudf_io::parquet_reader_options read_args = cudf::io::parquet_reader_options::builder(cudf_io::source_info{"list_types.parquet"});
   // cudf_io::read_parquet_args read_args{cudf_io::source_info{"testlists.parquet"}};   
   //cudf_io::read_parquet_args read_args{cudf_io::source_info{"listy.parquet"}};   
   //read_args.skip_rows = skip_rows;
   //read_args.num_rows = 10;
   //read_args.set_columns({"list_int"});
   auto result = cudf_io::read_parquet(read_args);       
   print_names(result.metadata.schema_info);
   printf("\n\n\n");
                                
  cudf::test::print(list0);  
  cudf::test::print(result.tbl->get_column(0));
  cudf::test::expect_columns_equal(list0, result.tbl->get_column(0));

  cudf::test::print(list1);  
  cudf::test::print(result.tbl->get_column(1));
  cudf::test::expect_columns_equal(list1, result.tbl->get_column(1));

  cudf::test::print(list2);  
  cudf::test::print(result.tbl->get_column(2));
  cudf::test::expect_columns_equal(list2, result.tbl->get_column(2));

  cudf::test::print(list3);
  cudf::test::print(result.tbl->get_column(3));
  cudf::test::expect_columns_equal(list3, result.tbl->get_column(3));

  cudf::test::print(list4);
  cudf::test::print(result.tbl->get_column(4));
  cudf::test::expect_columns_equal(list4, result.tbl->get_column(4));

  int whee = 10;
  whee++;
}

void parquet_crash_test()
{
  namespace cudf_io = cudf::io;

  std::string filename("concatenated_yellow_tripdata_2015-01_08_head_40M.parquet");
  
  // std::shared_ptr<arrow::io::RandomAccessFile> file = std::make_shared<arrow::io::RandomAccessFile>(filename);
  // file->Open(filename);
  //std::shared_ptr<arrow::io::RandomAccessFile> file = arrow::io::ReadableFile::Open(filename);
  std::shared_ptr<arrow::io::RandomAccessFile> file = arrow::io::ReadableFile::Open(filename).ValueOrDie();
  auto arrow_source = cudf::io::arrow_io_source{file};  
  cudf_io::parquet_reader_options pq_args = cudf::io::parquet_reader_options::builder(cudf_io::source_info{&arrow_source});

  // cudf_io::parquet_reader_options pq_args = cudf_io::parquet_reader_options::builder(cudf_io::source_info{filename});

  pq_args.enable_convert_strings_to_categories(false);
  pq_args.enable_use_pandas_metadata(false);
  pq_args.set_num_rows(1);
  pq_args.set_columns({"store_and_fwd_flag"});
  cudf_io::table_with_metadata table_out = cudf::io::read_parquet(pq_args);
  /*
  int count = 0;
  while(1){
    cudf_io::table_with_metadata table_out = cudf::io::read_parquet(pq_args);
    if(count % 10 == 0){
      printf("Runs : %d\n", count);
    }
    count++;
  }
  */

  for(int idx=0; idx<table_out.tbl->num_columns(); idx++){
    cudf::test::print(table_out.tbl->get_column(idx));
  }
}

/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <rmm/device_uvector.hpp>
#include <thrust/iterator/discard_iterator.h>

#include <numeric>

#define __NEW_PATH

namespace cudf {
namespace detail {
namespace {
/**
 * @brief Copies contents of `in` to `out`.  Copies validity if present
 * but does not compute null count.
 *
 * @param in column_view to copy from
 * @param out mutable_column_view to copy to.
 */
template <size_type block_size, typename T, bool has_validity>
__launch_bounds__(block_size) __global__
  void copy_in_place_kernel(column_device_view const in, mutable_column_device_view out)
{
  const size_type tid            = threadIdx.x + blockIdx.x * block_size;
  const int warp_id              = tid / cudf::detail::warp_size;
  const size_type warps_per_grid = gridDim.x * block_size / cudf::detail::warp_size;

  // begin/end indices for the column data
  size_type begin = 0;
  size_type end   = in.size();
  // warp indices.  since 1 warp == 32 threads == sizeof(bit_mask_t) * 8,
  // each warp will process one (32 bit) of the validity mask via
  // __ballot_sync()
  size_type warp_begin = cudf::word_index(begin);
  size_type warp_end   = cudf::word_index(end - 1);

  // lane id within the current warp
  const int lane_id = threadIdx.x % cudf::detail::warp_size;

  // current warp.
  size_type warp_cur = warp_begin + warp_id;
  size_type index    = tid;
  while (warp_cur <= warp_end) {
    bool in_range = (index >= begin && index < end);

    bool valid = true;
    if (has_validity) { valid = in_range && in.is_valid(index); }
    if (in_range) { out.element<T>(index) = in.element<T>(index); }

    // update validity
    if (has_validity) {
      // the final validity mask for this warp
      int warp_mask = __ballot_sync(0xFFFF'FFFF, valid && in_range);
      // only one guy in the warp needs to update the mask and count
      if (lane_id == 0) { out.set_mask_word(warp_cur, warp_mask); }
    }

    // next grid
    warp_cur += warps_per_grid;
    index += block_size * gridDim.x;
  }
}

/**
 * @brief Copies contents of one string column to another.  Copies validity if present
 * but does not compute null count.
 *
 * The purpose of this kernel is to reduce the number of
 * kernel calls for copying a string column from 2 to 1, since number of kernel calls is the
 * dominant factor in large scale contiguous_split() calls.  To do this, the kernel is
 * invoked with using max(num_chars, num_offsets) threads and then doing separate
 * bounds checking on offset, chars and validity indices.
 *
 * Outgoing offset values are shifted down to account for the new base address
 * each column gets as a result of the contiguous_split() process.
 *
 * @param in num_strings number of strings (rows) in the column
 * @param in offsets_in pointer to incoming offsets to be copied
 * @param out offsets_out pointer to output offsets
 * @param in validity_in_offset offset into validity buffer to add to element indices
 * @param in validity_in pointer to incoming validity vector to be copied
 * @param out validity_out pointer to output validity vector
 * @param in offset_shift value to shift copied offsets down by
 * @param in num_chars number of chars to copy
 * @param in chars_in input chars to be copied
 * @param out chars_out output chars to be copied.
 */
template <size_type block_size, bool has_validity>
__launch_bounds__(block_size) __global__
  void copy_in_place_strings_kernel(size_type num_strings,
                                    size_type const* __restrict__ offsets_in,
                                    size_type* __restrict__ offsets_out,
                                    size_type validity_in_offset,
                                    bitmask_type const* __restrict__ validity_in,
                                    bitmask_type* __restrict__ validity_out,
                                    size_type offset_shift,
                                    size_type num_chars,
                                    char const* __restrict__ chars_in,
                                    char* __restrict__ chars_out)
{
  const size_type tid            = threadIdx.x + blockIdx.x * block_size;
  const int warp_id              = tid / cudf::detail::warp_size;
  const size_type warps_per_grid = gridDim.x * block_size / cudf::detail::warp_size;

  // how many warps we'll be processing. with strings, the chars and offsets
  // lengths may be different.  so we'll just march the worst case.
  size_type warp_begin = cudf::word_index(0);
  size_type warp_end   = cudf::word_index(std::max(num_chars, num_strings + 1) - 1);

  // end indices for chars
  size_type chars_end = num_chars;
  // end indices for offsets
  size_type offsets_end = num_strings + 1;
  // end indices for validity and the last warp that actually should
  // be updated
  size_type validity_end      = num_strings;
  size_type validity_warp_end = cudf::word_index(num_strings - 1);

  // lane id within the current warp
  const int lane_id = threadIdx.x % cudf::detail::warp_size;

  size_type warp_cur = warp_begin + warp_id;
  size_type index    = tid;
  while (warp_cur <= warp_end) { 
    if (index < chars_end) { chars_out[index] = chars_in[index]; }

    if (index < offsets_end) {
      // each output column starts at a new base pointer. so we have to
      // shift every offset down by the point (in chars) at which it was split.
      offsets_out[index] = offsets_in[index] - offset_shift;
    }

    // if we're still in range of validity at all
    if (has_validity && warp_cur <= validity_warp_end) {
      bool valid = (index < validity_end) && bit_is_set(validity_in, validity_in_offset + index);

      // the final validity mask for this warp
      int warp_mask = __ballot_sync(0xFFFF'FFFF, valid);
      // only one guy in the warp needs to update the mask and count
      if (lane_id == 0) { validity_out[warp_cur] = warp_mask; }
    }

    // next grid
    warp_cur += warps_per_grid;
    index += block_size * gridDim.x;
  }
}

// align all column size allocations to this boundary so that all output column buffers
// start at that alignment.
static constexpr size_t split_align = 64;

/**
 * @brief Information about the split for a given column. Bundled together
 *        into a struct because tuples were getting pretty unreadable.
 */
struct column_split_info {
  size_t data_buf_size;      // size of the data (including padding)
  size_t validity_buf_size;  // validity vector size (including padding)

  size_t offsets_buf_size;  // (strings only) size of offset column (including padding)
  size_type num_chars;      // (strings only) number of chars in the column
  size_type chars_offset;   // (strings only) offset from head of chars data
};

/**
 * @brief Functor called by the `type_dispatcher` to incrementally compute total
 * memory buffer size needed to allocate a contiguous copy of all columns within
 * a source table.
 */
struct column_buffer_size_functor {
  template <typename T>
  size_t operator()(column_view const& c, column_split_info& split_info)
  {
    split_info.data_buf_size = cudf::util::round_up_safe(c.size() * sizeof(T), split_align);
    split_info.validity_buf_size =
      (c.has_nulls() ? cudf::bitmask_allocation_size_bytes(c.size(), split_align) : 0);
    return split_info.data_buf_size + split_info.validity_buf_size;
  }
};
template <>
size_t column_buffer_size_functor::operator()<string_view>(column_view const& c,
                                                           column_split_info& split_info)
{
  // this has already been precomputed in an earlier step. return the sum.
  return split_info.data_buf_size + split_info.validity_buf_size + split_info.offsets_buf_size;
}

/**
 * @brief Functor called by the `type_dispatcher` to copy a column into a contiguous
 * buffer of output memory.
 *
 * Used for copying each column in a source table into one contiguous buffer of memory.
 */
struct column_copy_functor {
  template <typename T>
  void operator()(column_view const& in,
                  column_split_info const& split_info,
                  char*& dst,
                  std::vector<column_view>& out_cols)
  {
    // outgoing pointers
    char* data             = dst;
    bitmask_type* validity = split_info.validity_buf_size == 0
                               ? nullptr
                               : reinterpret_cast<bitmask_type*>(dst + split_info.data_buf_size);

    // increment working buffer
    dst += (split_info.data_buf_size + split_info.validity_buf_size);

    // no work to do
    if (in.size() == 0) {
      out_cols.push_back(column_view{in.type(), 0, nullptr});
      return;
    }

    // custom copy kernel (which could probably just be an in-place copy() function in cudf).
    cudf::size_type num_els  = cudf::util::round_up_safe(in.size(), cudf::detail::warp_size);
    constexpr int block_size = 256;
    cudf::detail::grid_1d grid{num_els, block_size, 1};

    // output copied column
    mutable_column_view mcv{in.type(), in.size(), data, validity, in.null_count()};
    if (in.has_nulls()) {
      copy_in_place_kernel<block_size, T, true><<<grid.num_blocks, block_size, 0, 0>>>(
        *column_device_view::create(in), *mutable_column_device_view::create(mcv));
    } else {
      copy_in_place_kernel<block_size, T, false><<<grid.num_blocks, block_size, 0, 0>>>(
        *column_device_view::create(in), *mutable_column_device_view::create(mcv));
    }

    out_cols.push_back(mcv);
  }
};
template <>
void column_copy_functor::operator()<string_view>(column_view const& in,
                                                  column_split_info const& split_info,
                                                  char*& dst,
                                                  std::vector<column_view>& out_cols)
{
  // outgoing pointers
  char* chars_buf            = dst;
  bitmask_type* validity_buf = split_info.validity_buf_size == 0
                                 ? nullptr
                                 : reinterpret_cast<bitmask_type*>(dst + split_info.data_buf_size);
  size_type* offsets_buf =
    reinterpret_cast<size_type*>(dst + split_info.data_buf_size + split_info.validity_buf_size);

  // increment working buffer
  dst += (split_info.data_buf_size + split_info.validity_buf_size + split_info.offsets_buf_size);

  // offsets column.
  strings_column_view strings_c(in);
  column_view in_offsets = strings_c.offsets();
  // note, incoming columns are sliced, so their size is fundamentally different from their child
  // offset columns, which are unsliced.
  size_type num_offsets = in.size() + 1;
  cudf::size_type num_threads =
    cudf::util::round_up_safe(std::max(split_info.num_chars, num_offsets), cudf::detail::warp_size);
  column_view in_chars = strings_c.chars();

  // a column with no strings will still have a single offset.
  CUDF_EXPECTS(num_offsets > 0, "Invalid offsets child column");

  // 1 combined kernel call that copies chars, offsets and validity in one pass. see notes on why
  // this exists in the kernel brief.
  constexpr int block_size = 256;
  cudf::detail::grid_1d grid{num_threads, block_size, 1};
  if (in.has_nulls()) {
    copy_in_place_strings_kernel<block_size, true><<<grid.num_blocks, block_size, 0, 0>>>(
      in.size(),                                        // num_rows
      in_offsets.head<size_type>() + in.offset(),       // offsets_in
      offsets_buf,                                      // offsets_out
      in.offset(),                                      // validity_in_offset
      in.null_mask(),                                   // validity_in
      validity_buf,                                     // validity_out
      split_info.chars_offset,                          // offset_shift
      split_info.num_chars,                             // num_chars
      in_chars.head<char>() + split_info.chars_offset,  // chars_in
      chars_buf);
  } else {
    copy_in_place_strings_kernel<block_size, false><<<grid.num_blocks, block_size, 0, 0>>>(
      in.size(),                                        // num_rows
      in_offsets.head<size_type>() + in.offset(),       // offsets_in
      offsets_buf,                                      // offsets_out
      0,                                                // validity_in_offset
      nullptr,                                          // validity_in
      nullptr,                                          // validity_out
      split_info.chars_offset,                          // offset_shift
      split_info.num_chars,                             // num_chars
      in_chars.head<char>() + split_info.chars_offset,  // chars_in
      chars_buf);
  }

  // output child columns
  column_view out_offsets{in_offsets.type(), num_offsets, offsets_buf};
  column_view out_chars{in_chars.type(), static_cast<size_type>(split_info.num_chars), chars_buf};

  // result
  out_cols.push_back(column_view(
    in.type(), in.size(), nullptr, validity_buf, in.null_count(), 0, {out_offsets, out_chars}));
}

/**
 * @brief Information about a string column in a table view.
 *
 * Used internally by preprocess_string_column_info as part of a device-accessible
 * vector for computing final string information in a single kernel call.
 */
struct column_preprocess_info {
  size_type index;
  size_type offset;
  size_type size;
  bool has_nulls;
  cudf::column_device_view offsets;
};

/**
 * @brief Preprocess information about all strings columns in a table view.
 *
 * In order to minimize how often we touch the gpu, we need to preprocess various pieces of
 * information about the string columns in a table as a batch process.  This function builds a list
 * of the offset columns for all input string columns and computes this information with a single
 * thrust call.  In addition, the vector returned is allocated for -all- columns in the table so
 * further processing of non-string columns can happen afterwards.
 *
 * The key things this function avoids
 * - avoiding reaching into gpu memory on the cpu to retrieve offsets to compute string sizes.
 * - creating column_device_views on the base string_column_view itself as that causes gpu memory
 * allocation.
 */
thrust::host_vector<column_split_info> preprocess_string_column_info(
  cudf::table_view const& t,
  rmm::device_vector<column_split_info>& device_split_info,
  cudaStream_t stream)
{
  // build a list of all the offset columns and their indices for all input string columns and put
  // them on the gpu
  thrust::host_vector<column_preprocess_info> offset_columns;
  offset_columns.reserve(t.num_columns());  // worst case

  // collect only string columns
  size_type column_index = 0;
  std::for_each(t.begin(), t.end(), [&offset_columns, &column_index](cudf::column_view const& c) {
    if (c.type().id() == type_id::STRING) {
      cudf::column_device_view cdv((strings_column_view(c)).offsets(), 0, 0);
      offset_columns.push_back(
        column_preprocess_info{column_index, c.offset(), c.size(), c.has_nulls(), cdv});
    }
    column_index++;
  });
  rmm::device_vector<column_preprocess_info> device_offset_columns = offset_columns;

  // compute column split information
  rmm::device_vector<thrust::pair<size_type, size_type>> device_offsets(t.num_columns());
  auto* offsets_p = device_offsets.data().get();
  thrust::for_each(rmm::exec_policy(stream)->on(stream),
                   device_offset_columns.begin(),
                   device_offset_columns.end(),
                   [offsets_p] __device__(column_preprocess_info const& cpi) {
                     offsets_p[cpi.index] =
                       thrust::make_pair(cpi.offsets.head<int32_t>()[cpi.offset],
                                         cpi.offsets.head<int32_t>()[cpi.offset + cpi.size]);
                   });
  thrust::host_vector<thrust::pair<size_type, size_type>> host_offsets(device_offsets);
  thrust::host_vector<column_split_info> split_info(t.num_columns());
  std::for_each(offset_columns.begin(),
                offset_columns.end(),
                [&split_info, &host_offsets](column_preprocess_info const& cpi) {
                  int32_t offset_start = host_offsets[cpi.index].first;
                  int32_t offset_end   = host_offsets[cpi.index].second;
                  auto num_chars       = offset_end - offset_start;
                  split_info[cpi.index].data_buf_size =
                    cudf::util::round_up_safe(static_cast<size_t>(num_chars), split_align);
                  split_info[cpi.index].validity_buf_size =
                    cpi.has_nulls ? cudf::bitmask_allocation_size_bytes(cpi.size, split_align) : 0;
                  split_info[cpi.index].offsets_buf_size =
                    cudf::util::round_up_safe((cpi.size + 1) * sizeof(size_type), split_align);
                  split_info[cpi.index].num_chars    = num_chars;
                  split_info[cpi.index].chars_offset = offset_start;
                });
  return split_info;
}

/**
 * @brief Creates a contiguous_split_result object which contains a deep-copy of the input
 * table_view into a single contiguous block of memory.
 *
 * The table_view contained within the contiguous_split_result will pass an expect_tables_equal()
 * call with the input table.  The memory referenced by the table_view and its internal column_views
 * is entirely contained in single block of memory.
 */
contiguous_split_result alloc_and_copy(cudf::table_view const& t,
                                       rmm::device_vector<column_split_info>& device_split_info,
                                       rmm::mr::device_memory_resource* mr,
                                       cudaStream_t stream)
{
  // preprocess column split information for string columns.
  thrust::host_vector<column_split_info> split_info =
    preprocess_string_column_info(t, device_split_info, stream);

  // compute the rest of the column sizes (non-string columns, and total buffer size)
  size_t total_size      = 0;
  size_type column_index = 0;
  std::for_each(
    t.begin(), t.end(), [&total_size, &column_index, &split_info](cudf::column_view const& c) {
      total_size +=
        cudf::type_dispatcher(c.type(), column_buffer_size_functor{}, c, split_info[column_index]);
      column_index++;
    });

  // allocate
  auto device_buf = std::make_unique<rmm::device_buffer>(total_size, stream, mr);
  char* buf       = static_cast<char*>(device_buf->data());

  // copy (this would be cleaner with a std::transform, but there's an nvcc compiler issue in the
  // way)
  std::vector<column_view> out_cols;
  out_cols.reserve(t.num_columns());

  column_index = 0;
  std::for_each(
    t.begin(), t.end(), [&out_cols, &buf, &column_index, &split_info](cudf::column_view const& c) {
      cudf::type_dispatcher(
        c.type(), column_copy_functor{}, c, split_info[column_index], buf, out_cols);
      column_index++;
    });

  return contiguous_split_result{cudf::table_view{out_cols}, std::move(device_buf)};
}

};  // anonymous namespace

struct _size_of_helper {
  template <typename T>
  constexpr std::enable_if_t<not is_fixed_width<T>(), int> __device__ operator()() const
  {
    return 0;
  }

  template <typename T>
  constexpr std::enable_if_t<is_fixed_width<T>(), int> __device__ operator()() const noexcept
  {
    return sizeof(T);
  }
};


inline __device__ size_t _round_up_safe(size_t number_to_round, size_t modulus)
{
  auto remainder = number_to_round % modulus;
  if (remainder == 0) { return number_to_round; }
  auto rounded_up = number_to_round - remainder + modulus;
  return rounded_up;
}

struct _src_buf_info {
  cudf::type_id     type;
  const int         *offsets;
  bool              is_offset_column;
  bool              is_validity;
};

struct _dst_buf_info {
  size_t    buf_size;       // total size of buffer, including padding
  int       num_elements;   // # of elements to be copied
  int       element_size;
  int       src_row_index;
  int       dst_offset;   
  int       value_shift;
  int       bit_shift;  
};

struct dst_offset_output_iterator {
  _dst_buf_info *c;
  using value_type        = int;
  using difference_type   = int;
  using pointer           = int *;
  using reference         = int &;
  using iterator_category = thrust::output_device_iterator_tag;

  dst_offset_output_iterator operator+ __host__ __device__(int i)
  {
    return dst_offset_output_iterator{c + i};
  }

  void operator++ __host__ __device__() { c++; }

  reference operator[] __device__(int i) { return dereference(c + i); }
  reference operator*__device__() { return dereference(c); }

 private:
  reference __device__ dereference(_dst_buf_info *c)
  {
    return c->dst_offset;
  }
};

__device__ void copy_buffer(uint8_t * __restrict__ dst, uint8_t *__restrict__ _src,
                            int t, int num_elements, int element_size,
                            int src_row_index, uint32_t stride, int value_shift, int bit_shift)
{
  uint8_t *src = _src + (src_row_index * element_size);

  // handle misalignment. read 16 bytes in 4 byte reads. write in a single 16 byte store.
  const size_t num_bytes = num_elements * element_size;
  // how many bytes we're misaligned from 4-byte alignment
  const uint32_t ofs = reinterpret_cast<uintptr_t>(src) % 4;
  size_t pos = t*16;
  stride *= 16;
  while(pos+20 <= num_bytes){
    // read from the nearest aligned address.
    const uint32_t* in32 = reinterpret_cast<const uint32_t *>((src + pos) - ofs);
    uint4 v = { in32[0], in32[1], in32[2], in32[3] };
    if (ofs || bit_shift) {
      v.x = __funnelshift_r(v.x, v.y, ofs*8 + bit_shift);
      v.y = __funnelshift_r(v.y, v.z, ofs*8 + bit_shift);
      v.z = __funnelshift_r(v.z, v.w, ofs*8 + bit_shift);
      // don't read off the end
      v.w = __funnelshift_r(v.w, /*pos+20 <= num_bytes ? in32[4] : 0*/in32[4], ofs*8 + bit_shift);
    }           
    v.x -= value_shift;
    v.y -= value_shift;
    v.z -= value_shift;
    v.w -= value_shift;
    reinterpret_cast<uint4*>(dst)[pos/16] = v;
    pos += stride;
  }  

  // copy trailing bytes
  if(t == 0){
    size_t remainder = num_bytes < 16 ? num_bytes : 16 + (num_bytes % 16);

    /*
    if(blockIdx.x == 80){
      printf("R : num_bytes(%lu), pos(%lu), remainder(%lu)\n", num_bytes, pos, remainder);
    }
    */

    // if we're performing a value shift (offsets), or a bit shift (validity) the # of bytes and 
    // alignment must be a multiple of 4 
    if(value_shift || bit_shift){
      int idx = num_bytes-4;
      uint32_t carry = 0;
      while(remainder){
        uint32_t v = reinterpret_cast<uint32_t*>(src)[idx/4];        
        //printf("VA : v(%x), (shift)%d, carry(%x), result(%x)\n", v, bit_shift, carry, ((v - value_shift) >> bit_shift) | carry);
        reinterpret_cast<uint32_t*>(dst)[idx/4] = ((v - value_shift) >> bit_shift) | carry;        
        carry = (v & ((1<<bit_shift)-1)) << (32 - bit_shift);
        remainder -= 4;
        idx-=4;
      }
    } else {
      while(remainder){
        int idx = num_bytes - remainder--;
        reinterpret_cast<uint8_t*>(dst)[idx] = reinterpret_cast<uint8_t*>(src)[idx];
      }
    }
  }
}

__global__ void copy_partitions(int num_src_bufs, int num_partitions, int num_bufs,
                                uint8_t **src_bufs,
                                uint8_t **dst_bufs,
                                _dst_buf_info *buf_info)
{   
  int partition_index = blockIdx.x / num_src_bufs;
  int src_buf_index = blockIdx.x % num_src_bufs;
  int t = threadIdx.x; 
  size_t buf_index = (partition_index * num_src_bufs) + src_buf_index; 
  int num_elements = buf_info[buf_index].num_elements;    
  int element_size = buf_info[buf_index].element_size;
  int stride = blockDim.x;
 
  int src_row_index = buf_info[buf_index].src_row_index;
  uint8_t *src = src_bufs[src_buf_index];
  uint8_t *dst = dst_bufs[partition_index] + buf_info[buf_index].dst_offset;

  // copy, shifting offsets and validity bits as needed
  copy_buffer(dst, src, t, num_elements, element_size, src_row_index, stride, buf_info[buf_index].value_shift, buf_info[buf_index].bit_shift);
}

struct get_col_data {
  template<typename T, std::enable_if_t<cudf::is_fixed_width<T>()>* = nullptr>
  void operator()(column_view const& col, size_type &col_index, uint8_t **out_buf)
  {
    out_buf[col_index++] = reinterpret_cast<uint8_t*>(const_cast<T*>(col.begin<T>()));
    if(col.nullable()){      
      out_buf[col_index++] = reinterpret_cast<uint8_t*>(const_cast<bitmask_type*>(col.null_mask()));
    }
  }

  template<typename T, std::enable_if_t<!cudf::is_fixed_width<T>() && std::is_same<T, cudf::string_view>::value>* = nullptr>
  void operator()(column_view const& col, size_type &col_index, uint8_t **out_buf)
  {
    strings_column_view scv(col);
    out_buf[col_index++] = reinterpret_cast<uint8_t*>(const_cast<size_type*>(scv.offsets().begin<size_type>()));
    out_buf[col_index++] = reinterpret_cast<uint8_t*>(const_cast<int8_t*>(scv.chars().begin<int8_t>()));
    if(col.nullable()){      
      out_buf[col_index++] = reinterpret_cast<uint8_t*>(const_cast<bitmask_type*>(col.null_mask()));
    }
  }

  template<typename T, std::enable_if_t<!cudf::is_fixed_width<T>() && !std::is_same<T, cudf::string_view>::value>* = nullptr>
  void operator()(column_view const& col, size_type &col_index, uint8_t **out_buf)
  {
    CUDF_FAIL("unsupported type");    
  }
};


struct buf_size_functor {  
  _dst_buf_info const* ci;
  size_t operator() __device__ (int index)
  {
    //printf("Size : %d (%lu)\n", ci[index].buf_size, (uint64_t)(&ci[index]));
    return static_cast<size_t>(ci[index].buf_size);
  }
};

struct split_key_functor {
  int num_columns;
  int operator() __device__ (int t)
  {
    //printf("Key : %d (%d, %d)\n", t / num_columns, t , num_columns);
    return t / num_columns;
  }
};


template<typename ColIter>
size_type count_src_bufs(ColIter first, ColIter last);

struct buf_count_functor {
  template<typename T, std::enable_if_t<std::is_same<T, cudf::string_view>::value>* = nullptr>
  size_type operator()(column_view const& col)
  {
    return 2 + (col.nullable() ? 1 : 0);
  }

  template<typename T, std::enable_if_t<std::is_same<T, cudf::list_view>::value or
                                        std::is_same<T, cudf::struct_view>::value>* = nullptr>
  size_type operator()(column_view const& col)
  {
    return count_src_bufs(col.child_begin(), col.child_end()) + (col.nullable() ? 1 : 0);
  }

  template<typename T, std::enable_if_t<std::is_same<T, cudf::dictionary32>::value>* = nullptr>
  size_type operator()(column_view const& col)
  {
    CUDF_FAIL("Unsupported type");
  }

  template<typename T, std::enable_if_t<!cudf::is_compound<T>()>* = nullptr>
  size_type operator()(column_view const& col)
  {
    return 1 + (col.nullable() ? 1 : 0);
  }
};

template<typename ColIter>
size_type count_src_bufs(ColIter first, ColIter last)
{
  auto buf_iter = thrust::make_transform_iterator(first, [](column_view const& col){
    // auto const& col = cols[column_index];
    return cudf::type_dispatcher(col.type(), buf_count_functor{}, col);
  });
  return std::accumulate(buf_iter, buf_iter + std::distance(first, last), 0);
}

std::vector<contiguous_split_result> contiguous_split(cudf::table_view const& input,
                                                      std::vector<size_type> const& splits,
                                                      rmm::mr::device_memory_resource* mr,
                                                      cudaStream_t stream)
{ 
#if defined(__NEW_PATH)     
  if (input.num_columns() == 0) { return {}; }
  if(splits.size() > 0){
    CUDF_EXPECTS(splits.back() <= input.column(0).size(), "splits can't exceed size of input columns");
  }

  size_t num_root_columns = input.num_columns();

  // compute # of source buffers (column data, validity, children), # of partitions
  // and total # of buffers
  size_type num_src_bufs = count_src_bufs(input.begin(), input.end());

  /*
  size_t num_src_bufs = 0;
  for(size_t idx=0; idx<num_root_columns; idx++){
    if(input.column(idx).type().id() == type_id::STRING){      
      num_columns+=2;
    } else {
      num_columns++;
    }
    if(input.column(idx).nullable()){
      num_columns++;
    }
  }
  */
  size_t num_partitions = splits.size() + 1;
  size_t num_bufs = num_src_bufs * num_partitions;  

  // compute total size of host-side temp data
  size_t indices_size = cudf::util::round_up_safe((splits.size() + 1) * 2 * sizeof(size_type), split_align);
  size_t src_buf_info_size = cudf::util::round_up_safe(num_src_bufs * sizeof(_src_buf_info), split_align);  
  size_t buf_sizes_size = cudf::util::round_up_safe(num_partitions * sizeof(size_t), split_align);
  size_t dst_buf_info_size = cudf::util::round_up_safe(num_bufs * sizeof(_dst_buf_info), split_align);
  size_t src_bufs_size = cudf::util::round_up_safe(num_src_bufs * sizeof(uint8_t*), split_align);
  size_t dst_bufs_size = cudf::util::round_up_safe(num_partitions * sizeof(uint8_t*), split_align);
  size_t total_temp_size = indices_size + src_buf_info_size + buf_sizes_size + dst_buf_info_size + src_bufs_size + dst_bufs_size;
  
  // allocate host
  std::vector<uint8_t> host_buf(total_temp_size);
  
  // distribute
  uint8_t *cur_h_buf = host_buf.data();
  size_type     *h_indices = reinterpret_cast<size_type*>(cur_h_buf);           cur_h_buf += indices_size;
  _src_buf_info *h_src_buf_info = reinterpret_cast<_src_buf_info*>(cur_h_buf);  cur_h_buf += src_buf_info_size;  
  size_t        *h_buf_sizes = reinterpret_cast<size_t*>(cur_h_buf);            cur_h_buf += buf_sizes_size;
  _dst_buf_info *h_dst_buf_info = reinterpret_cast<_dst_buf_info*>(cur_h_buf);  cur_h_buf += dst_buf_info_size;
  uint8_t       **h_src_bufs = reinterpret_cast<uint8_t**>(cur_h_buf);          cur_h_buf += src_bufs_size;
  uint8_t       **h_dst_bufs = reinterpret_cast<uint8_t**>(cur_h_buf);


  // allocate device
  rmm::device_buffer device_buf{total_temp_size, stream, mr};

  // distribute
  uint8_t *cur_d_buf = reinterpret_cast<uint8_t*>(device_buf.data());
  size_type     *d_indices = reinterpret_cast<size_type*>(cur_d_buf);           cur_d_buf += indices_size;
  _src_buf_info *d_src_buf_info = reinterpret_cast<_src_buf_info*>(cur_d_buf);  cur_d_buf += src_buf_info_size;  
  size_t        *d_buf_sizes = reinterpret_cast<size_t*>(cur_d_buf);            cur_d_buf += buf_sizes_size;
  _dst_buf_info *d_dst_buf_info = reinterpret_cast<_dst_buf_info*>(cur_d_buf);  cur_d_buf += dst_buf_info_size;
  uint8_t       **d_src_bufs = reinterpret_cast<uint8_t**>(cur_d_buf);          cur_d_buf += src_bufs_size;
  uint8_t       **d_dst_bufs = reinterpret_cast<uint8_t**>(cur_d_buf);

  // compute splits -> indices
  {
    size_type *indices = h_indices;
    *indices = 0;
    indices++;
    std::for_each(splits.begin(), splits.end(), [&indices](auto split) {
      *indices = split;
      indices++;
      *indices = split;
      indices++; 
    });
    *indices = input.column(0).size();
  
    for (size_t i = 0; i < splits.size(); i++) {
      auto begin = h_indices[2 * i];
      auto end   = h_indices[2 * i + 1];
      CUDF_EXPECTS(begin >= 0, "Starting index cannot be negative.");
      CUDF_EXPECTS(end >= begin, "End index cannot be smaller than the starting index.");
      CUDF_EXPECTS(end <= input.column(0).size(), "Slice range out of bounds.");
    }   
  }
  
  // setup column types
  {
    int col_index = 0;
    for(size_t idx=0; idx<num_root_columns; idx++){
      if(input.column(idx).type().id() == type_id::STRING){
        strings_column_view scv(input.column(idx));

        h_src_buf_info[col_index].type = type_id::INT32; 
        h_src_buf_info[col_index].offsets = scv.offsets().begin<int>();
        h_src_buf_info[col_index].is_offset_column = true;
        h_src_buf_info[col_index].is_validity = false;
        col_index++;

        h_src_buf_info[col_index].type = type_id::INT8;  // chars
        h_src_buf_info[col_index].offsets = scv.offsets().begin<int>();
        h_src_buf_info[col_index].is_offset_column = false;
        h_src_buf_info[col_index].is_validity = false;        
        col_index++;
      } else {
        h_src_buf_info[col_index].type = input.column(idx).type().id();
        h_src_buf_info[col_index].offsets = nullptr;
        h_src_buf_info[col_index].is_offset_column = false;
        h_src_buf_info[col_index].is_validity = false; 
        col_index++;
      }

      // if we have validity
      if(input.column(idx).nullable()){
        h_src_buf_info[col_index].type = type_id::INT32; 
        h_src_buf_info[col_index].offsets = 0;
        h_src_buf_info[col_index].is_offset_column = false;
        h_src_buf_info[col_index].is_validity = true; 
        col_index++;
      }
    }
  }

  // HtoD indices and source buf info to device
  cudaMemcpyAsync(d_indices, h_indices, indices_size + src_buf_info_size, cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);

  // compute sizes of each column in each partition, including alignment.
  thrust::transform(rmm::exec_policy(stream)->on(stream),
    thrust::make_counting_iterator<size_t>(0),
    thrust::make_counting_iterator<size_t>(num_bufs),
    d_dst_buf_info,
    [num_src_bufs, d_indices, d_src_buf_info] __device__ (size_t t){      
      int split_index = t / num_src_bufs;
      int column_index = t % num_src_bufs;
      auto const& src_info = d_src_buf_info[column_index];
      
      int row_index_start = d_indices[split_index * 2];
      int row_index_end = d_indices[split_index *2 + 1];
      int value_shift = 0;
      int bit_shift = 0;

      // if I am an offsets column, all my values need to be shifted
      if(src_info.is_offset_column){
        //printf("Value shift pre: %lu, %d\n", (uint64_t)src_info.offsets, row_index_start);
        value_shift = src_info.offsets[row_index_start];
        //printf("Value shift post: %d\n", value_shift);
      }
      // otherwise, if I have an associated offsets column adjust indices
      else if(src_info.offsets != nullptr){      
        row_index_start = src_info.offsets[row_index_start];
        row_index_end = src_info.offsets[row_index_end];        
      }      
      int num_elements = row_index_end - row_index_start;
      if(src_info.is_offset_column){
        num_elements++;
      }
      if(src_info.is_validity){
        bit_shift = row_index_start % 32;        
        num_elements = (num_elements + 31) / 32;
        row_index_start /= 32;
        row_index_end /= 32;
      }
      int element_size = cudf::type_dispatcher(data_type{src_info.type}, _size_of_helper{});
      size_t bytes = num_elements * element_size;
      //printf("Col %d, split %d (%d, %d), (%d, %lu), (%d, %d)\n", column_index, split_index, row_index_start, row_index_end, num_elements, bytes, value_shift, bit_shift);
      return _dst_buf_info{_round_up_safe(bytes, 64), num_elements, element_size, row_index_start, 0, value_shift, bit_shift};
    });

  // DtoH buf sizes and dest buf info back to the host
  cudaMemcpyAsync(h_buf_sizes, d_buf_sizes, buf_sizes_size + dst_buf_info_size, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  // compute total size of each partition
  {
    // key is split index
    auto keys = thrust::make_transform_iterator(thrust::make_counting_iterator(0), split_key_functor{static_cast<int>(num_src_bufs)});
    auto values = thrust::make_transform_iterator(thrust::make_counting_iterator(0), buf_size_functor{d_dst_buf_info});

    thrust::reduce_by_key(rmm::exec_policy(stream)->on(stream), 
      keys, keys + num_bufs, values, thrust::make_discard_iterator(), d_buf_sizes);
  }
  
  /*
  // DtoH buf sizes and col info back to the host
  cudaMemcpyAsync(h_buf_sizes, d_buf_sizes, buf_sizes_size, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  for(size_t idx=0; idx<num_partitions; idx++){
    printf("partition %lu : %lu\n", idx, h_buf_sizes[idx]);
  } 
  */ 

  // compute start offset for each output buffer
  {
    auto keys = thrust::make_transform_iterator(thrust::make_counting_iterator(0), split_key_functor{static_cast<int>(num_src_bufs)});
    auto values = thrust::make_transform_iterator(thrust::make_counting_iterator(0), buf_size_functor{d_dst_buf_info});
    thrust::exclusive_scan_by_key(rmm::exec_policy(stream)->on(stream),
      keys, keys + num_bufs, values, dst_offset_output_iterator{d_dst_buf_info}, 0);
  }

  // DtoH buf sizes and col info back to the host
  cudaMemcpyAsync(h_buf_sizes, d_buf_sizes, buf_sizes_size + dst_buf_info_size, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  /*
  for(size_t idx=0; idx<h_buf_sizes.size(); idx++){
    printf("partition %lu : %lu\n", idx, h_buf_sizes[idx]);
  }
  */    
  // DtoH column info back to hose
  /*
  thrust::host_vector<_column_info> h_col_info(num_bufs);
  cudaMemcpyAsync(h_col_info.data(), d_col_info.data(), sizeof(_column_info) * num_bufs, cudaMemcpyDeviceToHost);
  */
  /*
  thrust::host_vector<size_t> h_dst_offsets(num_bufs);
  cudaMemcpyAsync(h_dst_offsets.data(), d_dst_offsets.data(), sizeof(size_t) * num_bufs, cudaMemcpyDeviceToHost);  
  */
  /*
  // debug  
  for(size_t idx=0; idx<h_col_info.size(); idx++){
    printf("size/offset : (%d, %d), %lu\n", h_col_info[idx].num_elements, h_col_info[idx].buf_size, h_dst_offsets[idx]);
  } 
  */   

  // allocate output partition buffers
  std::vector<rmm::device_buffer> out_buffers;
  out_buffers.reserve(num_partitions);
  std::transform(h_buf_sizes, h_buf_sizes + num_partitions, std::back_inserter(out_buffers), [stream, mr](size_t bytes){    
    return rmm::device_buffer{bytes, stream, mr};
  });

  // setup src buffers
  {
    size_type out_index = 0;  
    std::for_each(input.begin(), input.end(), [&out_index, &h_src_bufs](column_view const& col){
      cudf::type_dispatcher(col.type(), get_col_data{}, col, out_index, h_src_bufs);
    });  
  }

  // setup dst buffers
  {
    size_type out_index = 0;  
    std::for_each(out_buffers.begin(), out_buffers.end(), [&out_index, &h_dst_bufs](rmm::device_buffer& buf){
      h_dst_bufs[out_index++] = reinterpret_cast<uint8_t*>(buf.data());
    });
  }

  // HtoD src and dest buffers
  cudaMemcpyAsync(d_src_bufs, h_src_bufs, src_bufs_size + dst_bufs_size, cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);  
    
  // copy.  1 block per buffer  
  {
    // scope_timer timer("kernel");
    constexpr int block_size = 512;
    copy_partitions<<<num_bufs, block_size, 0, stream>>>(num_src_bufs, num_partitions, num_bufs,
                                                         d_src_bufs,
                                                         d_dst_bufs,
                                                         d_dst_buf_info);

    // TODO : put this after the column building step below to overlap work
    CUDA_TRY(cudaStreamSynchronize(stream));
  }

  // build the output.
  std::vector<contiguous_split_result> result;  
  result.reserve(num_partitions);
  size_t buf_index = 0;
  for(size_t idx=0; idx<num_partitions; idx++){
    std::vector<column_view> cols;
    cols.reserve(input.num_columns());    
    for(size_t s_idx=0; s_idx<num_root_columns; s_idx++){
      cudf::type_id id = input.column(s_idx).type().id();
      bool nullable = input.column(s_idx).nullable();
      
      if(id == type_id::STRING){
        cols.push_back(cudf::column_view{data_type{type_id::STRING}, 
                                          h_dst_buf_info[buf_index].num_elements-1, 
                                          nullptr,
                                          nullable ? reinterpret_cast<bitmask_type*>(h_dst_bufs[idx] + h_dst_buf_info[buf_index+2].dst_offset) : nullptr,
                                          nullable ? UNKNOWN_NULL_COUNT : 0,
                                          0,
                                          {
                                            cudf::column_view{data_type{type_id::INT32}, h_dst_buf_info[buf_index].num_elements, 
                                            reinterpret_cast<void*>(h_dst_bufs[idx] + h_dst_buf_info[buf_index].dst_offset)},

                                            cudf::column_view{data_type{type_id::INT8}, h_dst_buf_info[buf_index+1].num_elements, 
                                            reinterpret_cast<void*>(h_dst_bufs[idx] + h_dst_buf_info[buf_index+1].dst_offset)}
                                          }});

        // cudf::test::print(cols.back());
        buf_index+=2;
      } else {
        cols.push_back(cudf::column_view{data_type{id}, 
                                         h_dst_buf_info[buf_index].num_elements, 
                                         reinterpret_cast<void*>(h_dst_bufs[idx] + h_dst_buf_info[buf_index].dst_offset),
                                         nullable ? reinterpret_cast<bitmask_type*>(h_dst_bufs[idx] + h_dst_buf_info[buf_index+1].dst_offset) : nullptr,
                                         nullable ? UNKNOWN_NULL_COUNT : 0
                                         });
        buf_index++;
      }      
      if(nullable){
        buf_index++;
      }
    }
    result.push_back(contiguous_split_result{cudf::table_view{cols}, std::make_unique<rmm::device_buffer>(std::move(out_buffers[idx]))});
  }    

  return std::move(result);
#else
  auto subtables = cudf::split(input, splits);
  
  // optimization : for large numbers of splits this allocation can dominate total time
  //                spent if done inside alloc_and_copy().  so we'll allocate it once
  //                and reuse it.
  //
  //                benchmark:        1 GB data, 10 columns, 256 splits.
  //                no optimization:  106 ms (8 GB/s)
  //                optimization:     20 ms (48 GB/s)
  rmm::device_vector<column_split_info> device_split_info(input.num_columns());  

  std::vector<contiguous_split_result> result;
  std::transform(subtables.begin(),
                 subtables.end(),
                 std::back_inserter(result),
                 [mr, stream, &device_split_info](table_view const& t) {
                   return alloc_and_copy(t, device_split_info, mr, stream);
                 });                 

  return result;
#endif  
}

};  // namespace detail

std::vector<contiguous_split_result> contiguous_split(cudf::table_view const& input,
                                                      std::vector<size_type> const& splits,
                                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return cudf::detail::contiguous_split(input, splits, mr, (cudaStream_t)0);
}

};  // namespace cudf

template <typename T>
void BM_contiguous_split_common(std::vector<T>& src_cols,
                                int64_t num_rows,
                                int64_t num_splits,
                                int64_t bytes_total)
{
  // generate splits
  cudf::size_type split_stride = num_rows / num_splits;
  std::vector<cudf::size_type> splits;
  for (int idx = 0; idx < num_rows; idx += split_stride) {
    splits.push_back(std::min(idx + split_stride, static_cast<cudf::size_type>(num_rows)));
  }

  std::vector<std::unique_ptr<cudf::column>> columns(src_cols.size());
  std::transform(src_cols.begin(), src_cols.end(), columns.begin(), [](T& in) {
    auto ret = in.release();
    ret->null_count();
    return ret;
  });  
  cudf::table src_table(std::move(columns));

  thrust::device_vector<uint8_t> temp(bytes_total);

  std::vector<contiguous_split_result> result;
  {
    // scope_timer timer("split");
    CUDF_FUNC_RANGE();
    result = cudf::contiguous_split(src_table, splits);

    auto regular_split = cudf::split(src_table, splits);

    /*
    for(size_t idx=0; idx<result.size(); idx++){
      cudf::test::expect_tables_equal(result[idx].table, regular_split[idx]);
    }
    */
   for(size_t idx=0; idx<result.size(); idx++){
      printf("Column %lu: (size %d)\n", idx, result[idx].table.column(0).size());
      for(size_t s_idx=0; s_idx<result[idx].table.num_columns(); s_idx++){        
        cudf::test::expect_columns_equal(result[idx].table.column(s_idx), regular_split[idx].column(s_idx));
      }
    }
    cudf::test::print(result[80].table.column(0));
    //cudf::column_view c(src_table.get_column(0));
    //cudaMemcpyAsync(temp.data().get(), c.begin<int>(), bytes_total, cudaMemcpyDeviceToDevice, 0);
    //cudaStreamSynchronize(0);
  }
}

void BM_contiguous_split(int64_t total_bytes, int64_t _num_cols, int64_t _num_splits, bool _include_validity)
{
  int64_t total_desired_bytes = total_bytes;
  cudf::size_type num_cols    = _num_cols;
  cudf::size_type num_splits  = _num_splits;
  bool include_validity       = _include_validity;

  cudf::size_type el_size = 4;  // ints and floats
  int64_t num_rows        = total_desired_bytes / (num_cols * el_size);

  // generate input table
  srand(31337);
  auto valids = cudf::test::make_counting_transform_iterator(0, [](auto i) { return true; });
  std::vector<cudf::test::fixed_width_column_wrapper<int>> src_cols(num_cols);
  for (int idx = 0; idx < num_cols; idx++) {
    auto rand_elements =
      cudf::test::make_counting_transform_iterator(0, [](int i) { return rand(); });
    if (include_validity) {
      src_cols[idx] = cudf::test::fixed_width_column_wrapper<int>(
        rand_elements, rand_elements + num_rows, valids);
    } else {
      src_cols[idx] =
        cudf::test::fixed_width_column_wrapper<int>(rand_elements, rand_elements + num_rows);
    }
  }

  // size_t total_bytes = total_desired_bytes;
  if (include_validity) { total_desired_bytes += num_rows / (sizeof(cudf::bitmask_type) * 8); }

  BM_contiguous_split_common(src_cols, num_rows, num_splits, total_desired_bytes);
}

std::vector<cudf::size_type> splits_to_indices(std::vector<cudf::size_type> splits,
                                               cudf::size_type size)
{
  std::vector<cudf::size_type> indices{0};

  std::for_each(splits.begin(), splits.end(), [&indices](auto split) {
    indices.push_back(split);  // This for end
    indices.push_back(split);  // This for the start
  });

  indices.push_back(size);  // This to include rest of the elements

  return indices;
}

std::vector<cudf::table> create_expected_string_tables_for_splits(
  std::vector<std::string> const strings[2],
  std::vector<cudf::size_type> const& splits,
  bool nullable)
{
  std::vector<cudf::size_type> indices = splits_to_indices(splits, strings[0].size());
  return create_expected_string_tables(strings, indices, nullable);
}

void contig_split_test()
{           
  {
    cudf::test::fixed_width_column_wrapper<float> col0{0, 1, 2, 3, 4};
    cudf::test::fixed_width_column_wrapper<short> col1{5, 6, 7, 8, 9};
    cudf::test::fixed_width_column_wrapper<int> col2{10, 11, 12, 13, 14};
    cudf::test::fixed_width_column_wrapper<uint8_t> col3{15, 16, 17, 18, 19};
    cudf::test::fixed_width_column_wrapper<uint8_t> col4{20, 21, 22, 23, 24};
    //cudf::test::strings_column_wrapper col5{"cats", "dogs", "fish", "ants", "bears"};
    std::vector<size_type> splits{1};
    cudf::table_view tbl({col0, col1, col2, col3, col4});
    auto result = contiguous_split(tbl, splits);
    auto expect = cudf::split(tbl, splits);

    CUDF_EXPECTS(result.size() == expect.size(), "Size mismatch");
    for(size_t idx=0; idx<result.size(); idx++){
      cudf::test::expect_tables_equal(result[idx].table, expect[idx]);
    }
  }  

  {
    cudf::test::strings_column_wrapper col0{"abc", "e", "fg", "hijk", "lmnkop"};
    std::vector<size_type> splits{1, 2};
    cudf::table_view tbl({col0});
    auto result = contiguous_split(tbl, splits);
    auto expect = cudf::split(tbl, splits);

    CUDF_EXPECTS(result.size() == expect.size(), "Size mismatch");
    for(size_t idx=0; idx<result.size(); idx++){
      cudf::test::expect_tables_equal(result[idx].table, expect[idx]);
    }
  }

  {
    auto valids = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });

    std::vector<std::string> strings[2] = {
      {"", "this", "is", "a", "column", "of", "strings", "with", "in", "valid"},
      {"", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}};
    cudf::test::strings_column_wrapper sw[2] = {{strings[0].begin(), strings[0].end(), valids},
                                                {strings[1].begin(), strings[1].end(), valids}};

    std::vector<std::unique_ptr<cudf::column>> scols;
    scols.push_back(sw[0].release());
    scols.push_back(sw[1].release());
    cudf::table src_table(std::move(scols));

    std::vector<cudf::size_type> splits{2, 5, 9};

    std::vector<cudf::table> expected =
      create_expected_string_tables_for_splits(strings, splits, true);

    auto result = contiguous_split(src_table, splits);

    EXPECT_EQ(expected.size(), result.size());

    for (unsigned long index = 0; index < result.size(); index++) {
      cudf::test::expect_tables_equal(expected[index], result[index].table);
    }
  }  

  {
    // cudf::test::fixed_width_column_wrapper<int> col0{{0, 1, 2, 3, 4}, {1, 0, 1, 0, 1}};
    cudf::test::fixed_width_column_wrapper<int> col0{
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34}
    , {1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  0,  1}};
    std::vector<size_type> splits{2};
    cudf::table_view tbl({col0});
    auto result = contiguous_split(tbl, splits);
    //for(size_t idx=0; idx<result.size(); idx++){
//      cudf::test::print(result[idx].table.column(0));
    //}
    auto expect = split(tbl, splits);
    
    CUDF_EXPECTS(result.size() == expect.size(), "Size mismatch");
    for(size_t idx=0; idx<result.size(); idx++){
      cudf::test::expect_tables_equal(result[idx].table, expect[idx]);
    }
  }

  // baseline
  /*
   split : 0.89 ms
   split : 6.73 ms
   split : 39.13 ms
   split : 412.37 ms
   split : 2.24 ms
   split : 8.02 ms
   split : 44.10 ms
   split : 357.52 ms
   split : 10.80 ms
   split : 17.04 ms
   split : 50.41 ms
   split : 359.26 ms
   */
  
  // new method, baseline
  /*
    split : 0.63 ms
   split : 0.87 ms
   split : 1.78 ms
   split : 14.11 ms

   split : 2.06 ms
   split : 2.18 ms
   split : 2.98 ms
   split : 12.74 ms

   split : 14.21 ms
   split : 11.53 ms
   split : 11.22 ms
   split : 19.23 ms
   */

  // new method, optimized
  /*
   split : 0.21 ms
   split : 0.35 ms
   split : 0.67 ms
   split : 5.42 ms

   split : 1.99 ms
   split : 1.81 ms
   split : 1.87 ms
   split : 4.97 ms

   split : 15.78 ms
   split : 11.36 ms
   split : 10.90 ms
   split : 11.87 ms
   */
  //                  total bytes        num columns  num_splits
  // BM_contiguous_split(1 * 1024 * 1024,   10,          10,         false); 
  // BM_contiguous_split(1 * 1024 * 1024,   10,          100,        false); 
  //BM_contiguous_split(1 * 1024 * 1024,   1,          1,        false); 
  //BM_contiguous_split(1 * 1024 * 1024,   100,         100,        false); 
  //BM_contiguous_split(1 * 1024 * 1024,   1000,        100,        false);
  printf("\n");
  // BM_contiguous_split(128 * 1024 * 1024,  10,         10,         false); 
  // BM_contiguous_split(128 * 1024 * 1024,  10,         100,        false); 
  //BM_contiguous_split(128 * 1024 * 1024,   1,          1,        false); 
  //BM_contiguous_split(128 * 1024 * 1024,  100,        100,        false); 
  //BM_contiguous_split(128 * 1024 * 1024,  1000,       100,        false);
  printf("\n");
  // BM_contiguous_split(1024 * 1024 * 1024, 10,         10,         false); 
  // BM_contiguous_split(1024 * 1024 * 1024, 10,         100,        false); 
  //BM_contiguous_split(1024 * 1024 * 1024,   1,          80,        false); 
  //BM_contiguous_split(1024 * 1024 * 1024, 100,        100,        false); 
  //BM_contiguous_split(1024 * 1024 * 1024, 1000,       100,        false);
}

 struct stuff {
  int a;
  int b;

  __host__ __device__ stuff() { a = b = -1; }
  __host__ __device__ stuff(int i) { a = i; b = i; }
  __host__ __device__ stuff(stuff const& other) { a = other.a; b = other.b; }
};

void parquet_bug_test()
{
  namespace cudf_io = cudf::io;

  cudf_io::parquet_reader_options read_args = cudf::io::parquet_reader_options::builder(cudf_io::source_info{"test_parquet_reader_list_large_multi_rowgroup_nulls.parquet"});  
  auto result = cudf_io::read_parquet(read_args);
  print_names(result.metadata.schema_info);
  for(int idx=0; idx<result.tbl->view().num_columns(); idx++){
    cudf::test::print(result.tbl->view().column(idx));
  }
}

int main()
{    
   // init stuff
   cuInit(0);   
   auto resource       = cudf::test::create_memory_resource("pool");
   printf("MR : %lu\n", (uint64_t)resource.get());
   rmm::mr::set_current_device_resource(resource.get());      

  // parquet_crash_test();
  //parquet_test_test();    
  //dave::test::parquet_test();
  //parquet_speed_test();
  //parquet_old_test();
  //parquet_list_test();
  //dave::test::parquet_big_list_test(false);
  //parquet_big_list_test(true);
  //parquet_bug_test();
  //parquet_struct_test();   

  contig_split_test();

  // expect_columns_equal_test();

  //hierarchy_test();   
  // list_test();         
  // gather_test();
  // void large_split_tests();
  // large_split_tests();
  // void split_test();
  // split_test(); 
  // json_test();   
  // custom_sink_example();  
  // parquet_writer_test();    
  // parse_unicode_stuff();
  // utf8_test(); 
  // parquet_bug();
  // shift_tests();
  // sequence_test();
  // unicode_fix_test(); 

  // factory_test();  

  // shut stuff down
  // rmmFinalize();
}

#if 0
/*
#if defined(__NEW_PATH) 
  std::vector<size_type> indices{0};
  std::for_each(splits.begin(), splits.end(), [&indices](auto split) {
    indices.push_back(split);  // This for end
    indices.push_back(split);  // This for the start
  });
  indices.push_back(input.column(0).size());
 
  size_t num_columns = input.num_columns();
  size_t num_partitions = splits.size() + 1;
  size_t num_bufs = num_columns * num_partitions;
  
  // setup column types and split indices
  // 1 device allocation          O(num columns)
  // 1 HtoD,                      O(num columns)  
  // 1 device allocation          O(num partitions * 2)
  // 1 HtoD                       O(num partitions * 2)
  //
  thrust::host_vector<cudf::type_id> h_column_types(num_columns);
  for(size_type idx=0; idx<num_columns; idx++){
    h_column_types[idx] = input.column(idx).type().id();
  }
  thrust::device_vector<cudf::type_id> d_column_types(h_column_types);
  thrust::device_vector<size_type> d_indices(indices);

  // compute sizes of each column in each partition, including alignment.
  // 1 device allocation          O(num_columns * num_partitions)
  // 1 kernel call
  rmm::device_uvector<_column_info> d_col_info(num_bufs, stream);
  thrust::transform(rmm::exec_policy(stream)->on(stream), 
    thrust::make_counting_iterator<size_t>(0), 
    thrust::make_counting_iterator<size_t>(num_bufs), 
    d_col_info.begin(), 
    [num_columns = input.num_columns(), indices = d_indices.data().get(), column_types = d_column_types.data().get()] __device__ (size_t t){
      int split_index = t / num_columns;
      int column_index = t % num_columns;
      int row_index_start = indices[split_index * 2];
      int row_index_end = indices[split_index *2 + 1];
      int num_elements = row_index_end - row_index_start;
      size_t bytes = cudf::type_dispatcher(data_type{column_types[column_index]}, _size_of_helper{}) * num_elements;
      return _column_info{_round_up_safe(bytes, 64), num_elements, row_index_start};
      // return _column_info{bytes, num_elements};
      // return _column_info{_round_up_safe(bytes, 64), num_elements};
    });

  // compute total size of each partition
  // 1 device allocation O(num partitions)
  // 1 kernel call 
  // 1 DtoH O(num partitions)  
  rmm::device_uvector<size_t> d_buf_sizes(num_partitions, stream);
  {
    // key is split index
    auto keys = thrust::make_transform_iterator(thrust::make_counting_iterator(0), [num_columns = input.num_columns()] __device__ (int t){      
      return t / num_columns;
    });
    auto values = thrust::make_transform_iterator(d_col_info.begin(), [] __device__ (_column_info const& ci){
      return ci.buf_size;
    });
    thrust::reduce_by_key(rmm::exec_policy(stream)->on(stream), 
      keys, keys + num_bufs, values, thrust::make_discard_iterator(), d_buf_sizes.data());
  }
  thrust::host_vector<size_t> h_buf_sizes(num_partitions);  
  cudaMemcpyAsync(h_buf_sizes.data(), d_buf_sizes.data(), sizeof(size_t) * num_partitions, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  /*
  for(size_t idx=0; idx<h_buf_sizes.size(); idx++){
    printf("partition %lu : %lu\n", idx, h_buf_sizes[idx]);
  }
  */

  // compute start offset for each output buffer
  rmm::device_uvector<size_t> d_dst_offsets(num_bufs, stream);
  {
    auto keys = thrust::make_transform_iterator(thrust::make_counting_iterator(0), [num_columns = input.num_columns()] __device__ (int t){      
      return t / num_columns;
    });
    auto values = thrust::make_transform_iterator(d_col_info.begin(), [] __device__ (_column_info const& ci){
      return ci.buf_size;
    });
    thrust::exclusive_scan_by_key(rmm::exec_policy(stream)->on(stream),
      keys, keys + num_bufs, values, d_dst_offsets.begin(), 0);
  }
  
  thrust::host_vector<_column_info> h_col_info(num_bufs);
  cudaMemcpyAsync(h_col_info.data(), d_col_info.data(), sizeof(_column_info) * num_bufs, cudaMemcpyDeviceToHost);
  thrust::host_vector<size_t> h_dst_offsets(num_bufs);
  cudaMemcpyAsync(h_dst_offsets.data(), d_dst_offsets.data(), sizeof(size_t) * num_bufs, cudaMemcpyDeviceToHost);  
  /*
  // debug  
  for(size_t idx=0; idx<h_col_info.size(); idx++){
    printf("size/offset : (%d, %d), %lu\n", h_col_info[idx].num_elements, h_col_info[idx].buf_size, h_dst_offsets[idx]);
  } 
  */   

  // O(num partitions allocations)  
  std::vector<rmm::device_buffer> out_buffers;
  out_buffers.reserve(num_partitions);
  std::transform(h_buf_sizes.begin(), h_buf_sizes.end(), std::back_inserter(out_buffers), [stream, mr](size_t bytes){
    return rmm::device_buffer{bytes, stream, mr};
  });

  // setup src and dst buffers
  // 1 HtoD
  thrust::host_vector<uint8_t*> h_src_buffers;
  h_src_buffers.reserve(num_columns);
  std::transform(input.begin(), input.end(), std::back_inserter(h_src_buffers), [](column_view const& col){
    return cudf::type_dispatcher(col.type(), get_col_data{}, col);
  });
  thrust::device_vector<uint8_t*> d_src_buffers(h_src_buffers);

  thrust::host_vector<uint8_t*> h_dst_buffers;  
  h_dst_buffers.reserve(num_partitions);
  std::transform(out_buffers.begin(), out_buffers.end(), std::back_inserter(h_dst_buffers), [](rmm::device_buffer& buf){
    return reinterpret_cast<uint8_t*>(buf.data());
  });
  thrust::device_vector<uint8_t*> d_dst_buffers(h_dst_buffers);  
    
  // 1 block per buf
  constexpr int block_size = 1024;
  cudf::detail::grid_1d grid{static_cast<int>(num_bufs), block_size, 1};  
  copy_partitions<<<num_bufs, block_size, 0, stream>>>(num_columns, num_partitions, num_bufs,  
                                                              d_src_buffers.data().get(),                                                              
                                                              d_dst_buffers.data().get(), 
                                                              d_dst_offsets.data(),
                                                              d_col_info.data(), 
                                                              d_column_types.data().get());                                                                
  cudaStreamSynchronize(stream);

  std::vector<contiguous_split_result> result;
  /*  
  result.reserve(num_partitions);
  for(size_t idx=0; idx<num_partitions; idx++){
    std::vector<column_view> cols;
    cols.reserve(input.num_columns());
    for(size_t s_idx=0; s_idx<num_columns; s_idx++){
      size_t buf_index = (idx * num_columns) + s_idx;
      cols.push_back(cudf::column_view{data_type{h_column_types[s_idx]}, h_col_info[buf_index].num_elements, 
                                       reinterpret_cast<void*>(h_dst_buffers[idx] + h_dst_offsets[buf_index])});
    }
    result.push_back(contiguous_split_result{cudf::table_view{cols}, std::make_unique<rmm::device_buffer>(std::move(out_buffers[idx]))});
  }
  */

  return std::move(result);
#endif

#if 0
/*
    // 13 ms         
    while(t*16 < num_elements){
      int el0 = src[t*4];
      int el1 = src[t*4 + 1];
      int el2 = src[t*4 + 2];
      int el3 = src[t*4 + 3];

      dst[t] = 
      
      t += stride;
    } 
    */

#if 0
    if((uint64_t)src % 16 == 0){   
      /*
      if(!t){
        printf("aligned 16\n");
      }
      */

      int num_bytes = num_elements * sizeof(T);                
      
      // misaligned read -> shared memory
      while(t*16 < num_bytes){
        (reinterpret_cast<int4*>(dst))[t] = (reinterpret_cast<int4*>(src))[t];        
        t += stride;
      }

      int remainder = num_bytes%16;
      if (t==0 && remainder!=0) {
        while(remainder) {
          int idx = num_bytes - remainder--;
          reinterpret_cast<uint8_t*>(dst)[idx] = reinterpret_cast<uint8_t*>(src)[idx];
        }
      } 
/*
      // 19 ms    
      int num_bytes = num_elements * sizeof(T);
          
      // 512 ints == 128 int4s per stride
      __shared__ int bounce[512];

      // misaligned read -> shared memory
      while(t*4 < num_bytes){
        ((int*)bounce)[t % 512] = src[t];

        __syncthreads();

        // write back out      
        if(t % 4 == 0){
          (reinterpret_cast<int4*>(dst))[t/4] = (reinterpret_cast<int4*>(bounce))[(t%512)/4];
          // (reinterpret_cast<int4*>(dst))[t/4] = 0;
        }

        t += stride;
        
        __syncthreads();
      }

      int remainder = num_bytes%4;
      if (t==0 && remainder!=0) {
        while(remainder) {
          int idx = num_bytes - remainder--;
          reinterpret_cast<uint8_t*>(dst)[idx] = reinterpret_cast<uint8_t*>(src)[idx];
        }
      }            
      */
    } else if((uint64_t)src % 8 == 0){
      /*
      if(!t){
        printf("aligned 8\n");
      }
      */

      int num_bytes = num_elements * sizeof(T);                
      
      // misaligned read -> shared memory
      while(t*8 < num_bytes){
        (reinterpret_cast<int2*>(dst))[t] = (reinterpret_cast<int2*>(src))[t];        
        t += stride;
      }

      int remainder = num_bytes%8;
      if (t==0 && remainder!=0) {
        while(remainder) {
          int idx = num_bytes - remainder--;
          reinterpret_cast<uint8_t*>(dst)[idx] = reinterpret_cast<uint8_t*>(src)[idx];
        }
      } 
    } else 
    #endif
    {
      /*
      if(!t){
        printf("misaligned\n");
      }
      */

    /*
     int num_bytes = num_elements * sizeof(T);           

      // 13 ms         
      while(t*4 < num_elements){
        dst[t*4] = src[t*4];
        dst[t*4+1] = src[t*4+1];
        dst[t*4+2] = src[t*4+2];
        dst[t*4+3] = src[t*4+3];
        t += stride;
      } 

      int remainder = num_bytes%16;
      if (t==0 && remainder!=0) {
        while(remainder) {
          int idx = num_bytes - remainder--;
          reinterpret_cast<uint8_t*>(dst)[idx] = reinterpret_cast<uint8_t*>(src)[idx];
        }
      } 
      */

      /*
      int num_bytes = num_elements * sizeof(T);           

      // 13 ms         
      while(t*4 < num_elements){
        dst[t*4] = src[t*4];

        dst[t*4+1] = src[t*4+1];
        dst[t*4+2] = src[t*4+2];
        dst[t*4+3] = src[t*4+3];
        t += stride;
      } 

      int remainder = num_bytes%16;
      if (t==0 && remainder!=0) {
        while(remainder) {
          int idx = num_bytes - remainder--;
          reinterpret_cast<uint8_t*>(dst)[idx] = reinterpret_cast<uint8_t*>(src)[idx];
        }
      } 
      */
      
      /*
      uint32_t t = threadIdx.x;
      while(t*16 < num_bytes){
          //uint32_t ofs = 3 & reinterpret_cast<uintptr_t>(src);
          uint32_t ofs = reinterpret_cast<uintptr_t>(src) % 4;
          const uint16_t* in16 = reinterpret_cast<const uint16_t *>(src + t*16 - ofs);
          uint16_t vl[8] = { in16[0], in16[1], in16[2], in16[3], in16[4], in16[5], in16[6], in16[7] };
          uint4& v = reinterpret_cast<uint4&>(vl);
          if (ofs) {
              v.x = __funnelshift_r(v.x, v.y, ofs*8);
              v.y = __funnelshift_r(v.y, v.z, ofs*8);
              v.z = __funnelshift_r(v.z, v.w, ofs*8);

              uint32_t next = (in16[8] << 16) | (in16[9]);
              v.w = __funnelshift_r(v.w, next, ofs*8);
          }
          reinterpret_cast<uint4*>(dst)[t] = v;

          t += stride;
      }
      */  
      
      /*
      uint32_t t = threadIdx.x;
      while(t*16 < num_bytes){
          //uint32_t ofs = 3 & reinterpret_cast<uintptr_t>(src);
          uint32_t ofs = reinterpret_cast<uintptr_t>(src) % 4;
          const uint8_t* in8 = reinterpret_cast<const uint8_t *>(src + t*16 - ofs);
          uint8_t vl[16] = { in8[0], in8[1], in8[2], in8[3], in8[4], in8[5], in8[6], in8[7], 
                             in8[8], in8[9], in8[10], in8[11], in8[12], in8[13], in8[14], in8[15]};
          uint4& v = reinterpret_cast<uint4&>(vl);
          if (ofs) {
              v.x = __funnelshift_r(v.x, v.y, ofs*8);
              v.y = __funnelshift_r(v.y, v.z, ofs*8);
              v.z = __funnelshift_r(v.z, v.w, ofs*8);
              uint32_t next = (in8[16] << 24) | (in8[17] << 16) | (in8[18] << 8) | (in8[19]);
              v.w = __funnelshift_r(v.w, next, ofs*8);
          }       
          reinterpret_cast<uint4*>(dst)[t] = v;

          t += stride;
      }
      */
      
      int remainder = num_bytes%16;
      if (t==0 && remainder!=0) {
        while(remainder) {
          int idx = num_bytes - remainder--;
          reinterpret_cast<uint8_t*>(dst)[idx] = reinterpret_cast<uint8_t*>(src)[idx];
        }
      } 

     /*
     // 19 ms    
      int num_bytes = num_elements * sizeof(T);
          
      // 512 ints == 128 int4s per stride
      __shared__ int bounce[512];

      // misaligned read -> shared memory
      while(t*4 < num_bytes){
        ((int*)bounce)[t % 512] = src[t];

        __syncthreads();

        // write back out      
        if(t % 4 == 0){
          (reinterpret_cast<int4*>(dst))[t/4] = (reinterpret_cast<int4*>(bounce))[(t%512)/4];
          // (reinterpret_cast<int4*>(dst))[t/4] = 0;
        }

        t += stride;
        
        __syncthreads();
      }

      int remainder = num_bytes%16;
      if (t==0 && remainder!=0) {
        while(remainder) {
          int idx = num_bytes - remainder--;
          reinterpret_cast<uint8_t*>(dst)[idx] = reinterpret_cast<uint8_t*>(src)[idx];
        }
      }      
      */
     #endif

     #if 0
     __device__ void copy_buffer(uint8_t * __restrict__ dst, uint8_t *__restrict__ _src,
                            int t, int num_elements, int element_size,
                            int src_row_index, uint32_t stride, int value_shift, int bit_shift)
{
  uint8_t *src = _src + (src_row_index * element_size);

  // handle misalignment. read 16 bytes in 4 byte reads. write in a single 16 byte store.
  const size_t num_bytes = num_elements * element_size;
  // how many bytes we're misaligned from 4-byte alignment
  const uint32_t ofs = reinterpret_cast<uintptr_t>(src) % 4;
  size_t pos = t*16;
  stride *= 16;
  while(pos+16 <= num_bytes){
    // read from the nearest aligned address.
    const uint32_t* in32 = reinterpret_cast<const uint32_t *>((src + pos) - ofs);
    uint4 v = { in32[0], in32[1], in32[2], in32[3] };
    if (ofs || bit_shift) {
      v.x = __funnelshift_r(v.x, v.y, ofs*8 + bit_shift);
      v.y = __funnelshift_r(v.y, v.z, ofs*8 + bit_shift);
      v.z = __funnelshift_r(v.z, v.w, ofs*8 + bit_shift);
      // don't read off the end
      v.w = __funnelshift_r(v.w, pos+20 <= num_bytes ? in32[4] : 0, ofs*8 + bit_shift);
    }           
    v.x -= value_shift;
    v.y -= value_shift;
    v.z -= value_shift;
    v.w -= value_shift;
    reinterpret_cast<uint4*>(dst)[pos/16] = v;
    pos += stride;
  }

  // copy trailing bytes
  if(t == 0){
    size_t remainder = num_bytes % 16;

    if(value_shift > 0){
      // if we're performing a value shift, the # of bytes and alignment must be a multiple of 4
      // because these are offsets.        
      while(remainder){
        int idx = num_bytes - remainder;
        remainder -= 4;        
        int v = reinterpret_cast<int*>(src)[idx/4];
        reinterpret_cast<int*>(dst)[idx/4] = v - value_shift;        
      }
    } else {
      while(remainder){
        int idx = num_bytes - remainder--;
        reinterpret_cast<uint8_t*>(dst)[idx] = reinterpret_cast<uint8_t*>(src)[idx];
      }
    }
  }
}
#endif