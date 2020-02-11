#include "db_test.cuh"


//#if 0    // skeleton for working in cudf
//namespace db_test {

//using namespace cudf;
//using namespace std;
//using namespace rmm;
//u/sing namespace rmm::mr;

/*
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

namespace cudf_io = cudf::experimental::io;

cudf::test::TempDirTestEnvironment* const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
    ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));
void json_test()
{  
  const std::string fname = temp_env->get_temp_dir() + "ArrowFileSource.csv";

  std::ofstream outfile(fname, std::ofstream::out);
  outfile << "[9]\n[8]\n[7]\n[6]\n[5]\n[4]\n[3]\n[2]\n";
  outfile.close();
  // ASSERT_TRUE(checkFile(fname));

  std::shared_ptr<arrow::io::ReadableFile> infile;
  ASSERT_TRUE(arrow::io::ReadableFile::Open(fname, &infile).ok());

  cudf_io::read_json_args in_args(cudf_io::source_info{infile});
  in_args.lines = true;
  in_args.dtype = {"int8"};
  cudf_io::table_with_metadata result = cudf_io::read_json(in_args);

  EXPECT_EQ(result.tbl->num_columns(), static_cast<cudf::size_type>(in_args.dtype.size()));
  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::INT8);

  auto validity = cudf::test::make_counting_transform_iterator(
      0, [](auto i) { return true; });
  
  // cudf::test::expect_columns_equal(result.tbl->get_column(0), int8_wrapper{{9, 8, 7, 6, 5, 4, 3, 2}, validity});   
  cudf::test::expect_columns_equal(result.tbl->get_column(0), int8_wrapper{{9, 8, 7, 6, 5, 4, 3, 2}, validity}); 
  int whee = 10;
  whee++;
}
*/

#include <cudf/io/functions.hpp>
/*
void parquet_writer_test()
{
  namespace cudf_io = cudf::experimental::io;

  srand(31337);
  auto table1 = create_random_int_table(5, 5, false);
  auto table2 = create_random_int_table(5, 5, false);

  srand(31337);
  auto full_table = create_random_int_table(5, 10, false);
      
  print_table(*table1);
  print_table(*table2);

  cudf_io::write_parquet_chunked_args args{cudf_io::sink_info{"whee"}};
  auto state = cudf_io::write_parquet_chunked_begin(args);  
  cudf_io::write_parquet_chunked(*table1, state);
  cudf_io::write_parquet_chunked(*table2, state);  
  cudf_io::write_parquet_chunked_end(state);  

  //cudf_io::write_parquet_args args{cudf_io::sink_info{"whee"}, *table1};
  //cudf_io::write_parquet(args);

  cudf_io::read_parquet_args read_args{cudf_io::source_info{"whee"}};
  auto result = cudf_io::read_parquet(read_args);
print_table(*result.tbl);
  // cudf::test::expect_tables_equal(*result.tbl, *full_table);  
}
*/

namespace cudf_io = cudf::experimental::io;

template <typename T>
using column_wrapper =
    typename std::conditional<std::is_same<T, cudf::string_view>::value,
                              cudf::test::strings_column_wrapper,
                              cudf::test::fixed_width_column_wrapper<T>>::type;
using column = cudf::column;
using table = cudf::experimental::table;
using table_view = cudf::table_view;

// Global environment for temporary files
auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
    ::testing::AddGlobalTestEnvironment(
        new cudf::test::TempDirTestEnvironment));

// Helper function to compare two tables
void whee(cudf::table_view const& lhs,
                         cudf::table_view const& rhs) {
  EXPECT_EQ(lhs.num_columns(), rhs.num_columns());
  auto expected = lhs.begin();
  auto result = rhs.begin();
  while (result != rhs.end()) {
    cudf::test::expect_columns_equal(*expected++, *result++);
  }
}

void parquet_writer_test()
{   
  using TypeParam = int;

  using T = TypeParam;  
     
  srand(31337);
  auto table1 = create_random_fixed_table<int>(4, 16, true);
  auto split_views = cudf::experimental::split(*table1, { 8 });

  print_table(*table1);
  print_table(split_views[0]);
  print_table(split_views[1]);
  
  cudf_io::write_parquet_args args{cudf_io::sink_info{"SlicedBad.parquet"}, split_views[1]};
  cudf_io::write_parquet(args);  

  cudf_io::read_parquet_args read_args{cudf_io::source_info{"SlicedBad.parquet"}};
  auto result = cudf_io::read_parquet(read_args);      

  cudf::test::expect_tables_equal(*result.tbl, split_views[1]);

  int whee = 10;
  whee++;
}

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
    CUDA_TRY(cudaMalloc(&src, sizeof(int) * num_els));
    int val = 2;
    for(int idx=0; idx<num_els; idx++){      
      cudaMemcpy(src + idx, &val, sizeof(int), cudaMemcpyHostToDevice);
    }
    CUDA_TRY(cudaMalloc(&dst, sizeof(int) * num_els));

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

// ------------------
//
// Work
//
// ------------------
/*
void whee()
{
    cudf::size_type start = 0;    
    auto valids = cudf::test::make_counting_transform_iterator(start, [](auto i) { return true; });    
    
    std::vector<std::string> strings[2]     = { {"", "this", "is", "a", "column", "of", "strings", "with", "in", "valid"}, 
                                                {"", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"} };
    
    std::vector<std::unique_ptr<cudf::column>> cols;   
    
    auto iter0 = cudf::test::make_counting_transform_iterator(0, [](auto i) { return (i);});
    auto c0 = cudf::test::fixed_width_column_wrapper<int>(iter0, iter0 + 10, valids);
    cols.push_back(c0.release());
    
    auto iter1 = cudf::test::make_counting_transform_iterator(10, [](auto i) { return (i);});
    auto c1 = cudf::test::fixed_width_column_wrapper<int>(iter1, iter1 + 10, valids);
    cols.push_back(c1.release());

    auto c2 = cudf::test::strings_column_wrapper(strings[0].begin(), strings[0].end(), valids);
    cols.push_back(c2.release());

    auto c3 = cudf::test::strings_column_wrapper(strings[1].begin(), strings[1].end(), valids);
    cols.push_back(c3.release());

    auto iter4 = cudf::test::make_counting_transform_iterator(20, [](auto i) { return (i);});
    auto c4 = cudf::test::fixed_width_column_wrapper<int>(iter4, iter4 + 10, valids);
    cols.push_back(c4.release());

    auto tbl = cudf::experimental::table(std::move(cols));
    
    std::vector<cudf::size_type> splits{5};

    auto result = cudf::experimental::contiguous_split(tbl, splits);
    auto expected = cudf::experimental::split(tbl, splits);
    
    for (unsigned long index = 0; index < expected.size(); index++) {      
      cudf::test::expect_tables_equal(expected[index], result[index].table);
    }    
}*/

int main()
{  
   // init stuff
   cuInit(0);   
   rmmOptions_t options; // {PoolAllocation, 0, false};   
   rmmInitialize(&options);

   // there's some "do stuff the first time" issues that cause bogus timings.
   // this function just flushes all that junk out
   // clear_baffles();

   // copy_if_else_test();
   // rolling_window_test();
   // timestamp_parse_test();
   // column_equal_test();
   // copy_if_else_scalar_test();
   //void large_split_tests();
   //large_split_tests();
   // void split_test();
   // split_test(); 
   // json_test();
   // crash_test();
   // parquet_writer_test();   
   // PQ_write((int64_t)3 * 1024 * 1024 * 1024, 8);

   // BM_contiguous_split_strings();
  
  /*
  size_t gran;
  CUmemAllocationProp prop;
  memset(&prop, 0, sizeof(CUmemAllocationProp));
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;  
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;  
  prop.location.id = device_id;
  cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  */

  // mem_tester();

  /*
  cudf_io::read_parquet_args read_args{cudf_io::source_info{"bad_table.parquet"}};
  auto result = cudf_io::read_parquet(read_args);
  //printf("%d\n", result.tbl->num_columns());
  //print_column(result.tbl->get_column(31), true, 10);

  auto parts = cudf::experimental::_contiguous_split(*result.tbl, { 4985 }, rmm::mr::get_default_resource() );
  printf("%d\n", parts[0].table.num_columns());
  print_column(parts[0].table.column(31), true, 10);
  */

  mem_tester();

  // shut stuff down
  rmmFinalize();
}