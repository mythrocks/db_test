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
/*
#include <cudf/io/functions.hpp>
void parquet_writer_test()
{
  namespace cudf_io = cudf::experimental::io;

  srand(31337);
  auto table1 = create_random_int_table(5, 5, false);
  auto table2 = create_random_int_table(5, 5, false);
      
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

// ------------------
//
// Work
//
// ------------------

int main()
{
   // init stuff
   cuInit(0);   
   rmmOptions_t options{PoolAllocation, 0, false};   
   rmmInitialize(&options);

   // there's some "do stuff the first time" issues that cause bogus timings.
   // this function just flushes all that junk out
   // clear_baffles();

   // copy_if_else_test();
   // rolling_window_test();
   // timestamp_parse_test();
   // column_equal_test();
   // copy_if_else_scalar_test();
   // void large_split_tests();
   // large_split_tests();
   void split_test();
   split_test(); 
   // json_test();
   // crash_test();
   // parquet_writer_test();   

   // BM_contiguous_split_strings();

    // shut stuff down
   rmmFinalize();
}