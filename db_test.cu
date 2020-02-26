#include "db_test.cuh"


#if 0    // skeleton for working in cudf
//namespace db_test {

//using namespace cudf;
//using namespace std;
//using namespace rmm;
//u/sing namespace rmm::mr;

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

int main()
{  
   // init stuff
   cuInit(0);   
   rmmOptions_t options{PoolAllocation, 0, false};   
   rmmInitialize(&options);   
   
   // void large_split_tests();
   // large_split_tests();
   // void split_test();
   // split_test(); 
   // json_test();   
   // parquet_writer_test();    
   // custom_sink_example();   

   // shut stuff down
   rmmFinalize();
}