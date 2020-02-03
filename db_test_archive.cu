#include "db_test.cuh"

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

#endif   // copy if else

#if 0
void copy_if_else_test()
{         
   /*
   using T = int;

   // short one. < 1 warp/bitmask length
   int num_els = 4;

   bool mask[]    = { 1, 0, 0, 0 };
   bool_wrapper mask_w(mask, mask + num_els);

   T lhs[]        = { 5, 5, 5, 5 }; 
   bool lhs_v[]   = { 1, 1, 1, 1 };
   wrapper<T> lhs_w(lhs, lhs + num_els, lhs_v);

   T rhs[]        = { 6, 6, 6, 6 };
   bool rhs_v[]   = { 1, 1, 1, 1 };
   wrapper<T> rhs_w(rhs, rhs + num_els, rhs_v);
   
   T expected[]   = { 5, 6, 6, 6 };
   // bool exp_v[]   = { 1, 1, 1, 1 };
   wrapper<T> expected_w(expected, expected + num_els);

   auto out = cudf::experimental::copy_if_else(lhs_w, rhs_w, mask_w);
   print_column(*out);
   cudf::test::expect_columns_equal(out->view(), expected_w);
   */
   auto valids = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i % 2 == 0 ? true : false; });

   std::vector<const char*> h_string1{ "eee" };   
   cudf::string_scalar strings1{h_string1[0]};
   std::vector<const char*> h_strings2{ "zz",  "", "yyy", "w", "ééé", "ooo" };
   cudf::test::strings_column_wrapper strings2( h_strings2.begin(), h_strings2.end(), valids );   

   bool mask[] = { 1, 0, 1, 0, 1, 0 };
   bool_wrapper mask_w(mask, mask + 6);  
      
   auto results = cudf::experimental::copy_if_else(strings1, strings2, mask_w);
      
   std::vector<const char*> h_expected;
   for( cudf::size_type idx=0; idx < static_cast<cudf::size_type>(h_strings2.size()); ++idx )
   {
      if( mask[idx] ){
         h_expected.push_back( h_string1[0] );
      } else {
         h_expected.push_back( h_strings2[idx] );
      }
   }
   cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end(), valids);   
   print_column(*results);
   cudf::test::expect_columns_equal(*results,expected);;

   

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
#endif

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

// #endif // copy_if_else
#endif


#if 0

/**
 * @brief Copies contents of `in` to `out`.  Copies validity if present
 * but does not compute null count.
 *  
 * @param in column_view to copy from
 * @param out mutable_column_view to copy to.
 */
template <size_type block_size, typename T, bool has_validity>
__launch_bounds__(block_size)
__global__
void _copy_in_place_kernel( column_device_view const in,
                           size_type validity_size,
                           mutable_column_device_view out,
                           T val_subtract)
{
   const size_type tid = threadIdx.x + blockIdx.x * block_size;
   const int warp_id = tid / cudf::experimental::detail::warp_size;
   const size_type warps_per_grid = gridDim.x * block_size / cudf::experimental::detail::warp_size;      

   // begin/end indices for the column data
   size_type begin = 0;      
   size_type end = in.size();
   //if(tid == 0){ printf("end : %d\n", end); }
   size_type validity_end = validity_size;
   //if(tid == 0){ printf("validity_end : %d\n", validity_end); }
   //printf("end : %d\n", end);
   //printf("validity_end : %d\n", validity_end);
   // warp indices.  since 1 warp == 32 threads == sizeof(bit_mask_t) * 8,
   // each warp will process one (32 bit) of the validity mask via
   // __ballot_sync()
   size_type warp_begin = cudf::word_index(begin);
   //if(tid == 0){ printf("warp_begin : %d\n", warp_begin); }
   size_type warp_end = cudf::word_index(end-1);      
   //if(tid == 0){ printf("warp_end : %d\n", warp_begin); }

   // lane id within the current warp   
   const int lane_id = threadIdx.x % cudf::experimental::detail::warp_size;
   
   // current warp.
   size_type warp_cur = warp_begin + warp_id;   
   size_type index = tid;
   while(warp_cur <= warp_end){
      bool validity_in_range = (index >= begin && index < validity_end);
      bool valid = true;      
      if(has_validity){         
         valid = validity_in_range && in.is_valid(index);
      }

      bool in_range = (index >= begin && index < end);
      if(in_range){
         //printf("copy : %d\n", index);
         out.element<T>(index) = in.element<T>(index) - val_subtract;
      }
      
      // update validity      
      if(has_validity && validity_in_range){
         // the final validity mask for this warp 
         int warp_mask = __ballot_sync(0xFFFF'FFFF, valid && validity_in_range);
         // only one guy in the warp needs to update the mask and count
         if(lane_id == 0){            
            out.set_mask_word(warp_cur, warp_mask);            
         }
      }            

      // next grid
      warp_cur += warps_per_grid;
      index += block_size * gridDim.x;
   }      
}

// align all column size allocations to this boundary so that all output column buffers
// start at that alignment.
static constexpr size_t split_align = 64;

struct column_split_info {
   size_t   data_size;     // size of the data
   size_t   validity_size; // validity vector size
   
   size_t   offsets_size;  // (strings only) size of offset column
   size_t   chars_size;    // (strings only) # of chars in the column
   size_t   chars_offset;  // (strings only) offset from head of chars data
};

/**
 * @brief Functor called by the `type_dispatcher` to incrementally compute total
 * memory buffer size needed to allocate a contiguous copy of all columns within
 * a source table. 
 */
struct _column_buffer_size_functor {   
   template <typename T, std::enable_if_t<not is_fixed_width<T>()>* = nullptr>
   size_t operator()(column_view const& c, column_split_info &split_info)
   {
      // this has already been precomputed in an earlier step      
      return split_info.data_size + split_info.validity_size + split_info.offsets_size;
   }

   template <typename T, std::enable_if_t<is_fixed_width<T>()>* = nullptr>
   size_t operator()(column_view const& c, column_split_info &split_info)
   {      
      split_info.data_size = cudf::util::round_up_safe(c.size() * sizeof(T), split_align);  
      split_info.validity_size = (c.nullable() ? cudf::bitmask_allocation_size_bytes(c.size(), split_align) : 0);
      return split_info.data_size + split_info.validity_size;
   }
};

/**
 * @brief Functor called by the `type_dispatcher` to copy a column into a contiguous
 * buffer of output memory. 
 * 
 * Used for copying each column in a source table into one contiguous buffer of memory.
 */
struct _column_copy_functor {
   template <typename T, std::enable_if_t<not is_fixed_width<T>()>* = nullptr>
   void operator()(column_view const& in, column_split_info const& split_info, char*& dst, std::vector<column_view>& out_cols)
   {            
      strings_column_view strings_c(in);      

      // outgoing pointers
      char* chars_buf = dst;
      bitmask_type* validity_buf = split_info.validity_size == 0 ? nullptr : reinterpret_cast<bitmask_type*>(dst + split_info.data_size);
      char* offsets_buf = dst + split_info.data_size + split_info.validity_size;

      // increment working buffer
      dst += (split_info.data_size + split_info.validity_size + split_info.offsets_size);

      // 2 kernel calls. 1 to copy offsets and validity, and another to copy chars
      
      column_view offsets_col = strings_c.offsets();
      print_column(offsets_col);
      column_view _chars_col = strings_c.chars();
      print_column(_chars_col);
      
      // copy offsets and validity
      mutable_column_view temp_offsets_and_validity{
                              offsets_col.type(), offsets_col.size(), offsets_buf,
                              validity_buf, validity_buf == nullptr ? UNKNOWN_NULL_COUNT : 0,
                              0 };
      {         
         // contruct a column which wraps the validity vector and the offsets from the child column. 
         // this is weird but it removes an extra kernel call. however, since the length of the offsets column
         // is always 1 greater than the # of strings, the validity vector will be short by 1. the kernel will have to
         // compensate for that. 
         CUDF_EXPECTS(in.size() == offsets_col.size()-1, "Expected offsets column to be the same size as parent");
         CUDF_EXPECTS(in.offset() == offsets_col.offset(), "Expected offsets column offset to be the same as parent");
         CUDF_EXPECTS(offsets_col.type() == cudf::data_type(INT32), "Expected offsets column type to be int32");
         column_view in_offsets_and_validity{
                                 offsets_col.type(), offsets_col.size(), offsets_col.head<int32_t>(),
                                 in.null_mask(), in.null_mask() == nullptr ? UNKNOWN_NULL_COUNT : 0,
                                 in.offset()};
         
         cudf::size_type num_els = cudf::util::round_up_safe(strings_c.offsets().size(), cudf::experimental::detail::warp_size);
         constexpr int block_size = 256;
         cudf::experimental::detail::grid_1d grid{num_els, block_size, 1};         
         if(in.nullable()){
            _copy_in_place_kernel<block_size, size_type, true><<<grid.num_blocks, block_size, 0, 0>>>(
                              *column_device_view::create(in_offsets_and_validity), 
                              in.size(),  // validity vector length
                              *mutable_column_device_view::create(temp_offsets_and_validity), split_info.chars_offset);
         } else {
            _copy_in_place_kernel<block_size, size_type, false><<<grid.num_blocks, block_size, 0, 0>>>(
                              *column_device_view::create(in_offsets_and_validity),
                              in.size(),  // validity vector length
                              *mutable_column_device_view::create(temp_offsets_and_validity), split_info.chars_offset);
         }
      }

      // get the chars column directly instead of calling .chars(), since .chars() will end up
      // doing gpu work we specifically want to avoid.
      column_view chars_col = in.child(strings_column_view::chars_column_index);

      // copy chars
      mutable_column_view out_chars{chars_col.type(), static_cast<size_type>(split_info.chars_size), chars_buf};
      {         
         CUDF_EXPECTS(!chars_col.nullable(), "Expected input chars column to not be nullable");
         CUDF_EXPECTS(chars_col.offset() == 0, "Expected input chars column to have an offset of 0");
         column_view in_chars{ chars_col.type(), static_cast<size_type>(split_info.chars_size), chars_col.data<char>() + split_info.chars_offset };
                                 
         cudf::size_type num_els = cudf::util::round_up_safe(static_cast<size_type>(split_info.chars_size), cudf::experimental::detail::warp_size);
         constexpr int block_size = 256;
         cudf::experimental::detail::grid_1d grid{num_els, block_size, 1};         
         _copy_in_place_kernel<block_size, char, false><<<grid.num_blocks, block_size, 0, 0>>>(
                           *column_device_view::create(in_chars),
                           split_info.chars_size,
                           *mutable_column_device_view::create(out_chars), 0);         
      }

      // construct output string column_view.  offsets and validity have been glued together so
      // we have to rearrange things a bit.      
      column_view out_offsets{strings_c.offsets().type(), strings_c.offsets().size(), offsets_buf};
      
      out_cols.push_back(column_view(in.type(), in.size(), nullptr,
                                     validity_buf, UNKNOWN_NULL_COUNT, 0,
                                     { out_offsets, out_chars }));
                                    
                                    // out_cols.push_back({});
   }

   template <typename T, std::enable_if_t<is_fixed_width<T>()>* = nullptr>
   void operator()(column_view const& in, column_split_info const& split_info, char*& dst, std::vector<column_view>& out_cols)
   {     
      // outgoing pointers
      char* data = dst;
      bitmask_type* validity = split_info.validity_size == 0 ? nullptr : reinterpret_cast<bitmask_type*>(dst + split_info.data_size);

      // increment working buffer
      dst += (split_info.data_size + split_info.validity_size);

      // custom copy kernel (which should probably just be an in-place copy() function in cudf.
      cudf::size_type num_els = cudf::util::round_up_safe(in.size(), cudf::experimental::detail::warp_size);
      constexpr int block_size = 256;
      cudf::experimental::detail::grid_1d grid{num_els, block_size, 1};
      
      // so there's a significant performance issue that comes up. our incoming column_view objects
      // are the result of a slice.  because of this, they have an UNKNOWN_NULL_COUNT.  because of that,
      // calling column_device_view::create() will cause a recompute of the count, which ends up being
      // extremely slow because a.) the typical use case here will involve huge numbers of calls and
      // b.) the count recompute involves tons of device allocs and memcopies.
      //
      // so to get around this, I am manually constructing a fake-ish view here where the null
      // count is arbitrarily bashed to 0.            
      //            
      // Remove this hack once rapidsai/cudf#3600 is fixed.
      column_view   in_wrapped{in.type(), in.size(), in.head<T>(), 
                               in.null_mask(), in.null_mask() == nullptr ? UNKNOWN_NULL_COUNT : 0,
                               in.offset() };
      mutable_column_view  mcv{in.type(), in.size(), data, 
                               validity, validity == nullptr ? UNKNOWN_NULL_COUNT : 0 };      
      if(in.nullable()){               
         _copy_in_place_kernel<block_size, T, true><<<grid.num_blocks, block_size, 0, 0>>>(
                           *column_device_view::create(in_wrapped), 
                           in.size(),
                           *mutable_column_device_view::create(mcv), 0);         
      } else {
         _copy_in_place_kernel<block_size, T, false><<<grid.num_blocks, block_size, 0, 0>>>(
                           *column_device_view::create(in_wrapped), 
                           in.size(),
                           *mutable_column_device_view::create(mcv), 0);
      }
      mcv.set_null_count(cudf::UNKNOWN_NULL_COUNT);                 

      out_cols.push_back(mcv);
   }
};

template <typename S>
__device__ inline S round_up_safe_nothrow(S number_to_round, S modulus) {
    auto remainder = number_to_round % modulus;
    if (remainder == 0) { return number_to_round; }
    auto rounded_up = number_to_round - remainder + modulus;    
    return rounded_up;
}

// Computes required allocation size of a bitmask
__device__ std::size_t bitmask_allocation_size_bytes_nothrow(size_type number_of_bits,
                                          std::size_t padding_boundary) {  
  auto necessary_bytes =
      cudf::util::div_rounding_up_safe<size_type>(number_of_bits, CHAR_BIT);

  auto padded_bytes =
      padding_boundary * cudf::util::div_rounding_up_safe<size_type>(
                             necessary_bytes, padding_boundary);
  return padded_bytes;
}

/**
 * @brief Creates a contiguous_split_result object which contains a deep-copy of the input
 * table_view into a single contiguous block of memory. 
 * 
 * The table_view contained within the contiguous_split_result will pass an expect_tables_equal()
 * call with the input table.  The memory referenced by the table_view and its internal column_views
 * is entirely contained in single block of memory.
 */
contiguous_split_result _alloc_and_copy(cudf::table_view const& t, rmm::mr::device_memory_resource* mr, cudaStream_t stream)
{      
   /*
   // offsets for this column
   for(size_t idx=0; idx<(size_t)t.num_columns(); idx++){
      size_type whee[256] = {0};
      column_device_view ccco = offset_columns[idx];
      cudaMemcpy(whee, ccco.data<int32_t>(), ccco.size() * sizeof(size_type), cudaMemcpyDeviceToHost);
      
      printf("col %d (size : %d)\n", (size_type)idx, (size_type)ccco.size());
      for(size_type oidx=0; oidx<ccco.size(); oidx++){      
         printf("%d ", whee[oidx]);
      }
      printf("\n");
      printf("col.head : %d\n", whee[ccco.offset()]);
      printf("col.data : %d\n", whee[0]);
   }
   */

   /*
   // bring it back to the cpu
   thrust::host_vector<size_type> string_lengths = device_string_lengths;   
   printf("yay : ");
   for(size_t idx=0; idx<string_lengths.size(); idx++){
      printf("%d ", string_lengths[idx]);
   }
   printf("\n");

   int whee = 10;
   whee++;
   */  

   // preprocess column sizes for string columns.  the idea here is this:
   // - determining string lengths involves reaching into device memory to look at offsets, which is slow.
   // - contiguous_split() is typically used in situations with very large numbers of output columns, magnifying the slowness.
   // - so rather than reaching into device memory once per column (in column_buffer_size_functor), 
   //   we are doing it once per split.  For an example case of a table with 512 columns split 256 ways, that reduces
   //   our number of trips to/from the gpu from 128k -> 256

   // build a list of all the offset columns and their indices for all input string columns and put them on the gpu
   //
   // i'm using this pair structure instead of thrust::tuple because using tuple somehow causes the cudf::column_device_view
   // default constructor to get called (compiler error) when doing the assignment to device_offset_columns below
   thrust::host_vector<thrust::pair<thrust::pair<size_type, bool>, cudf::column_device_view>> offset_columns;
   offset_columns.reserve(t.num_columns());  // worst case
   size_type column_index = 0;
   std::for_each(t.begin(), t.end(), [&offset_columns, &column_index](cudf::column_view const& c){
      if(c.type().id() == STRING){
         // constructing device view from the offsets column only, because doing so for the entire
         // strings_column_view will result in memory allocation/cudaMemcpy() calls, which would
         // defeat the whole purpose of this step.
         cudf::column_device_view cdv((strings_column_view(c)).offsets(), 0, 0);
         offset_columns.push_back(thrust::pair<thrust::pair<size_type, bool>, cudf::column_device_view>(
                  thrust::pair<size_type, bool>(column_index, c.nullable()), cdv));
      }
      column_index++;
   });   
   thrust::device_vector<thrust::pair<thrust::pair<size_type, bool>, cudf::column_device_view>> device_offset_columns = offset_columns;   

   // compute column sizes for all string columns
   thrust::device_vector<column_split_info> device_split_info(device_offset_columns.size());   
   auto *sizes_p = device_split_info.data().get();   
   thrust::for_each(rmm::exec_policy(stream)->on(stream), device_offset_columns.begin(), device_offset_columns.end(),
      [sizes_p] __device__ (auto column_info){
         size_type                  col_index = column_info.first.first;
         bool                       include_validity = column_info.first.second;
         cudf::column_device_view   col = column_info.second;
         size_type                  num_elements = col.size()-1;

         size_t align = split_align;

         auto num_chars = col.data<int32_t>()[num_elements] - col.data<int32_t>()[0];         
         sizes_p[col_index].data_size = round_up_safe_nothrow(static_cast<size_t>(num_chars), align);         
         // can't use cudf::bitmask_allocation_size_bytes() because it throws
         sizes_p[col_index].validity_size = include_validity ? bitmask_allocation_size_bytes_nothrow(num_elements, align) : 0;                  
         // can't use cudf::util::round_up_safe() because it throws
         sizes_p[col_index].offsets_size = round_up_safe_nothrow(col.size() * sizeof(size_type), align);
         sizes_p[col_index].chars_size = num_chars;
         sizes_p[col_index].chars_offset = col.data<int32_t>()[0];
      }
   );
   // copy sizes back from gpu. entries from non-string columns are uninitialized at this point.
   thrust::host_vector<column_split_info> split_info = device_split_info;
     
   // compute the rest of the column sizes (non-string columns, and total buffer size)
   size_t total_size = 0;
   column_index = 0;
   std::for_each(t.begin(), t.end(), [&total_size, &column_index, &split_info](cudf::column_view const& c){   
      total_size += cudf::experimental::type_dispatcher(c.type(), _column_buffer_size_functor{}, c, split_info[column_index]);
      column_index++;
   });

   /*   
   for(size_t idx=0; idx<column_sizes.size(); idx++){      
      printf("col %d,  (%d, %d, %d)\n", (int)idx, (int)thrust::get<0>(column_sizes[idx]), (int)thrust::get<1>(column_sizes[idx]), (int)thrust::get<2>(column_sizes[idx]));
      print_column(t.column(idx));
   }
   printf("Total size : %d\n", (int)total_size);
   */

   // allocate
   auto device_buf = std::make_unique<rmm::device_buffer>(total_size, stream, mr);
   char *buf = static_cast<char*>(device_buf->data());

   // copy (this would be cleaner with a std::transform, but there's an nvcc compiler issue in the way)   
   std::vector<column_view> out_cols;
   out_cols.reserve(t.num_columns());
   column_index = 0;   
   std::for_each(t.begin(), t.end(), [&out_cols, &buf, &column_index, &split_info](cudf::column_view const& c){
      cudf::experimental::type_dispatcher(c.type(), _column_copy_functor{}, c, split_info[column_index], buf, out_cols);
      column_index++;
   });   
   
   return contiguous_split_result{cudf::table_view{out_cols}, std::move(device_buf)};   
}

}; // anonymous namespace

std::vector<contiguous_split_result> _contiguous_split(cudf::table_view const& input,
                                                      std::vector<size_type> const& splits,
                                                      rmm::mr::device_memory_resource* mr,
                                                      cudaStream_t stream)
{          
   auto subtables = cudf::experimental::split(input, splits);

/*
   for(int idx=0; idx<(int)subtables.size(); idx++){
      auto st = subtables[idx];
      auto col = st.column(0);      

      printf("table : %d\n", idx);
      
      auto offsets_col = col.child(0);
      auto chars_col = col.child(1);      
      printf("parent\nsize : %d\noffset : %d\n", col.size(), col.offset());
      printf("offsets\n");
      printf("   size : %d\n", offsets_col.size());
      printf("   offset : %d\n", offsets_col.offset());
      printf("   data (head) : %llx\n", (long long unsigned int) offsets_col.head());
      printf("   data (data) : %llx\n", (long long unsigned int) offsets_col.data<int32_t>());

      printf("chars\n");
      printf("   size : %d\n", chars_col.size());
      printf("   offset : %d\n", chars_col.offset());
      printf("   data (head) : %llx\n", (long long unsigned int) chars_col.head());
      printf("   data (data) : %llx\n", (long long unsigned int) chars_col.data<int32_t>());

      strings_column_view sv(col);
      auto offsets_col_s = sv.offsets();
      auto chars_col_s = sv.chars();      
      printf("offsets S\n");
      printf("   size : %d\n", offsets_col_s.size());
      printf("   offset : %d\n", offsets_col_s.offset());
      printf("   data (head) : %llx\n", (long long unsigned int) offsets_col_s.head());
      printf("   data (data) : %llx\n", (long long unsigned int) offsets_col_s.data<int32_t>());

      printf("chars S\n");
      printf("   size : %d\n", chars_col_s.size());
      printf("   offset : %d\n", chars_col_s.offset());
      printf("   data (head) : %llx\n", (long long unsigned int) chars_col_s.head());
      printf("   data (data) : %llx\n", (long long unsigned int) chars_col_s.data<int32_t>());
      
      printf("\n\n");
   }
   */   

   std::vector<contiguous_split_result> result;
   int idx = 0;
   std::transform(subtables.begin(), subtables.end(), std::back_inserter(result), [mr, stream, &idx](table_view const& t) { 
      idx++;
      return _alloc_and_copy(t, mr, stream);
   });

   return result;
}

}; // namespace detail

std::vector<contiguous_split_result> _contiguous_split(cudf::table_view const& input,
                                                      std::vector<size_type> const& splits,
                                                      rmm::mr::device_memory_resource* mr)
{    
   return cudf::experimental::detail::_contiguous_split(input, splits, mr, (cudaStream_t)0);   
}

#endif

#if 0
namespace cudf {

namespace experimental {

namespace detail {

namespace {

using namespace::cudf;
using namespace::cudf::experimental;

template <typename S>
__device__ inline S round_up_safe_nothrow(S number_to_round, S modulus) {
    auto remainder = number_to_round % modulus;
    if (remainder == 0) { return number_to_round; }
    auto rounded_up = number_to_round - remainder + modulus;    
    return rounded_up;
}

// Computes required allocation size of a bitmask
__device__ std::size_t bitmask_allocation_size_bytes_nothrow(size_type number_of_bits,
                                          std::size_t padding_boundary) {  
  auto necessary_bytes =
      cudf::util::div_rounding_up_safe<size_type>(number_of_bits, CHAR_BIT);

  auto padded_bytes =
      padding_boundary * cudf::util::div_rounding_up_safe<size_type>(
                             necessary_bytes, padding_boundary);
  return padded_bytes;
}


/**
 * @brief Copies contents of `in` to `out`.  Copies validity if present
 * but does not compute null count.
 *  
 * @param in column_view to copy from
 * @param out mutable_column_view to copy to.
 */
template <size_type block_size, typename T, bool has_validity>
__launch_bounds__(block_size)
__global__
void copy_in_place_kernel( column_device_view const in,
                           mutable_column_device_view out)
{
   const size_type tid = threadIdx.x + blockIdx.x * block_size;
   const int warp_id = tid / cudf::experimental::detail::warp_size;
   const size_type warps_per_grid = gridDim.x * block_size / cudf::experimental::detail::warp_size;      

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
         }
      }            

      // next grid
      warp_cur += warps_per_grid;
      index += block_size * gridDim.x;
   }      
}

template <size_type block_size, bool has_validity>
__launch_bounds__(block_size)
__global__
void copy_in_place_strings_kernel(size_type                        num_rows,
                                  size_type const* __restrict__    offsets_in,
                                  size_type* __restrict__          offsets_out,
                                  size_type                        validity_in_offset,
                                  bitmask_type const* __restrict__ validity_in,
                                  bitmask_type* __restrict__       validity_out,

                                  size_type                        offset_shift,

                                  size_type                        num_chars,
                                  char const* __restrict__         chars_in,
                                  char* __restrict__               chars_out)
{   
   const size_type tid = threadIdx.x + blockIdx.x * block_size;
   const int warp_id = tid / cudf::experimental::detail::warp_size;
   const size_type warps_per_grid = gridDim.x * block_size / cudf::experimental::detail::warp_size;   
   
   // how many warps we'll be processing. with strings, the chars and offsets
   // lengths may be different.  so we'll just march the worst case.
   size_type warp_begin = cudf::word_index(0);
   size_type warp_end = cudf::word_index(std::max(num_chars, num_rows+1)-1);

   // end indices for chars   
   size_type chars_end = num_chars;
   // end indices for offsets   
   size_type offsets_end = num_rows+1;
   // end indices for validity and the last warp that actually should
   // be updated
   size_type validity_end = num_rows;
   size_type validity_warp_end = cudf::word_index(num_rows-1);  

   // lane id within the current warp   
   const int lane_id = threadIdx.x % cudf::experimental::detail::warp_size;

   size_type warp_cur = warp_begin + warp_id;
   size_type index = tid;
   while(warp_cur <= warp_end){      
      if(index < chars_end){
         chars_out[index] = chars_in[index];
      }
      
      if(index < offsets_end){
         // each output column starts at a new base pointer. so we have to
         // shift every offset down by the point (in chars) at which it was split.
         offsets_out[index] = offsets_in[index] - offset_shift;
      }

      // if we're still in range of validity at all
      if(has_validity && warp_cur <= validity_warp_end){
         bool valid = (index < validity_end) && bit_is_set(validity_in, validity_in_offset + index);
               
         // the final validity mask for this warp 
         int warp_mask = __ballot_sync(0xFFFF'FFFF, valid);
         // only one guy in the warp needs to update the mask and count
         if(lane_id == 0){                        
            validity_out[warp_cur] = warp_mask;
         }
      }            

      // next grid
      warp_cur += warps_per_grid;
      index += block_size * gridDim.x;
   }    
}

// align all column size allocations to this boundary so that all output column buffers
// start at that alignment.
static constexpr size_t split_align = 64;

struct column_split_info {
   size_type   data_size;     // size of the data
   size_type   validity_size; // validity vector size
   
   size_type   offsets_size;  // (strings only) size of offset column
   size_type   chars_size;    // (strings only) # of chars in the column
   size_type   chars_offset;  // (strings only) offset from head of chars data
};

/**
 * @brief Functor called by the `type_dispatcher` to incrementally compute total
 * memory buffer size needed to allocate a contiguous copy of all columns within
 * a source table. 
 */
struct column_buffer_size_functor {
   template <typename T, std::enable_if_t<not is_fixed_width<T>()>* = nullptr>
   size_t operator()(column_view const& c, column_split_info &split_info)
   {
      // this has already been precomputed in an earlier step      
      return split_info.data_size + split_info.validity_size + split_info.offsets_size;
   }

   template <typename T, std::enable_if_t<is_fixed_width<T>()>* = nullptr>
   size_t operator()(column_view const& c, column_split_info &split_info)
   {      
      split_info.data_size = cudf::util::round_up_safe(c.size() * sizeof(T), split_align);  
      split_info.validity_size = (c.nullable() ? cudf::bitmask_allocation_size_bytes(c.size(), split_align) : 0);
      return split_info.data_size + split_info.validity_size;
   }
};

/**
 * @brief Functor called by the `type_dispatcher` to copy a column into a contiguous
 * buffer of output memory. 
 * 
 * Used for copying each column in a source table into one contiguous buffer of memory.
 */
struct column_copy_functor {
   template <typename T, std::enable_if_t<not is_fixed_width<T>()>* = nullptr>
   void operator()(column_view const& in, column_split_info const& split_info, char*& dst, std::vector<column_view>& out_cols)
   {            
      // outgoing pointers
      char* chars_buf = dst;
      bitmask_type* validity_buf = split_info.validity_size == 0 ? nullptr : reinterpret_cast<bitmask_type*>(dst + split_info.data_size);
      size_type* offsets_buf = reinterpret_cast<size_type*>(dst + split_info.data_size + split_info.validity_size);

      // increment working buffer
      dst += (split_info.data_size + split_info.validity_size + split_info.offsets_size);

      // offsets column
      strings_column_view strings_c(in);
      column_view in_offsets = strings_c.offsets();
      // get the chars column directly instead of calling .chars(), since .chars() will end up
      // doing gpu work we specifically want to avoid.
      column_view in_chars = in.child(strings_column_view::chars_column_index);      
      
      // 1 combined kernel call that copies chars, offsets and validity in one pass
      cudf::size_type num_els = cudf::util::round_up_safe(std::max(split_info.chars_size, in_offsets.size() + 1)/*strings_c.offsets().size()*/, cudf::experimental::detail::warp_size);
      constexpr int block_size = 256;
      cudf::experimental::detail::grid_1d grid{num_els, block_size, 1};            
      if(in.nullable()){
         copy_in_place_strings_kernel<block_size, true><<<grid.num_blocks, block_size, 0, 0>>>(
                           in.size(),                                            // num_rows
                           in_offsets.data<size_type>(),                         // offsets_in
                           offsets_buf,                                          // offsets_out
                           in.offset(),                                          // validity_in_offset
                           in.null_mask(),                                       // validity_in
                           validity_buf,                                         // validity_out

                           split_info.chars_offset,                              // offset_shift

                           split_info.chars_size,                                // num_chars
                           in_chars.head<char>() + split_info.chars_offset,      // chars_in
                           chars_buf);                                                      
      } else {                                       
         copy_in_place_strings_kernel<block_size, false><<<grid.num_blocks, block_size, 0, 0>>>(
                           in.size(),                                            // num_rows
                           in_offsets.data<size_type>(),                         // offsets_in
                           offsets_buf,                                          // offsets_out
                           0,                                                    // validity_in_offset
                           nullptr,                                              // validity_in
                           nullptr,                                              // validity_out

                           split_info.chars_offset,                              // offset_shift

                           split_info.chars_size,                                // num_chars
                           in_chars.head<char>() + split_info.chars_offset,      // chars_in
                           chars_buf);                                                      
      }      

      // output child columns      
      column_view out_offsets{strings_c.offsets().type(), strings_c.offsets().size(), offsets_buf};
      column_view out_chars{in_chars.type(), static_cast<size_type>(split_info.chars_size), chars_buf};      

      // result
      out_cols.push_back(column_view(in.type(), in.size(), nullptr,
                                     validity_buf, UNKNOWN_NULL_COUNT, 0,
                                     { out_offsets, out_chars }));                                     

      /*
      strings_column_view strings_c(in);

      // outgoing pointers
      char* chars_buf = dst;
      bitmask_type* validity_buf = split_info.validity_size == 0 ? nullptr : reinterpret_cast<bitmask_type*>(dst + split_info.data_size);
      size_type* offsets_buf = reinterpret_cast<size_type*>(dst + split_info.data_size + split_info.validity_size);

      // increment working buffer
      dst += (split_info.data_size + split_info.validity_size + split_info.offsets_size);
                                     
      // copy offsets and validity
      column_view offsets_col = strings_c.offsets();
      mutable_column_view temp_offsets_and_validity{
                              offsets_col.type(), offsets_col.size(), offsets_buf,
                              validity_buf, validity_buf == nullptr ? UNKNOWN_NULL_COUNT : 0,
                              0 };
      {         
         // contruct a column which wraps the validity vector and the offsets from the child column. 
         // this is weird but it removes an extra kernel call. however, since the length of the offsets column
         // is always 1 greater than the # of strings, the validity vector will be short by 1. the kernel will have to
         // compensate for that. 
         CUDF_EXPECTS(in.size() == offsets_col.size()-1, "Expected offsets column to be the same size as parent");
         CUDF_EXPECTS(in.offset() == offsets_col.offset(), "Expected offsets column offset to be the same as parent");
         CUDF_EXPECTS(offsets_col.type() == cudf::data_type(INT32), "Expected offsets column type to be int32");
         column_view in_offsets_and_validity{
                                 offsets_col.type(), offsets_col.size(), offsets_col.head<int32_t>(),
                                 in.null_mask(), in.null_mask() == nullptr ? UNKNOWN_NULL_COUNT : 0,
                                 in.offset()};
         
         cudf::size_type num_els = cudf::util::round_up_safe(strings_c.offsets().size(), cudf::experimental::detail::warp_size);
         constexpr int block_size = 256;
         cudf::experimental::detail::grid_1d grid{num_els, block_size, 1};
         if(in.nullable()){
            copy_in_place_kernel<block_size, size_type, true><<<grid.num_blocks, block_size, 0, 0>>>(
                              *column_device_view::create(in_offsets_and_validity), 
                              in.size(),  // validity vector length
                              *mutable_column_device_view::create(temp_offsets_and_validity), split_info.chars_offset);
         } else {
            copy_in_place_kernel<block_size, size_type, false><<<grid.num_blocks, block_size, 0, 0>>>(
                              *column_device_view::create(in_offsets_and_validity),
                              in.size(),  // validity vector length
                              *mutable_column_device_view::create(temp_offsets_and_validity), split_info.chars_offset);
         }
      }

      // get the chars column directly instead of calling .chars(), since .chars() will end up
      // doing gpu work we specifically want to avoid.
      column_view chars_col = in.child(strings_column_view::chars_column_index);

      // copy chars
      mutable_column_view out_chars{chars_col.type(), static_cast<size_type>(split_info.chars_size), chars_buf};      
      {         
         CUDF_EXPECTS(!chars_col.nullable(), "Expected input chars column to not be nullable");
         CUDF_EXPECTS(chars_col.offset() == 0, "Expected input chars column to have an offset of 0");
         column_view in_chars{ chars_col.type(), static_cast<size_type>(split_info.chars_size), chars_col.data<char>() + split_info.chars_offset };
                                 
         cudf::size_type num_els = cudf::util::round_up_safe(static_cast<size_type>(split_info.chars_size), cudf::experimental::detail::warp_size);
         constexpr int block_size = 256;
         cudf::experimental::detail::grid_1d grid{num_els, block_size, 1};         
         copy_in_place_kernel<block_size, char, false><<<grid.num_blocks, block_size, 0, 0>>>(
                           *column_device_view::create(in_chars),
                           split_info.chars_size,
                           *mutable_column_device_view::create(out_chars), 0);
      }
      */
   }

   template <typename T, std::enable_if_t<is_fixed_width<T>()>* = nullptr>
   void operator()(column_view const& in, column_split_info const& split_info, char*& dst, std::vector<column_view>& out_cols)
   {     
      // outgoing pointers
      char* data = dst;
      bitmask_type* validity = split_info.validity_size == 0 ? nullptr : reinterpret_cast<bitmask_type*>(dst + split_info.data_size);

      // increment working buffer
      dst += (split_info.data_size + split_info.validity_size);

      // custom copy kernel (which should probably just be an in-place copy() function in cudf.
      cudf::size_type num_els = cudf::util::round_up_safe(in.size(), cudf::experimental::detail::warp_size);
      constexpr int block_size = 256;
      cudf::experimental::detail::grid_1d grid{num_els, block_size, 1};
      
      // so there's a significant performance issue that comes up. our incoming column_view objects
      // are the result of a slice.  because of this, they have an UNKNOWN_NULL_COUNT.  because of that,
      // calling column_device_view::create() will cause a recompute of the count, which ends up being
      // extremely slow because a.) the typical use case here will involve huge numbers of calls and
      // b.) the count recompute involves tons of device allocs and memcopies.
      //
      // so to get around this, I am manually constructing a fake-ish view here where the null
      // count is arbitrarily bashed to 0.            
      //            
      // Remove this hack once rapidsai/cudf#3600 is fixed.
      column_view   in_wrapped{in.type(), in.size(), in.head<T>(), 
                               in.null_mask(), in.null_mask() == nullptr ? UNKNOWN_NULL_COUNT : 0,
                               in.offset() };
      mutable_column_view  mcv{in.type(), in.size(), data, 
                               validity, validity == nullptr ? UNKNOWN_NULL_COUNT : 0 };      
      if(in.nullable()){               
         copy_in_place_kernel<block_size, T, true><<<grid.num_blocks, block_size, 0, 0>>>(
                           *column_device_view::create(in_wrapped),                            
                           *mutable_column_device_view::create(mcv));         
      } else {
         copy_in_place_kernel<block_size, T, false><<<grid.num_blocks, block_size, 0, 0>>>(
                           *column_device_view::create(in_wrapped),                            
                           *mutable_column_device_view::create(mcv));
      }
      mcv.set_null_count(cudf::UNKNOWN_NULL_COUNT);                 

      out_cols.push_back(mcv);
   }
};

/**
 * @brief Creates a contiguous_split_result object which contains a deep-copy of the input
 * table_view into a single contiguous block of memory. 
 * 
 * The table_view contained within the contiguous_split_result will pass an expect_tables_equal()
 * call with the input table.  The memory referenced by the table_view and its internal column_views
 * is entirely contained in single block of memory.
 */
contiguous_split_result alloc_and_copy(cudf::table_view const& t, thrust::device_vector<column_split_info>& device_split_info, rmm::mr::device_memory_resource* mr, cudaStream_t stream)      
{            
   // preprocess column sizes for string columns.  the idea here is this:
   // - determining string lengths involves reaching into device memory to look at offsets, which is slow.
   // - contiguous_split() is typically used in situations with very large numbers of output columns, exaggerating
   //   the problem.
   // - so rather than reaching into device memory once per column (in column_buffer_size_functor), 
   //   we are doing it once per split (for all string columns in the split).  For an example case of a table with 
   //   512 columns split 256 ways, that reduces our number of trips to/from the gpu from 128k -> 256
   
   // build a list of all the offset columns and their indices for all input string columns and put them on the gpu
   //
   // i'm using this pair structure instead of thrust::tuple because using tuple somehow causes the cudf::column_device_view
   // default constructor to get called (compiler error) when doing the assignment to device_offset_columns below
   thrust::host_vector<thrust::pair<thrust::pair<size_type, bool>, cudf::column_device_view>> offset_columns;
   offset_columns.reserve(t.num_columns());  // worst case
   size_type column_index = 0;
   std::for_each(t.begin(), t.end(), [&offset_columns, &column_index](cudf::column_view const& c){
      if(c.type().id() == STRING){
         // constructing device view from the offsets column only, because doing so for the entire
         // strings_column_view will result in memory allocation/cudaMemcpy() calls, which would
         // defeat the whole purpose of this step.
         cudf::column_device_view cdv((strings_column_view(c)).offsets(), 0, 0);
         offset_columns.push_back(thrust::pair<thrust::pair<size_type, bool>, cudf::column_device_view>(
                  thrust::pair<size_type, bool>(column_index, c.nullable()), cdv));
      }
      column_index++;
   });   
   thrust::device_vector<thrust::pair<thrust::pair<size_type, bool>, cudf::column_device_view>> device_offset_columns = offset_columns;   
     
   // compute column sizes for all string columns   
   auto *sizes_p = device_split_info.data().get();   
   thrust::for_each(rmm::exec_policy(stream)->on(stream), device_offset_columns.begin(), device_offset_columns.end(),
      [sizes_p] __device__ (auto column_info){
         size_type                  col_index = column_info.first.first;
         bool                       include_validity = column_info.first.second;
         cudf::column_device_view   col = column_info.second;
         size_type                  num_elements = col.size()-1;

         size_t align = split_align;

         auto num_chars = col.data<int32_t>()[num_elements] - col.data<int32_t>()[0];
         sizes_p[col_index].data_size = round_up_safe_nothrow(static_cast<size_t>(num_chars), align);         
         // can't use cudf::bitmask_allocation_size_bytes() because it throws
         sizes_p[col_index].validity_size = include_validity ? bitmask_allocation_size_bytes_nothrow(num_elements, align) : 0;
         // can't use cudf::util::round_up_safe() because it throws
         sizes_p[col_index].offsets_size = round_up_safe_nothrow(col.size() * sizeof(size_type), align);
         sizes_p[col_index].chars_size = num_chars;
         sizes_p[col_index].chars_offset = col.data<int32_t>()[0];
      }
   );
   
   // copy sizes back from gpu. entries from non-string columns are uninitialized at this point.
   thrust::host_vector<column_split_info> split_info = device_split_info;     
     
   // compute the rest of the column sizes (non-string columns, and total buffer size)
   size_t total_size = 0;
   column_index = 0;
   std::for_each(t.begin(), t.end(), [&total_size, &column_index, &split_info](cudf::column_view const& c){   
      total_size += cudf::experimental::type_dispatcher(c.type(), column_buffer_size_functor{}, c, split_info[column_index]);
      column_index++;
   });

   // allocate
   auto device_buf = std::make_unique<rmm::device_buffer>(total_size, stream, mr);
   char *buf = static_cast<char*>(device_buf->data());

   // copy (this would be cleaner with a std::transform, but there's an nvcc compiler issue in the way)   
   std::vector<column_view> out_cols;
   out_cols.reserve(t.num_columns());
   column_index = 0;   
   std::for_each(t.begin(), t.end(), [&out_cols, &buf, &column_index, &split_info](cudf::column_view const& c){
      cudf::experimental::type_dispatcher(c.type(), column_copy_functor{}, c, split_info[column_index], buf, out_cols);
      column_index++;
   });   
   
   return contiguous_split_result{cudf::table_view{out_cols}, std::move(device_buf)};   
}

}; // anonymous namespace

std::vector<contiguous_split_result> _contiguous_split(cudf::table_view const& input,
                                                      std::vector<size_type> const& splits,
                                                      rmm::mr::device_memory_resource* mr,
                                                      cudaStream_t stream)
{   
   auto subtables = cudf::experimental::split(input, splits);

   // optimization : for large #'s of splits this allocation can dominate total time
   //                spent if done inside alloc_and_copy().  so we'll allocate it once
   //                and reuse it.
   // 
   //                benchmark:        1 GB data, 10 columns, 256 splits.
   //                no optimization:  106 ms (8 GB/s)
   //                optimization:     20 ms (48 GB/s)
   thrust::device_vector<column_split_info> device_split_info(input.num_columns());

   std::vector<contiguous_split_result> result;
   std::transform(subtables.begin(), subtables.end(), std::back_inserter(result), [mr, stream, &device_split_info](table_view const& t) { 
      return alloc_and_copy(t, device_split_info, mr, stream);
   });

   return result;
}

}; // namespace detail

std::vector<contiguous_split_result> _contiguous_split(cudf::table_view const& input,
                                                      std::vector<size_type> const& splits,
                                                      rmm::mr::device_memory_resource* mr)
{    
   return cudf::experimental::detail::_contiguous_split(input, splits, mr, (cudaStream_t)0);   
}

}; // namespace experimental

}; // namespace cudf

#endif

using namespace cudf;
using namespace cudf::experimental;

namespace {
std::vector<cudf::size_type> splits_to_indices(std::vector<cudf::size_type> splits, cudf::size_type size){
    std::vector<cudf::size_type> indices{0};

    std::for_each(splits.begin(), splits.end(),
            [&indices](auto split) {
                indices.push_back(split); // This for end
                indices.push_back(split); // This for the start
            });

    if (splits.back() != size) {
        indices.push_back(size); // This to include rest of the elements
    } else {
        indices.pop_back(); // Not required as it is extra 
    }

    return indices;
}

void verify_split_results( cudf::experimental::table const& src_table, 
                           std::vector<contiguous_split_result> const &dst_tables,
                           std::vector<size_type> const& splits,
                           int verbosity = 0)
{     
   table_view src_v(src_table.view());
   
   int col_count = 0;
   for(size_t c_idx = 0; c_idx<(size_t)src_v.num_columns(); c_idx++){
      // grab this column from each subtable
      auto src_subcols = cudf::experimental::split(src_v.column(c_idx), splits);

      for(size_t t_idx=0; t_idx<src_subcols.size(); t_idx++){
         cudf::test::expect_columns_equal(src_subcols[t_idx], dst_tables[t_idx].table.column(c_idx), true);
         
         if(verbosity > 0 && (col_count % verbosity == 0)){
            printf("----------------------------\n");            
            print_column(src_subcols[t_idx], false, 20);
            print_column(dst_tables[t_idx].table.column(c_idx), false, 20);
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

int rand_range(int r)
{
   return static_cast<int>((static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * (float)(r-1));
}

template<typename T>
void single_split_test_common(std::vector<T>& src_cols, 
                              int64_t num_cols, 
                              int64_t num_rows, 
                              int64_t num_splits,
                              int verbosity)
{
   scope_timer_manual null_count_gen("null count gen");
   null_count_gen.start();

   std::vector<std::unique_ptr<column>> columns((size_t)num_cols);
   std::transform(src_cols.begin(), src_cols.end(), columns.begin(), [](T& in){   
      auto ret = in.release();
      ret->null_count();
      return ret;
   });
   null_count_gen.end();

   cudf::experimental::table src_table(std::move(columns));   
   // print_table(src_table);
   printf("# columns : %d\n", (int)num_cols);
   
   // generate splits 
   int split_stride = num_rows / num_splits;
   std::vector<size_type> splits;  
   scope_timer_manual split_gen("split gen");
   split_gen.start();
   for(size_t idx=0; idx<(size_t)num_rows; idx+=split_stride){      
      splits.push_back(min((int)(idx + split_stride), (int)num_rows));
   }
   split_gen.end();
     
   // do the split
   scope_timer_manual split_time("contiguous_split total");
   split_time.start();   
   auto dst_tables = cudf::experimental::contiguous_split(src_table.view(), splits, rmm::mr::get_default_resource());
   cudaDeviceSynchronize();
   split_time.end();

   scope_timer_manual verify_time("verify_split_results");
   verify_time.start();
   verify_split_results(src_table, dst_tables, splits, verbosity);
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

void single_split_test( int64_t total_desired_bytes, 
                        int64_t num_cols,                     
                        int64_t num_rows,
                        int64_t num_splits,
                        bool include_validity)
{
   printf("total data size : %.2f GB\n", (float)total_desired_bytes / (float)(1024 * 1024 * 1024));
   
   srand(31337);

   // generate input columns and table      
   scope_timer_manual src_table_gen("src table gen");   
   src_table_gen.start();

   auto valids = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i%2 == 0 ? true : false; });
   std::vector<cudf::test::fixed_width_column_wrapper<int>> src_cols(num_cols);
   for(int idx=0; idx<num_cols; idx++){
      auto rand_elements = cudf::test::make_counting_transform_iterator(0, [](int i){return rand();});
      if(include_validity){
         src_cols[idx] = cudf::test::fixed_width_column_wrapper<int>(rand_elements, rand_elements + num_rows, valids);
      } else {
         src_cols[idx] = cudf::test::fixed_width_column_wrapper<int>(rand_elements, rand_elements + num_rows);
      }
   }      
   src_table_gen.end();

   single_split_test_common(src_cols, num_cols, num_rows, num_splits, 1000);   
}

void single_split_test_string( int64_t total_desired_bytes,
                               int64_t num_cols,                        
                               int64_t num_splits,
                               bool include_validity)
{     
   printf("total data size : %.2f GB\n", (float)total_desired_bytes / (float)(1024 * 1024 * 1024));
   
   srand(31337);

   // generate input columns and table      
   scope_timer_manual src_table_gen("src table gen");   
   src_table_gen.start();
   
   // const int64_t string_len[8] = { 8, 4, 5, 7, 2, 3, 8, 6 };
   const int64_t avg_string_len = 6;   // eh. don't really need to hit the # of bytes exactly. just ballpark
   std::vector<const char*> h_strings{ "aaaaaaaa", "b", "ccccc", "ddddddd", "ee", "fff", "gggggggg", "hhhhhh" };   
   
   int64_t col_len_bytes = total_desired_bytes / num_cols;
   int64_t num_rows = col_len_bytes / avg_string_len;      
      
   // generate table
   auto valids = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i%2 == 0 ? true : false; });   
   std::vector<cudf::test::strings_column_wrapper> src_cols;
   {
      std::vector<const char*> one_col(num_rows);
      for(int64_t idx=0; idx<num_cols; idx++){
         // fill in a random set of strings
         for(int64_t s_idx=0; s_idx<num_rows; s_idx++){
            one_col[s_idx] = h_strings[rand_range(h_strings.size())];         
         }
         if(include_validity){
            src_cols.push_back(cudf::test::strings_column_wrapper(one_col.begin(), one_col.end(), valids));
         } else {
            src_cols.push_back(cudf::test::strings_column_wrapper(one_col.begin(), one_col.end()));
         }
      }
   }
   src_table_gen.end();

   single_split_test_common(src_cols, num_cols, num_rows, num_splits, 1000);
}

};

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
   
   /*
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
      int64_t total_desired_bytes = (int64_t)1 * (1024 * 1024 * 1024);
      int64_t num_cols = 512;      
      int64_t num_rows = total_desired_bytes / (num_cols * el_size);
      int64_t num_splits = num_cols / 2;
      single_split_test(total_desired_bytes, num_cols, num_rows, num_splits, true);
   }   
   */ 
      
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
   }
   */
   
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
      int64_t total_desired_bytes = (int64_t)2 * (1024 * 1024 * 1024);
      int64_t num_cols = 10;
      int64_t num_rows = total_desired_bytes / (num_cols * el_size);
      int64_t num_splits = 256;
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
      int64_t total_desired_bytes = (int64_t)4 * (1024 * 1024 * 1024);      
      // int64_t total_desired_bytes = (int64_t)(512 * 1024 * 1024);
      int64_t num_cols = 512;            
      int64_t num_splits = num_cols / 2;
      single_split_test_string(total_desired_bytes, num_cols, num_splits, true);
   }   
}

inline std::vector<cudf::experimental::table> create_expected_string_tables(std::vector<std::string> const strings[2], std::vector<cudf::size_type> const& indices, bool nullable) {

    std::vector<cudf::experimental::table> result = {};

    for(unsigned long index = 0; index < indices.size(); index+=2) {
        std::vector<std::unique_ptr<cudf::column>> cols = {};
        
        for(int idx=0; idx<2; idx++){     
            if(not nullable) {
                cudf::test::strings_column_wrapper wrap(strings[idx].begin()+indices[index], strings[idx].begin()+indices[index+1]);                
                cols.push_back(wrap.release());
            } else {
                auto valids = cudf::test::make_counting_transform_iterator(indices[index], [](auto i) { return i % 2 == 0 ? true : false; });
                cudf::test::strings_column_wrapper wrap(strings[idx].begin()+indices[index], strings[idx].begin()+indices[index+1], valids);
                cols.push_back(wrap.release());
            }
        }

        result.push_back(cudf::experimental::table(std::move(cols)));
    }

    return result;
}

std::vector<cudf::experimental::table> create_expected_string_tables_for_splits(std::vector<std::string> const strings[2], std::vector<cudf::size_type> const& splits, bool nullable){    
    std::vector<cudf::size_type> indices = splits_to_indices(splits, strings[0].size());    
    return create_expected_string_tables(strings, indices, nullable);
}

std::unique_ptr<column> make_strings(std::vector<const char*> _strings)
{   
   cudf::test::strings_column_wrapper strings( _strings.begin(), _strings.end(),
        thrust::make_transform_iterator( _strings.begin(), [] (auto str) { return str!=nullptr; }));
      
   return strings.release();
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

   std::vector<size_type> splits { 5, 10 };

   auto out = contiguous_split(t.view(), splits);
   
   for(size_t idx=0; idx<out.size(); idx++){
      print_table(out[idx].table);
   }
   
   int whee = 10;
   whee++;
   */
   
   /*
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

   std::vector<size_type> splits { 5, 10 };

   auto out = cudf::experimental::contiguous_split(t.view(), splits, rmm::mr::get_default_resource());
   
   //verify_split_results(t, out, splits, true);           
   
   size_t num_out_tables = out.size();
   for(size_t idx=0; idx<num_out_tables; idx++){
      print_table(out[idx].table);
   }   

   int whee = 10;
   whee++; 
   */  

   /*         
   int num_els = 3;
   int c0[] = { 0, 1, 2 };
   wrapper<int> c0_w(c0, c0 + num_els);
   std::vector<size_type> splits { 0, 2, 1, 3 };
      
   auto out = cudf::experimental::slice(c0_w, splits);

   for(size_t idx=0; idx<out.size(); idx++){
      print_column(out[idx]);      
   }
   */     
   
   /*
   auto c = make_strings( {"1", "2", "3", "4", "5" } );
   column_view cv = c->view();   

   std::vector<size_type> ssplits { 1, 5 };
   auto sout = cudf::experimental::slice(cv, ssplits);
   
   for(size_t idx=0; idx<sout.size(); idx++){
      print_column(sout[idx]);      
   } 

   int whee = 10;
   whee++;   
   */
              
    auto valids = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i%2 == 0 ? true : false; });   
    std::vector<std::string> strings[2]     = { {"this", "is", "a", "column", "of", "strings"}, 
                                                {"one", "two", "three", "four", "five", "six"} };
    cudf::test::strings_column_wrapper sw[2] = { {strings[0].begin(), strings[0].end(), valids},
                                                 {strings[1].begin(), strings[1].end(), valids} };

    std::vector<std::unique_ptr<cudf::column>> scols;
    scols.push_back(sw[0].release());
    scols.push_back(sw[1].release());
    cudf::experimental::table src_table(std::move(scols));

    std::vector<cudf::size_type> splits{2};
    
    std::vector<cudf::experimental::table> expected = create_expected_string_tables_for_splits(strings, splits, true);

    auto result = cudf::experimental::contiguous_split(src_table, splits, rmm::mr::get_default_resource());
    
    EXPECT_EQ(expected.size(), result.size());

    for (unsigned long index = 0; index < result.size(); index++) {       
        {
         
         printf("---------------------\n");
         print_table(expected[index]);
         print_table(result[index].table);
         printf("---------------------\n");
         cudf::test::expect_tables_equal(expected[index], result[index].table);
      }
    } 
    

   /*
    auto valids = cudf::test::make_counting_transform_iterator(0, [](auto i) { return true; });
    std::vector<std::string> strings = { "this", "is", "a", "column", "of", "strings", "with", "in", "valid" };
    cudf::test::strings_column_wrapper sw = { strings.begin(), strings.end(), valids };
    auto scols = sw.release();    
   
    std::vector<cudf::size_type> splits{2, 5, 9};        

    std::vector<cudf::column_view> result = cudf::experimental::split(*scols, splits);

   size_t idx;
   for(idx=0; idx<result.size(); idx++){
      cudf::test::print(result[idx]); std::cout << "\n";      
   }
   */
}