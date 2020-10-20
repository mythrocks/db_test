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

   auto out = cudf::copy_if_else(lhs_w, rhs_w, mask_w);
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
      
   auto results = cudf::copy_if_else(strings1, strings2, mask_w);
      
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
   auto filter = [bool_mask_device] __device__ (cudf::size_type i) { return bool_mask_device.element<cudf::bool8>(i); };

   return cudf::detail::copy_if_else(lhs, rhs, filter, mr, stream);
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
      
   auto out = cudf::copy_if_else(lhs_w, rhs_w, mask_w);
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

   auto out = cudf::copy_if_else(lhs_w, rhs_w, mask_w);
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

      auto out = cudf::copy_if_else(lhs_w, rhs_w, mask_w);     
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

      auto out = cudf::copy_if_else(lhs_w, rhs_w, mask_w);     
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

      auto out = cudf::copy_if_else(lhs_w, rhs_w, mask_w);     
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

      auto out = cudf::copy_if_else(lhs_w, rhs_w, mask_w);     
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
   const int warp_id = tid / cudf::detail::warp_size;
   const size_type warps_per_grid = gridDim.x * block_size / cudf::detail::warp_size;      

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
   const int lane_id = threadIdx.x % cudf::detail::warp_size;
   
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
         
         cudf::size_type num_els = cudf::util::round_up_safe(strings_c.offsets().size(), cudf::detail::warp_size);
         constexpr int block_size = 256;
         cudf::detail::grid_1d grid{num_els, block_size, 1};         
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
                                 
         cudf::size_type num_els = cudf::util::round_up_safe(static_cast<size_type>(split_info.chars_size), cudf::detail::warp_size);
         constexpr int block_size = 256;
         cudf::detail::grid_1d grid{num_els, block_size, 1};         
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
      cudf::size_type num_els = cudf::util::round_up_safe(in.size(), cudf::detail::warp_size);
      constexpr int block_size = 256;
      cudf::detail::grid_1d grid{num_els, block_size, 1};
      
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
      total_size += cudf::type_dispatcher(c.type(), _column_buffer_size_functor{}, c, split_info[column_index]);
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
      cudf::type_dispatcher(c.type(), _column_copy_functor{}, c, split_info[column_index], buf, out_cols);
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
   auto subtables = cudf::split(input, splits);

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
   return cudf::detail::_contiguous_split(input, splits, mr, (cudaStream_t)0);   
}

#endif

#if 0
namespace cudf {

namespace experimental {

namespace detail {

namespace {

using namespace::cudf;

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
   const int warp_id = tid / cudf::detail::warp_size;
   const size_type warps_per_grid = gridDim.x * block_size / cudf::detail::warp_size;      

   // begin/end indices for the column data
   size_type begin = 0;
   size_type end = in.size();
   // warp indices.  since 1 warp == 32 threads == sizeof(bit_mask_t) * 8,
   // each warp will process one (32 bit) of the validity mask via
   // __ballot_sync()
   size_type warp_begin = cudf::word_index(begin);
   size_type warp_end = cudf::word_index(end-1);      

   // lane id within the current warp   
   const int lane_id = threadIdx.x % cudf::detail::warp_size;
   
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
   const int warp_id = tid / cudf::detail::warp_size;
   const size_type warps_per_grid = gridDim.x * block_size / cudf::detail::warp_size;   
   
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
   const int lane_id = threadIdx.x % cudf::detail::warp_size;

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
      cudf::size_type num_els = cudf::util::round_up_safe(std::max(split_info.chars_size, in_offsets.size() + 1)/*strings_c.offsets().size()*/, cudf::detail::warp_size);
      constexpr int block_size = 256;
      cudf::detail::grid_1d grid{num_els, block_size, 1};            
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
         
         cudf::size_type num_els = cudf::util::round_up_safe(strings_c.offsets().size(), cudf::detail::warp_size);
         constexpr int block_size = 256;
         cudf::detail::grid_1d grid{num_els, block_size, 1};
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
                                 
         cudf::size_type num_els = cudf::util::round_up_safe(static_cast<size_type>(split_info.chars_size), cudf::detail::warp_size);
         constexpr int block_size = 256;
         cudf::detail::grid_1d grid{num_els, block_size, 1};         
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
      cudf::size_type num_els = cudf::util::round_up_safe(in.size(), cudf::detail::warp_size);
      constexpr int block_size = 256;
      cudf::detail::grid_1d grid{num_els, block_size, 1};
      
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
      total_size += cudf::type_dispatcher(c.type(), column_buffer_size_functor{}, c, split_info[column_index]);
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
      cudf::type_dispatcher(c.type(), column_copy_functor{}, c, split_info[column_index], buf, out_cols);
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
   auto subtables = cudf::split(input, splits);

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
   return cudf::detail::_contiguous_split(input, splits, mr, (cudaStream_t)0);   
}

}; // namespace experimental

}; // namespace cudf

#endif

#if 0
using namespace cudf;

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

void verify_split_results( cudf::table const& src_table, 
                           std::vector<contiguous_split_result> const &dst_tables,
                           std::vector<size_type> const& splits,
                           int verbosity = 0)
{     
   table_view src_v(src_table.view());
   
   int col_count = 0;
   for(size_t c_idx = 0; c_idx<(size_t)src_v.num_columns(); c_idx++){
      // grab this column from each subtable
      auto src_subcols = cudf::split(src_v.column(c_idx), splits);

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

   cudf::table src_table(std::move(columns));   
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
   auto dst_tables = cudf::contiguous_split(src_table.view(), splits, rmm::mr::get_default_resource());
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
   // auto valids = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i%2 == 0 ? true : false; });   
   auto valids = cudf::test::make_counting_transform_iterator(0, [](auto i) { return rand() < (RAND_MAX/2) ? true : false; /*(i%2 == 0 ? true : false; */});   
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

inline std::vector<cudf::table> create_expected_string_tables(std::vector<std::string> const strings[2], std::vector<cudf::size_type> const& indices, bool nullable) {

    std::vector<cudf::table> result = {};

    for(unsigned long index = 0; index < indices.size(); index+=2) {
        std::vector<std::unique_ptr<cudf::column>> cols = {};
        
        for(int idx=0; idx<2; idx++){     
            if(not nullable) {
                cudf::test::strings_column_wrapper wrap(strings[idx].begin()+indices[index], strings[idx].begin()+indices[index+1]);                
                cols.push_back(wrap.release());
            } else {
                // auto valids = cudf::test::make_counting_transform_iterator(indices[index], [](auto i) { return i % 2 == 0 ? true : false; });
                auto valids = cudf::test::make_counting_transform_iterator(indices[index], [](auto i) { return true; });
                cudf::test::strings_column_wrapper wrap(strings[idx].begin()+indices[index], strings[idx].begin()+indices[index+1], valids);
                cols.push_back(wrap.release());
            }
        }

        result.push_back(cudf::table(std::move(cols)));
    }

    return result;
}

std::vector<cudf::table> create_expected_string_tables_for_splits(std::vector<std::string> const strings[2], std::vector<cudf::size_type> const& splits, bool nullable){    
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
   
   cudf::table t(std::move(columns));
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
   
   cudf::table t(std::move(columns));
   print_table(t.view());

   std::vector<size_type> splits { 5 };

   auto out = cudf::contiguous_split(t.view(), splits, rmm::mr::get_default_resource());
   
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
      
   auto out = cudf::slice(c0_w, splits);

   for(size_t idx=0; idx<out.size(); idx++){
      print_column(out[idx]);      
   }
   */     
   
   /*
   auto c = make_strings( {"1", "2", "3", "4", "5" } );
   column_view cv = c->view();   

   std::vector<size_type> ssplits { 1, 5 };
   auto sout = cudf::slice(cv, ssplits);
   
   for(size_t idx=0; idx<sout.size(); idx++){
      print_column(sout[idx]);      
   } 

   int whee = 10;
   whee++;   
   */    
    // auto valids = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i%2 == 0 ? true : false; });
    //auto valids = cudf::test::make_counting_transform_iterator(0, [](auto i) { return true; });
    std::vector<std::string> strings[2]     = { {"this", "is", "a", "column", "of", "strings"}, 
                                                {"one", "two", "three", "four", "five", "six"} };
    cudf::test::strings_column_wrapper sw[2] = { {strings[0].begin(), strings[0].end(), /*valids*/},
                                                 {strings[1].begin(), strings[1].end(), /*valids*/} };

    std::vector<std::unique_ptr<cudf::column>> scols;
    scols.push_back(sw[0].release());
    scols.push_back(sw[1].release());
    cudf::table src_table(std::move(scols));

    std::vector<cudf::size_type> splits{-1};
    
    //std::vector<cudf::table> expected = create_expected_string_tables_for_splits(strings, splits, true);

    auto result = cudf::contiguous_split(src_table, splits, rmm::mr::get_default_resource());
    
    //EXPECT_EQ(expected.size(), result.size());

    for (unsigned long index = 0; index < result.size(); index++) {       
      {
        //bool a = expected[index].get_column(0).nullable();
        bool b = result[index].table.column(0).nullable();

        printf("---------------------\n");
        //print_table(expected[index]);
        print_table(result[index].table);
        printf("---------------------\n");
        // cudf::test::expect_tables_equal(expected[index], result[index].table);
        // cudf::test::expect_tables_equivalent(expected[index], result[index].table);
      }
    }

    int whee = 10;
    whee++;

   /*
    auto valids = cudf::test::make_counting_transform_iterator(0, [](auto i) { return true; });
    std::vector<std::string> strings = { "this", "is", "a", "column", "of", "strings", "with", "in", "valid" };
    cudf::test::strings_column_wrapper sw = { strings.begin(), strings.end(), valids };
    auto scols = sw.release();    
   
    std::vector<cudf::size_type> splits{2, 5, 9};        

    std::vector<cudf::column_view> result = cudf::split(*scols, splits);

   size_t idx;
   for(idx=0; idx<result.size(); idx++){
      cudf::test::print(result[idx]); std::cout << "\n";      
   }
   */
}
#endif

/*
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
*/


/*
void parquet_writer_test()
{   
  using TypeParam = int;

  using T = TypeParam;  
     
  srand(31337);
  auto table1 = create_random_fixed_table<int>(4, 16, true);
  auto split_views = cudf::split(*table1, { 8 });

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
*/

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

    auto tbl = cudf::table(std::move(cols));
    
    std::vector<cudf::size_type> splits{5};

    auto result = cudf::contiguous_split(tbl, splits);
    auto expected = cudf::split(tbl, splits);
    
    for (unsigned long index = 0; index < expected.size(); index++) {      
      cudf::test::expect_tables_equal(expected[index], result[index].table);
    }    
}*/

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
/*
void PQ_write()
{
    int64_t total_desired_bytes = (int64_t)3 * 1024 * 1024 * 1024;
    cudf::size_type num_cols = 1024;
    cudf::size_type num_tables = 64;

    cudf::size_type el_size = 4;
    int64_t num_rows = (total_desired_bytes / (num_cols * el_size)) / num_tables;

    srand(31337);
    std::vector<std::unique_ptr<cudf::table>> tables;        
    for(cudf::size_type idx=0; idx<num_tables; idx++){
      tables.push_back(create_random_fixed_table<int>(num_cols, num_rows, true));
    }

    //for(auto _ : state){
        //cuda_event_timer raii(state, true); // flush_l2_cache = true, stream = 0            
        printf("WRITE BEGIN\n");
        cudf_io::write_parquet_chunked_args args{cudf_io::sink_info()};

        auto state = cudf_io::write_parquet_chunked_begin(args);
        std::for_each(tables.begin(), tables.end(), [&state](std::unique_ptr<cudf::table> const& tbl){
          cudf_io::write_parquet_chunked(*tbl, state);
        });
        cudf_io::write_parquet_chunked_end(state);
    //}

    //state.SetBytesProcessed(
      //  static_cast<int64_t>(state.iterations())*state.range(0));
}
*/

/*
namespace cudf_io = cudf::io;

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

/*
#include <cudf/io/functions.hpp>
void parquet_writer_test()
{
  namespace cudf_io = cudf::io;

  srand(31337);
  auto table1 = create_random_fixed_table<int>(5, 5, false);
  auto table2 = create_random_fixed_table<int>(5, 5, false);

  srand(31337);
  auto full_table = create_random_fixed_table<int>(5, 10, false);
      
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
  //cudf::test::expect_tables_equal(*result.tbl, *full_table);  
}
*/

/*
namespace cudf_io = cudf::io;

template <typename T>
using column_wrapper =
    typename std::conditional<std::is_same<T, cudf::string_view>::value,
                              cudf::test::strings_column_wrapper,
                              cudf::test::fixed_width_column_wrapper<T>>::type;
using column = cudf::column;
using table = cudf::table;
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

void pq()
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(512, 4096, true);
  auto table2 = create_random_fixed_table<int>(512, 8192, true);
  
  auto full_table = cudf::concatenate({*table1, *table2});          

  auto filepath = temp_env->get_temp_filepath("ChunkedLarge.parquet");
  cudf_io::write_parquet_chunked_args args{cudf_io::sink_info{filepath}};
  auto state = cudf_io::write_parquet_chunked_begin(args);  
  cudf_io::write_parquet_chunked(*table1, state);
  cudf_io::write_parquet_chunked(*table2, state);  
  cudf_io::write_parquet_chunked_end(state);    

  cudf_io::read_parquet_args read_args{cudf_io::source_info{filepath}};
  auto result = cudf_io::read_parquet(read_args);
  
  cudf::test::expect_tables_equal(*result.tbl, *full_table);  
}
*/




/*
class custom_data_sink : public cudf::io::data_sink {
public:
  explicit custom_data_sink(){
    printf("CUSTOM DATA SINK CONSTRUCTOR\n");
  }

  virtual ~custom_data_sink() {
    printf("CUSTOM DATA SINK DESTRUCTOR\n");
  }

  void write(void const* data, size_t size) override {
    printf("CUSTOM DATA SINK WRITE : %lu, %lu\n", reinterpret_cast<int64_t>(data), size);
  }

  bool supports_gpu_write(){
    return true;
  }

  void write_gpu(void const* gpu_data, size_t size){        
    printf("CUSTOM DATA SINK GPU WRITE : %lu, %lu\n", reinterpret_cast<int64_t>(gpu_data), size);
  }

  void flush() override {
    printf("CUSTOM DATA SINK FLUSH\n");  
  }

  size_t bytes_written() override {
    printf("CUSTOM DATA SINK BYTES WRITTEN\n");  
    return 0;
  }
};

void custom_sink_example()
{
  custom_data_sink custom_sink;

  namespace cudf_io = cudf::io;

  srand(31337);
  auto table1 = create_random_fixed_table<int>(5, 5, false);  

  // custom_sink lives across multiple write_parquet() calls
  {
    cudf_io::write_parquet_args args{cudf_io::sink_info{&custom_sink}, *table1};  
    cudf_io::write_parquet(args);
  }
  {
    cudf_io::write_parquet_args args{cudf_io::sink_info{&custom_sink}, *table1};  
    cudf_io::write_parquet(args);
  }
  {
    cudf_io::write_parquet_args args{cudf_io::sink_info{&custom_sink}, *table1};  
    cudf_io::write_parquet(args);
  }  
}
*/

/*
#include <cudf/io/functions.hpp>
void parquet_bug()
{
   using namespace cudf;   

   cudf::io::read_parquet_args read_args{cudf::io::source_info{"nullstrings.parquet"}};
   auto result = cudf::io::read_parquet(read_args);
   int num_columns = result.tbl->num_columns();
   column_view col = result.tbl->get_column(0);
   int null_count = result.tbl->get_column(0).null_count();
   print_table(*result.tbl);
      
   int mask_bytes = bitmask_allocation_size_bytes(col.size());
   int mask_size = mask_bytes / 4;
   bitmask_type *host_mask = new bitmask_type[mask_size];   
   cudaMemcpy(host_mask, col.null_mask(), mask_bytes, cudaMemcpyDeviceToHost);
   int idx;
   for(idx=0; idx<col.size(); idx++){
      if(!bit_is_set(host_mask, idx)){
         printf("NULL at %d\n", idx);
      }
   }   

   int whee = 10;
   whee++;
}
*/

/*
void sequence_test()
{
   auto result = cudf:::sequence(0, cudf::numeric_scalar<float>(10), 
                                                 cudf::numeric_scalar<float>(-6));
                                                    
   print_column(*result);  

   int whee = 10;
   whee++;
}
*/

struct UShiftRight {
    template <typename TypeOut, typename TypeLhs, typename TypeRhs>
    static TypeOut operate(TypeLhs x, TypeRhs y) {
        return static_cast<typename std::make_unsigned<TypeLhs>::type>(x) >> y;
    }
};

void shift_tests()
{
  using T = int;
  int num_els = 4;

  uint32_t a = (static_cast<std::make_unsigned_t<uint32_t>>(-8) >> 1);
  printf("%u\n", a);
  uint32_t b = (static_cast<std::make_unsigned_t<uint32_t>>(78) >> 1);
  printf("%u\n", b);
  uint32_t c = (static_cast<std::make_unsigned_t<uint32_t>>(-93) >> 3);
  printf("%u\n", c);
  uint32_t d = (static_cast<std::make_unsigned_t<uint32_t>>(0) >> 2);
  printf("%u\n", d);
  uint32_t e = (static_cast<std::make_unsigned_t<uint32_t>>(-INT_MAX) >> 16);
  printf("%u\n", e);

  /*
  T lhs[]   = { -4, -4, -4, -4 };  
  wrapper<T> lhs_w(lhs, lhs + num_els);

  T shift[]   = { 1, 1, 1, 1 };  
  wrapper<T> shift_w(shift, shift + num_els);
  
  T expected[]   = { 8, 8, 8, 8 };  
  wrapper<T> expected_w(expected, expected + num_els);

  auto out = cudf::binary_operation(
      lhs_w, shift_w, cudf::binary_operator::SHIFT_LEFT,
      cudf::data_type(cudf::type_to_id<T>()));

  print_column(lhs_w);
  print_column(expected_w);
  print_column(*out);

  auto outright = cudf::binary_operation(
      *out, shift_w, cudf::binary_operator::SHIFT_RIGHT_UNSIGNED,
      cudf::data_type(cudf::type_to_id<T>()));

  print_column(*outright);  
  */

  int whee = 10;
  whee++;
}

/*
enum code_type { 
   UNKNOWN = -1,
   UPPERCASE,
   LOWERCASE,
   TITLECASE,   
};

struct code_data {
   code_type   type;

   int         lower;   
   int         title[3];
   int         title_len;
   int         upper[3];
   int         upper_len;   
};
code_data Code_data[65535];

void parse_special_casing()
{
   FILE *f = fopen("SpecialCasingUnconditional.txt", "rt");   

   char line[1024];
   while(!feof(f)){
      fgets(line, 1023, f);      
      if(line[0] == '#'){
         continue;
      }
      const char *tok = strtok(line, ";");
      if(tok == NULL || tok[0] == '\0' || tok[0] == '\n'){
         continue;
      }

      int code;
      sscanf(tok, "%x", &code);
      code_data *cd = &Code_data[code];
            
      int lower;
      tok = strtok(NULL, ";");
      sscanf(tok, "%x", &lower);
      if(lower != cd->lower){
         printf("Found lowercase mismatch at %x\n", code);
      }
      
      tok = strtok(NULL, ";");
      int title[3];
      int title_len = sscanf(tok, "%x %x %x", &title[0], &title[1], &title[2]);
      if(cd->title_len > 0 && ((title_len != cd->title_len || title[0] != cd->title[0]))){
         printf("Found titlecase mismatch at %x\n", code);
      }
      cd->title_len = title_len; cd->title[0] = title[0]; cd->title[1] = title[1]; cd->title[2] = title[2];

      tok = strtok(NULL, ";");
      int upper[3];
      int upper_len = sscanf(tok, "%x %x %x", &upper[0], &upper[1], &upper[2]);
      if(cd->upper_len > 0 && ((upper_len != cd->upper_len || upper[0] != cd->upper[0]))){
         // printf("Found uppercase mismatch at %x\n", code);
      }
      cd->upper_len = upper_len; cd->upper[0] = upper[0]; cd->upper[1] = upper[1]; cd->upper[2] = upper[2];                    
   }
   fclose(f);  
}

void parse_regular_casing()
{
   FILE *f = fopen("SimpleCasing.txt", "rt");   

   char line[1024];
   while(!feof(f)){
      fgets(line, 1023, f);      
      if(line[0] == '#'){
         continue;
      }

      code_type type = UNKNOWN;
      int code;
      int lower = 0;      
      int title[3] = { 0 };
      int title_len = 0;
      int upper[3] = { 0 };
      int upper_len = 0;

      const char *start = line;
      const char *end = strchr(start, ';');
      int field = 0;
      while(1){
         switch(field){
         case 0:
            sscanf(start, "%x", &code);
            break;
         case 2:
            if(!strncmp(start, "Lu", 2)){
               type = UPPERCASE;
            } else if(!strncmp(start, "Ll", 2)){
               type = LOWERCASE;
            } else if(!strncmp(start, "Lt", 2)){
               type = TITLECASE;
            }
            break;
         case 12:
            upper_len = sscanf(start, "%x %x %x", &upper[0], &upper[1], &upper[2]);
            break;
         case 13:
            sscanf(start, "%x", &lower);
            break;         
         case 14:
            title_len = sscanf(start, "%x %x %x", &title[0], &title[1], &title[2]);
            break;
         default:
            break;
         }

         if(code == 0x1b0){
            int whee = 10;
            whee++;
         }         

         if(end == nullptr){
            break;
         }               

         field++;
         start = end+1;

         if(start[0] == '\n'){
            break;
         }

         end = strchr(start, ';');
      }

      // not a codepoint we care about
      if(type == UNKNOWN){
         continue;
      }

      if(code > 65535){
         continue;         
      }          

      code_data *cd = &Code_data[code];
      cd->type = type;
      switch(cd->type){
      case UPPERCASE:
         if(lower == 0){
            //printf("Uppercase codepoint with no lowercase : %x\n", code);
         } else {
            cd->lower = lower;
            cd->title[0] = title[0]; cd->title[1] = title[1]; cd->title[2] = title[2];
            cd->title_len = title_len;
         }         
         break;
      case LOWERCASE:
         if(upper_len == 0){
            //printf("Lowercase codepoint with no uppercase : %x\n", code);
         } else { 
            cd->upper[0] = upper[0]; cd->upper[1] = upper[1]; cd->upper[2] = upper[2];
            cd->upper_len = upper_len;
            cd->title[0] = title[0]; cd->title[1] = title[1]; cd->title[2] = title[2];
            cd->title_len = title_len;
         }
         // set lowercase to be myself
         cd->lower = code;
         break;
      case TITLECASE:
         cd->lower = lower;
         cd->upper[0] = upper[0]; cd->upper[1] = upper[1]; cd->upper[2] = upper[2];
         cd->upper_len = upper_len;
         break;
      default:
         break;
      }      
      
      int whee = 10;
      whee++;
   }
   fclose(f);
}

#include <strings/char_types/char_cases.h>
void parse_unicode_stuff()
{
   int idx;
   for(idx=0; idx<65535; idx++){
      Code_data[idx].type = UNKNOWN;
   }
   parse_regular_casing();
   parse_special_casing();   
}
*/

// #include <cudf/strings/char_types/char_cases.hpp>
// cudf::strings::detail::generate_special_mapping_hash_table();

/*
#include <cudf/binaryop.hpp>
#include <tests/binaryop/assert-binops.h>
void binop_test()
{      
  using TypeOut = int32_t;
  using TypeLhs = float;
  using TypeRhs = float;

  using ADD = cudf::library::operation::Add<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = cudf::test::fixed_width_column_wrapper<float> { 1.3f, 1.6f };
  auto rhs = cudf::test::fixed_width_column_wrapper<float> { 1.3f, 1.6f };
    
  auto out = cudf::binary_operation(
      lhs, rhs, cudf::binary_operator::ADD,
      cudf::data_type(cudf::type_to_id<TypeOut>()));

   cudf::test::print(*out);   

  float x = 1.6f;
  float y = 1.6f;
  int z = ((int)x + (int)y);
  int a = (int)(x + y);

  int whee = 10;
  whee++;

  // ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, ADD()); 
}
*/

#if 0
template<typename TA, typename... Tail>
std::unique_ptr<cudf::column> make_nested_column(TA p, Tail... tail);

struct column_builder {
   // fixed-width
   template<typename ColumnType, typename TA, typename... Tail, std::enable_if_t<cudf::is_fixed_width<ColumnType>()>* = nullptr>
   std::unique_ptr<cudf::column> operator()(TA t, Tail... tail)
   {      
      printf("Child:\n");
      auto ret = cudf::make_fixed_width_column(std::get<0>(t), std::get<1>(t));
      cudf::test::print(*ret);
      return ret;
   }   

   /*
   // strings   
   template<typename ColumnType, typename TA, typename... Tail, std::enable_if_t<std::is_same<ColumnType, cudf::string_view>::value>* = nullptr>
   std::unique_ptr<cudf::column> operator()(TA t, Tail... tail)
   {      
      cudf::data_type type = std::get<0>(t);
      cudf::size_type size = std::get<1>(t);
      std::vector<std::string> &strings = std::get<2>(t);      

      return{};
   }
   */
      
   // bogus leaf
   template<typename ColumnType, typename TA, typename... Tail, std::enable_if_t<std::is_same<ColumnType, cudf::list_view>::value>* = nullptr>
   std::unique_ptr<cudf::column> operator()(TA t)
   { 
      CUDF_FAIL("Tried to build a nested structure with a nested type (list) as a leaf node.");      
   }

   // nesting
   template<typename ColumnType, typename TA, typename... Tail, std::enable_if_t<std::is_same<ColumnType, cudf::list_view>::value>* = nullptr>
   std::unique_ptr<cudf::column> operator()(TA t, Tail... tail)
   {
      printf("Parent:\n");
      // construct children
      std::vector<std::unique_ptr<cudf::column>> children;
      // offsets column
      children.push_back(make_numeric_column(cudf::data_type{cudf::INT32}, std::get<1>(t)+1, cudf::mask_state::UNALLOCATED));
      // child column
      children.push_back(make_nested_column(tail...));
      
      // the parent/me. hmm, it's a little weird for us to be setting a size but not actually having a
      // data buffer.  but that's kind of the new reality in the nested types world.
      auto ret = std::make_unique<cudf::column>(std::get<0>(t), static_cast<cudf::size_type>(std::get<1>(t)), 
                                                rmm::device_buffer{}, rmm::device_buffer{}, 
                                                cudf::UNKNOWN_NULL_COUNT, std::move(children));
      
      return ret;
   }

   // TODO : handle other types (timestamps, strings, structs, etc)
   template<typename ColumnType, typename TA, typename... Tail, std::enable_if_t<!cudf::is_fixed_width<ColumnType>() && 
                                                                                 !std::is_same<ColumnType, cudf::list_view>::value/* &&
                                                                                 !std::is_same<ColumnType, cudf::string_view>::value*/>* = nullptr>
   std::unique_ptr<cudf::column> operator()(TA t, Tail... tail)
   {
      return {};
   }
};

template<typename TA, typename... Tail>
std::unique_ptr<cudf::column> make_nested_column(TA p, Tail... tail)
{      
   return cudf::type_dispatcher(std::get<0>(p), column_builder{}, p, tail...);
}
#endif

#if 0
template<typename TA, typename... Tail>
std::unique_ptr<cudf::column> make_nested_column_hierarchy(TA p, Tail... tail);

template <typename ColumnType, class TupleType>
constexpr inline bool is_string_specialization() {
   return std::tuple_size<TupleType>::value == 3 && std::is_same<ColumnType, cudf::string_view>::value;
} 

struct column_builder {   
   // fixed-width
   template<typename ColumnType, typename TA, typename... Tail, std::enable_if_t<cudf::is_fixed_width<ColumnType>()>* = nullptr>
   std::unique_ptr<cudf::column> operator()(TA t, Tail... tail)
   {      
      printf("Child (fixed):\n");
      auto ret = cudf::make_fixed_width_column(std::get<0>(t), std::get<1>(t));
      cudf::test::print(*ret);
      return ret;
   }   
      
   // strings   
   template<typename ColumnType, typename TA, typename... Tail, std::enable_if_t<is_string_specialization<ColumnType, TA>()>* = nullptr>
   std::unique_ptr<cudf::column> operator()(TA t, Tail... tail)
   {      
      printf("Child (string):\n");
      cudf::data_type type = std::get<0>(t);
      cudf::size_type size = std::get<1>(t);
      std::vector<std::string> &strings = std::get<2>(t);       

      return{};
   }    
      
   // bogus leaf
   template<typename ColumnType, typename TA, typename... Tail, std::enable_if_t<std::is_same<ColumnType, cudf::list_view>::value>* = nullptr>
   std::unique_ptr<cudf::column> operator()(TA t)
   { 
      CUDF_FAIL("Tried to build a nested structure with a nested type (list) as a leaf node.");      
   }

   // nesting
   template<typename ColumnType, typename TA, typename... Tail, std::enable_if_t<std::is_same<ColumnType, cudf::list_view>::value>* = nullptr>
   std::unique_ptr<cudf::column> operator()(TA t, Tail... tail)
   {
      printf("Parent:\n");
      // construct children
      std::vector<std::unique_ptr<cudf::column>> children;
      // offsets column
      children.push_back(make_numeric_column(cudf::data_type{cudf::INT32}, std::get<1>(t)+1, cudf::mask_state::UNALLOCATED));
      // child column
      children.push_back(make_nested_column_hierarchy(tail...));
      
      // the parent/me. hmm, it's a little weird for us to be setting a size but not actually having a
      // data buffer.  but that's kind of the new reality in the nested types world.
      auto ret = std::make_unique<cudf::column>(std::get<0>(t), static_cast<cudf::size_type>(std::get<1>(t)), 
                                                rmm::device_buffer{}, rmm::device_buffer{}, 
                                                cudf::UNKNOWN_NULL_COUNT, std::move(children));
      
      return ret;
   }

   // TODO : handle other types (timestamps, strings, structs, etc)
   template<typename ColumnType, typename TA, typename... Tail, std::enable_if_t<!cudf::is_fixed_width<ColumnType>() && 
                                                                                 !std::is_same<ColumnType, cudf::list_view>::value &&
                                                                                 !is_string_specialization<ColumnType, TA>()>* = nullptr>
   std::unique_ptr<cudf::column> operator()(TA t, Tail... tail)
   {      
      return {};
   }
};

template<typename TA, typename... Tail>
std::unique_ptr<cudf::column> make_nested_column_hierarchy(TA p, Tail... tail)
{    
   return cudf::type_dispatcher(std::get<0>(p), column_builder{}, p, tail...);
}

void factory_test()
{   
   std::vector<std::string> init { "hello" };
   auto what = make_nested_column_hierarchy(std::tuple<cudf::data_type, cudf::size_type>(cudf::data_type{cudf::LIST}, 1),
                                    std::tuple<cudf::data_type, cudf::size_type, std::vector<std::string>&>(cudf::data_type{cudf::STRING}, 1, init));

   auto whaaaaat = make_nested_column_hierarchy(std::tuple<cudf::data_type, cudf::size_type>(cudf::data_type{cudf::LIST}, 1),
                                       std::tuple<cudf::data_type, cudf::size_type>(cudf::data_type{cudf::INT32}, 1));                                    
}
#endif

#if 0
class list_column_wrapper {
public:   
/*
   template<typename TA, typename... Tail>
   void splat(TA t)
   {
      printf("D : %d\n", t.size());
   }

   template<typename TA, typename... Tail>
   void splat(TA t, Tail... tail)
   {
       printf("C : %d\n", t.size());
       splat(tail...);
   }
   
   template<typename TA, typename... Tail>
   void splat(std::initializer_list<TA> t)
   {
       printf("B : %d\n", t.size());
   }

   template<typename TA, typename... Tail>
   void splat(std::initializer_list<TA> t, std::initializer_list<Tail>... tail)
   {
       printf("A : %d\n", t.size());
       splat(tail...);
   }
   */   

   /*
   template<typename TA, typename... Tail>
   void splat(std::initializer_list<std::initializer_list<TA>> t, std::initializer_list<std::initializer_list<Tail>>... tail)
   {
      printf("LIST : %d\n", t.size());
      splat(tail...);
   } 
   */  
         
         /*
   template<typename... Tail>
   list_column_wrapper(std::initializer_list<Tail>... tail)
   {
      printf("Start\n");
      splat(tail...);
   }
   */  
   /*
   template<typename... Tail>
   list_column_wrapper(std::initializer_list<std::initializer_list<Tail>>... tail)
   {
      printf("Start\n");
      splat(tail...);
   }
   */

   template<typename T>
   void splat(T t)
   {
      printf("whee\n");
   }

   template<typename T, typename... Tail>
   void splat(std::initializer_list<T> t, std::initializer_list<Tail>... tail)
   {
      printf("LIST(%d)\n", t.size());      
      //std::for_each(t.begin(), t.end(), [this](T t){
        // splat(t);
      //});
   }

   /*
   template<typename... T>
   list_column_wrapper(std::initializer_list<T>... t)
   {
      printf("Start\n");
      splat(t...);
   }
   */

   int is_list = false;
   int inside = 0;

   template<typename T>
   list_column_wrapper(std::initializer_list<T> t)
   {
      is_list = false;
      inside = t.size();
      printf("Start(%d)\n", t.size());      
   }
   
   list_column_wrapper(std::initializer_list<list_column_wrapper> t)
   {
      is_list = true;
      printf("Start(%d)\n", t.size());
      std::for_each(t.begin(), t.end(), [](list_column_wrapper w){      
         bool chk = w.is_list;
         int whee = w.inside;
         whee++;
      });
   }   
};

#endif

#if 0
#include <jit/type.h>
#include <cudf/concatenate.hpp>

class list_column_wrapper : public cudf::test::detail::column_wrapper {
public:   
   template<typename T>
   list_column_wrapper(std::initializer_list<T> t)
   {            
      wrapped = cudf::test::fixed_width_column_wrapper<T>(t).release();      
      test::print(*wrapped);
   }

   void build_wrapper(std::initializer_list<list_column_wrapper> t)
   {
      // generate offsets column and do some type checking to make sure the user hasn't passed an invalid initializer list      
      type_id child_id = EMPTY;
      size_type count = 0;
      std::vector<size_type> offsetv;
      std::transform(t.begin(), t.end(), std::back_inserter(offsetv), [&](list_column_wrapper const& l){
         // verify all children are of the same type (C++ allows you to use initializer
         // lists that could construct an invalid list column type)
         if(child_id == EMPTY){
            child_id = l.wrapped->type().id();
         } else {
            CUDF_EXPECTS(child_id == l.wrapped->type().id(), "Malformed list elements");
         }  

         size_type ret = count;
         count += l.wrapped->size();
         return ret;
      });
      // add the final offset
      offsetv.push_back(count);      
      test::fixed_width_column_wrapper<size_type> offsets(offsetv.begin(), offsetv.end());
      test::print(offsets);

      // generate final column

      // if the child columns are primitive types, merge them into a new primitive column and make that my child
      if(child_id != LIST){         
         // merge data
         std::vector<column_view> children;
         std::transform(t.begin(), t.end(), std::back_inserter(children), [&](list_column_wrapper const& l){
            CUDF_EXPECTS(l.wrapped->type().id() == child_id, "Unexpected type mismatch");
            return static_cast<column_view>(*l.wrapped);
         });
         auto child_data = concatenate(children);         

         test::print(*child_data);         
         wrapped = make_lists_column(t.size(), offsets.release(), std::move(child_data));
      } 
      // if the child columns -are- lists, merge them into a new list column and make that my child
      else {
         // TODO :  this should just be cudf::concatenate().  that is, concatenate should support list_views. For now
         //         this remains as a one-off in here.

         size_type child_list_count = 0;
         thrust::host_vector<column_device_view> child_offset_columns;

         // merge data. 
         // also prep data needed for offset merging         
         std::vector<column_view> children;
         std::transform(t.begin(), t.end(), std::back_inserter(children), [&](list_column_wrapper const& l){
            CUDF_EXPECTS(l.wrapped->type().id() == LIST, "Unexpected type mismatch");
            child_list_count += l.wrapped->size();

            // OFFSET HACK            
            column_device_view cdv(l.wrapped->child(0), 0, 0);
            child_offset_columns.push_back(cdv);

            // OFFSET_HACK
            return static_cast<column_view>(l.wrapped->child(1));
         });
         auto child_data = concatenate(children);
         test::print(*child_data);

         // merge offsets.
         auto child_offsets = make_fixed_width_column(data_type{INT32}, child_list_count+1);
         mutable_column_device_view d_child_offsets(*child_offsets, 0, 0);
         size_type shift = 0;
         count = 0;
         std::for_each(child_offset_columns.begin(), child_offset_columns.end(),
            [&d_child_offsets, &shift, &count](column_device_view const& offsets){
               thrust::transform(rmm::exec_policy(0)->on(0), 
                  offsets.begin<size_type>(), 
                  offsets.end<size_type>(), 
                  d_child_offsets.begin<size_type>() + count,
                  [shift] __device__ (size_type offset){                     
                     return offset + shift;
                  }
               );
               shift += offsets.size();
               count += offsets.size()-1;
            }

            // poke the last offset in there
         );
         test::print(*child_offsets);

         // make the child list column
         auto child_list = make_lists_column(child_list_count, std::move(child_offsets), std::move(child_data));

         // now construct me         
         wrapped = make_lists_column(t.size(), offsets.release(), std::move(child_list));
      }
   }
      
   list_column_wrapper(std::initializer_list<list_column_wrapper> t)
   {      
      build_wrapper(t);      
   }

   std::string get_column_type_str()
   {
      return get_column_type_str(*wrapped);
   }

protected:  
   std::string get_column_type_str(cudf::column_view const& view)
   {                  
      if(view.type().id() == cudf::LIST){      
         // OFFSET HACK.  
         return cudf::jit::get_type_name(view.type()) + "<" + get_column_type_str(view.child(1)) + ">";
      } 
      return cudf::jit::get_type_name(view.type());
   }
};
#endif

#if 0
#include <jit/type.h>
#include <cudf/concatenate.hpp>
#include <cudf/lists/lists_column_view.hpp>

void list_test()
{   
   // inferred type
   {            
      // List<int32>, 1 row
      test::list_column_wrapper list1 { {0, 1} };
      test::print(list1);
      
      // List<int32>, 4 rows      
      test::list_column_wrapper list2 { {12, -7, 25}, {0}, {0, -127, 127, 50}, {0} };
      test::print(list2);

      // List<List<int32>> 1 rows
      test::list_column_wrapper list3 { {{1, 2}, {3, 4}} };
      test::print(list3);

      // List<List<int32>> 2 rows
      test::list_column_wrapper list4 { {{1, 2}, {3, 4}}, {{5, 6, 7}, {0}, {8}} };
      test::print(list4);

      // List<List<int32>> 3 rows
      test::list_column_wrapper list5 { {{1, 2}, {3, 4}}, {{5, 6, 7}, {0}, {8}}, {{9, 10}} };
      test::print(list5);

      // List<List<List<int32>>> 2 rows
      test::list_column_wrapper list6 { {{{1, 2}, {3, 4}}, {{5, 6, 7}, {0}}}, {{{-1, -2}, {-3, -4}}, {{-5, -6, -7}, {0}}} };
      test::print(list6);
   }

   // explicit type
   {           
      using T = float;
      using L = std::initializer_list<T>;

      // List<T>, 1 row      
      test::list_column_wrapper list1 { L{0, 1} };
      test::print(list1);
      
      // List<T>, 4 rows      
      test::list_column_wrapper list2 { L{12, -7, 25}, L{0}, L{0, -127, 127, 50}, L{0} };
      test::print(list2);

      // List<List<T>> 1 rows
      test::list_column_wrapper list3 { {L{1, 2}, L{3, 4}} };
      test::print(list3);

      // List<List<T>> 2 rows
      test::list_column_wrapper list4 { {L{1, 2}, L{3, 4}}, {L{5, 6, 7}, L{0}, L{8}} };
      test::print(list4);

      // List<List<T>> 3 rows
      test::list_column_wrapper list5 { {L{1, 2}, L{3, 4}}, {L{5, 6, 7}, L{0}, L{8}}, {L{9, 10}} };
      test::print(list5);

      // List<List<List<T>>> 2 rows
      test::list_column_wrapper list6 { {{L{1, 2}, L{3, 4}}, {L{5, 6, 7}, L{0}}}, {{L{-1, -2}, L{-3, -4}}, {L{-5, -6, -7}, L{0}}} };
      test::print(list6);
   }

   // string
   {
      // List<string>, 1 rows
      test::list_column_wrapper list1 { {"one", "two"}  };
      test::print(list1);
      
      // List<string>, 4 rows      
      test::list_column_wrapper list2 { {"one", "two", "three"}, {"four"}, {"five", "six", "seven", "eight"}, {"nine"} };
      test::print(list2);

      // List<List<string>> 1 rows
      test::list_column_wrapper list3 { {{"one", "two"}, {"three", "four"}} };
      test::print(list3);

      // List<List<string>> 2 rows
      test::list_column_wrapper list4 { {{"one", "two"}, {"three", "four"}}, {{"five", "six", "seven"}, {"eight"}, {"nine"}} };
      test::print(list4);
   }

   // inferred type with validity
   {
      auto valids = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i % 2 == 0 ? true : false; });

      // List<int32>, 1 row      
      test::list_column_wrapper list1 { {{0, 1}, valids} };
      test::print(list1);      

      // List<int32>, 4 rows      
      test::list_column_wrapper list2 { {{12, -7, 25}, valids}, {{0}, valids}, {{0, -127, 127, 50}, valids}, {{0}, valids} };
      test::print(list2);

      // List<List<int32>> 2 rows
      test::list_column_wrapper list3 { {{{1, 2}, {3, 4}}, valids}, {{{5, 6, 7}, {0}, {8}}, valids} };
      test::print(list3);
   }

   // explicit type with validity
   {
      using T = float;
      using L = std::initializer_list<T>;

      auto valids = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i % 2 == 0 ? true : false; });

      // List<int32>, 1 row      
      test::list_column_wrapper list1 { {L{0, 1}, valids} };
      test::print(list1);      

      // List<int32>, 4 rows      
      test::list_column_wrapper list2 { {L{12, -7, 25}, valids}, {L{0}, valids}, {L{0, -127, 127, 50}, valids}, {L{0}, valids} };
      test::print(list2);

      // List<List<int32>> 2 rows
      test::list_column_wrapper list3 { {{L{1, 2}, L{3, 4}}, valids}, {{L{5, 6, 7}, L{0}, L{8}}, valids} };
      test::print(list3);
   }

   // bogus stuff you can do
   {
      test::list_column_wrapper list1 {0, 1};
      test::print(list1);      

      test::list_column_wrapper list2 ( {0, 1} );
      test::print(list2);

      auto valids = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i % 2 == 0 ? true : false; });
      test::list_column_wrapper list3 { {0, 1}, valids };
      test::print(list3);      
   }
      /*
      // List<List<int32>>, 1 rows
      test::list_column_wrapper_style1 list3 { {{0, 1}, {2, 3, 4}} };
      test::print(list3);   
      test::list_column_wrapper_style2 list3a ( {{0, 1}, {2, 3, 4}} );
      test::print(list3a);

      // List<List<int32>>, 2 rows
      test::list_column_wrapper_style1 list4 { {{0, 1}, {2, 3, 4}}, {{5, 6}, {7, 8, 9}} };
      test::print(list4);
      test::list_column_wrapper_style2 list4a { {{0, 1}, {2, 3, 4}}, {{5, 6}, {7, 8, 9}} };
      test::print(list4a);

      // List<List<List<int32>>>, 1 row
      test::list_column_wrapper_style1 list5 { {{{0, 1}, {2, 3, 4}}, {{5, 6}, {7, 8, 9}}} };
      test::print(list5);
      test::list_column_wrapper_style2 list5a ( {{{0, 1}, {2, 3, 4}}, {{5, 6}, {7, 8, 9}}, {{11, 12}, {13, 14, 15}}} );
      test::print(list5a);

      // List<string>, 1 row
      test::list_column_wrapper_style1 list6 { {"a", "b"} };
      test::print(list6);
      test::list_column_wrapper_style2 list6a ( { "a", "b" } );
      test::print(list6a);

      // List<string>, 2 rows
      test::list_column_wrapper_style1 list7 { {"a", "b"}, {"c", "d"} };
      test::print(list7);
      test::list_column_wrapper_style2 list7a ( { "a", "b" }, {"c", "d"} );
      test::print(list7a);

      // List<List<string>>, 2 rows
      test::list_column_wrapper_style1 list8 { {{"a", "b"}, {"c", "d"}}, {{"e", "f"}, {"g", "h"}} };
      test::print(list8);
      test::list_column_wrapper_style2 list8a { {{"a", "b"}, {"c", "d"}}, {{"e", "f"}, {"g", "h"}} };
      test::print(list8a);
   }

   test::list_column_wrapper_style2 m { {{1, 2}, {3, 4}}, {{5, 6, 7}, {0}, {8}}, {{9, 10}, {11}} };
   test::print(m);

   test::list_column_wrapper_style2 m2 { {{1, 2}, {3, 4}}, {{5, 6, 7}, {0}, {8}}, {{9, 10}} };
   test::print(m2);
      */
   /*   
   {
      auto valids = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i % 2 == 0 ? false : true; });
      auto valids2 = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i % 2 == 0 ? false : true; });

      // validity
      //test::list_column_wrapper_style1 list1 { {2, 3, 4}, valids };
      //test::print(list1);

      test::list_column_wrapper_style2 list1 ( {0, 1}, valids );

      test::list_column_wrapper_style2 list1a ( { test::list_column_wrapper_style2({0, 1}, valids),
                                                test::list_column_wrapper_style2({2, 3, 4}, valids) });
      test::print(list1a);

      test::list_column_wrapper_style2 list2a ( { test::list_column_wrapper_style2({0, 1}, valids),
                                                test::list_column_wrapper_style2({2, 3, 4}, valids) }, valids);
      test::print(list2a);

      test::list_column_wrapper_style2 list3a {
                                                {test::list_column_wrapper_style2({0, 1}), test::list_column_wrapper_style2({2, 3, 4}) },
                                                {test::list_column_wrapper_style2({5, 6}), test::list_column_wrapper_style2({7, 8, 9}) }  
                                              };
      test::print(list3a);

      test::list_column_wrapper_style2 list4a {
                                                {{test::list_column_wrapper_style2({0, 1}), test::list_column_wrapper_style2({2, 3, 4})}, valids},
                                                {{test::list_column_wrapper_style2({5, 6}), test::list_column_wrapper_style2({7, 8, 9})}, valids},
                                              };
      test::print(list4a);     
   }
   */
            
   //test::list_column_wrapper_style2 list3b { test::list_column_wrapper_style2({0, 1}, {2, 3, 4}) };
         
   //test::list_column_wrapper_style2 list4a { test::list_column_wrapper_style2({0, 1}, {2, 3, 4}), 
     //                                        test::list_column_wrapper_style2({5, 6}, {7, 8, 9}) };
   //test::list_column_wrapper_style2 lista ( {{0, 1}, {2, 3, 4}}, {{5, 6}, {7, 8, 9}} );
   //test::print(list4a);

   /*
   test::list_column_wrapper list2 ( {2, 3, 4} );
   test::print(list2);

   auto cat = concatenate({list1, list2});
   test::print(*cat);

   test::list_column_wrapper list3 ( {{0, 1}, {2, 3, 4}} );
   test::print(list3);   

   test::list_column_wrapper list4 { test::list_column_wrapper({0, 1}, {2, 3, 4}), test::list_column_wrapper({5, 6}, {7, 8, 9}) };
   test::print(list4);   
   */

   //test::list_column_wrapper list4 { {{0, 1}, {2, 3, 4}}, {{5, 6}, {7, 8, 9}} };
   //test::print(list4);   

   /*
   auto valids = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i % 2 == 0 ? true : false; });
   
   test::list_column_wrapper list6 { test::list_column_wrapper({0, 1}, valids), 
                                     test::list_column_wrapper({3, 4}, valids) };
   test::print(list6);

   test::list_column_wrapper list7 ({ test::list_column_wrapper({0, 1}, valids), 
                                     test::list_column_wrapper({3, 4}, valids) }, valids);
   test::print(list7);
   */
   // test::list_column_wrapper list4 ( {{0, 1}, valids}, {{2, 3, 4}, valids} );
   // test::print(list4);

   // test::list_column_wrapper list5 ( {2, 3, 4}, valids );
   // test::print(list5);

   //auto cat2 = concatenate({list4, list5});
   //test::print(*cat2);

   //test::list_column_wrapper list2( {{0, 1, 2}, {2, 3, 4}} );
   //test::print(list2);   
   
   /*
   test::list_column_wrapper list2( {2, 3, 4} );
   test::print(list2);
   auto cat = concatenate({list1, list2});
   test::print(*cat);
   
   test::list_column_wrapper list3 { {0, 1}, {2, 3, 4} };
   test::print(list3);
   */

/*
   test::list_column_wrapper list4 { {{0, 1}, {2, 3, 4}}, {{5, 6}, {7, 8, 9, 10}} };
   test::print(list4);

   test::list_column_wrapper list5 { "a", "b" };
   test::print(list5);
   test::list_column_wrapper list6 ( { {"ayo", "holup"}, {"wuuut", "whee", "dang"} } );   
   test::print(list6);

   test::list_column_wrapper list7 { {{{{0, 1}, {2, 3, 4}}, {{4, 5}, {6, 7, 8, 9}}}} };
   test::print(list7);

   // w/validity
   auto valids = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i % 2 == 0 ? true : false; });
   
   test::list_column_wrapper list100{ {iter, iter+5, valids} };
   test::print(list100);
   
   test::list_column_wrapper list8 { {{1, 2}, valids}, {{3, 4}, valids} } ;   
   test::print(list8);
      
   test::list_column_wrapper list9 { {{{1, 2}, valids}, {{3, 4}, valids}}, {{{5, 6}, valids}, {{7, 8}, valids}} };
   test::print(list9);

   test::list_column_wrapper list10 { {{"hee", "whats"}, valids}, {{"hoo", "up"}, valids} } ;   
   test::print(list10);

   test::list_column_wrapper list11 { {{{"hee", "whats"}, valids}, {{"hoo", "up"}, valids}}, {{{"hey", "cash"}, valids}, {{"ho", "money"}, valids}}  };
   test::print(list11);

   test::list_column_wrapper list12 { {{1, 2}, {3, 4}, {5, 6}}, valids};
   test::print(list12);
   

   //list_column_wrapper list1 { {1.0f}, {2.0f, 3.0f}, {4.0f, 5.0f, 6.0f} };
   //test::print(list1);   
   
   //list_column_wrapper list2 { {{1.0f}, {2.0f, 3.0f}}, {{4.0f, 5.0f, 6.0f}} };   
   //test::print(list2);

   //list_column_wrapper list3 { {{{1.0f}, {2.0f, 3.0f}}, {{4.0f, 5.0f, 6.0f}}} };
   //test::print(list3);
   */
}
#endif

//#include <tests/utilities/column_wrapper.hpp>
//#include <cudf/lists/lists_column_view.hpp>

//#include <thrust/transform_scan.h>
//#include <thrust/binary_search.h>

/*
template <typename MapItType>
std::unique_ptr<column> gather_list(lists_column_view const& source_column,
                                    MapItType gather_map_begin,
                                    MapItType gather_map_end,
                                    bool nullify_out_of_bounds = false,
                                    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                    cudaStream_t stream = 0)
{
   auto output_count = std::distance(gather_map_begin, gather_map_end);
   auto offset_count = output_count + 1;    
   printf("Offset count : %lu\n", offset_count);
      
   size_type const* src_offsets{source_column.offsets().data<size_type>()};
      
   auto child_offsets = cudf::make_fixed_width_column(data_type{INT32}, offset_count); 
   mutable_column_view mcv = child_offsets->mutable_view();    
   size_type* dest_offsets = mcv.data<size_type>();

   // compact offsets.      
   auto count_iter =  thrust::make_counting_iterator<size_type>(0);
   thrust::plus<size_type> sum;
   thrust::transform_exclusive_scan(rmm::exec_policy(stream)->on(stream), count_iter, count_iter + offset_count, dest_offsets, 
         [gather_map_begin, output_count, src_offsets] __device__ (size_type index) -> size_type {
         // last offset index is always the previous offset_index + 1, since each entry in the gather map represents
         // a virtual pair of offsets
         size_type offset_index = index < output_count ? gather_map_begin[index] : gather_map_begin[index-1] + 1;
         // the length of this list
         return src_offsets[offset_index + 1] - src_offsets[offset_index];
         },
         0,
         sum);

   // the size of the gather map for the children is the last offset value (similar to the last offset being the
   // length of the chars column for a strings column)
   size_t child_gather_map_size = 7;

   // test::print(*child_offsets);

   // the upper bound for a given span of the output offsets. example:
   // dest_offsets = {0, 2, 7}
   // span_upper_bound[0] == 2
   // span_upper_bound[1] == 7
   // since the output offsets are compacted, each offset tells us where the indexing boundaries are for the child gather map
   auto span_upper_bound = thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0), [dest_offsets] __device__ (size_type index){
      return dest_offsets[index+1];
   });
   auto child_gather_iter = thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0), [span_upper_bound, output_count, dest_offsets, src_offsets, gather_map_begin] __device__ (size_type index){
      // figure out what span we are in. can't figure out a way around doing it without the upper bound.
      auto bound = thrust::upper_bound(thrust::device, span_upper_bound, span_upper_bound + output_count, index);      
      size_type offset_index = thrust::distance(span_upper_bound, bound);
                  
      // compute index into the bracket
      size_type bracket_local_index = offset_index == 0 ? index : index - dest_offsets[offset_index];

      // translate this into the gather index
      size_type gather_index = bracket_local_index + src_offsets[gather_map_begin[offset_index]];
      printf("BOUND : %d, %d, %d, %d, %d\n", index, *bound, offset_index, bracket_local_index, gather_index);
      return gather_index;
   });   
   thrust::for_each(rmm::exec_policy(stream)->on(stream), child_gather_iter, child_gather_iter + child_gather_map_size, [] __device__ (size_type gather_index){
      printf("%d, ", gather_index);
   });   

   // gather children
   
   // recurse on children
   // auto child = cudf::gather(source_column.child(), 
   auto child = cudf::make_fixed_width_column(data_type{INT32}, 1); 
   
   return make_lists_column(output_count, std::move(child_offsets), std::move(child), 0, rmm::device_buffer{});
}
*/

/*
template <typename MapItType>
std::unique_ptr<column> gather_list(lists_column_view const& source_column,                                    
                                    MapItType gather_map_begin,
                                    MapItType gather_map_end,
                                    bool nullify_out_of_bounds = false,
                                    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                    cudaStream_t stream = 0)
   {
   auto output_count = std::distance(gather_map_begin, gather_map_end);
   auto offset_count = output_count + 1;    
   printf("Offset count : %lu\n", offset_count);
      
   size_type const* src_offsets{source_column.offsets().data<size_type>()};
      
   auto child_offsets = cudf::make_fixed_width_column(data_type{INT32}, offset_count); 
   mutable_column_view mcv = child_offsets->mutable_view();    
   size_type* dest_offsets = mcv.data<size_type>();

   // compact offsets. generate new base indices
   auto count_iter =  thrust::make_counting_iterator<size_type>(0);
   thrust::plus<size_type> sum;
   thrust::transform_exclusive_scan(rmm::exec_policy(stream)->on(stream), count_iter, count_iter + offset_count, dest_offsets, 
         [gather_map_begin, output_count, src_offsets] __device__ (size_type index) -> size_type {
         // last offset index is always the previous offset_index + 1, since each entry in the gather map represents
         // a virtual pair of offsets
         size_type offset_index = index < output_count ? gather_map_begin[index] : gather_map_begin[index-1] + 1;
         // the length of this list
         return src_offsets[offset_index + 1] - src_offsets[offset_index];
         },
         0,
         sum);
   
   // test::print(*child_offsets);

   // the upper bound for a given span of the output offsets. example:
   // dest_offsets = {0, 2, 7}
   // span_upper_bound[0] == 2
   // span_upper_bound[1] == 7
   // since the output offsets are compacted, each offset tells us where the indexing boundaries are for the child gather map
   auto span_upper_bound = thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0), [dest_offsets] __device__ (size_type index){
      return dest_offsets[index+1];
   });
   auto child_gather_iter = thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0), [span_upper_bound, output_count, dest_offsets] __device__ (size_type index){
      // figure out what span we are in. can't figure out a way around doing it without the upper bound.
      auto bound = thrust::upper_bound(thrust::device, span_upper_bound, span_upper_bound + output_count, index);      
      size_type offset_index = thrust::distance(span_upper_bound, bound);
                  
      // compute index into the bracket
      size_type bracket_local_index = offset_index == 0 ? index : index - dest_offsets[offset_index];

      // translate this into the gather index
      size_type gather_index = bracket_local_index + base_indices[offset_index];
      printf("BOUND : %d, %d, %d, %d, %d\n", index, *bound, offset_index, bracket_local_index, gather_index);
      return gather_index;
   });         
   
   // recurse on children
   // auto child = cudf::gather(source_column.child(), 
   auto child = cudf::make_fixed_width_column(data_type{INT32}, 1); 
   
   return make_lists_column(output_count, std::move(child_offsets), std::move(child), 0, rmm::device_buffer{});
}
*/

/*
*/

#if 0

/**
 * @brief `column_wrapper` derived class for wrapping columns of lists.
 */
template <typename T>
class lists_column_wrapper : public detail::column_wrapper {
 public:       
 #if 0
  lists_column_wrapper(std::initializer_list<lists_column_wrapper<T>> elements) : column_wrapper{}
  {
    printf("BBB\n");
    std::vector<column_view> cols;
    std::transform(elements.begin(), elements.end(), std::back_inserter(cols), [](lists_column_wrapper<T> const& w){
      return static_cast<column_view>(w);
    });  
    build(cols, {});
  }   
  /*
  template <typename ValidityIterator>
  lists_column_wrapper(std::initializer_list<lists_column_wrapper<T>> elements, ValidityIterator v) : column_wrapper{}
  {
    printf("DDD\n");
    std::vector<column_view> cols;
    std::transform(elements.begin(), elements.end(), std::back_inserter(cols), [](lists_column_wrapper<T> const& w){
      return static_cast<column_view>(w);
    });  
    std::vector<bool> validity;
    std::transform(elements.begin(),
                   elements.end(),
                   v,
                   std::back_inserter(validity),
                   [](lists_column_wrapper<T> const& l, bool valid) { return valid; });
    build(cols, validity);
  }
  */
 /*
  template <typename ValidityIterator>
  lists_column_wrapper(std::initializer_list<column_wrapper> elements, ValidityIterator v) : column_wrapper{}
  {
    printf("DDD\n");
    std::vector<column_view> cols;
    std::transform(elements.begin(), elements.end(), std::back_inserter(cols), [](lists_column_wrapper<T> const& w){
      return static_cast<column_view>(w);
    });  
    std::vector<bool> validity;
    std::transform(elements.begin(),
                   elements.end(),
                   v,
                   std::back_inserter(validity),
                   [](lists_column_wrapper<T> const& l, bool valid) { return valid; });
    build(cols, validity);
  }
  */
 

  template <typename Elements = T, std::enable_if_t<cudf::is_fixed_width<Elements>()>* = nullptr>
  lists_column_wrapper(cudf::test::fixed_width_column_wrapper<T> const& col) : column_wrapper{}
  {
    printf("FFF1\n");    
    build({static_cast<column_view>(col)}, {}); 
  }    
  /*
  template <typename Elements = T, std::enable_if_t<cudf::is_fixed_width<Elements>()>* = nullptr>
  lists_column_wrapper(std::initializer_list<cudf::test::fixed_width_column_wrapper<T>> elements) : column_wrapper{}
  {
    printf("FFF\n");
    std::vector<column_view> cols;
    std::transform(elements.begin(), elements.end(), std::back_inserter(cols), [](cudf::test::fixed_width_column_wrapper<T> const& col){
      return static_cast<column_view>(col);
    });        
    build(cols, {}); 
  }
  */    
  /*
  template <typename Elements = T, typename ValidityIterator, std::enable_if_t<cudf::is_fixed_width<Elements>()>* = nullptr>
  lists_column_wrapper(std::initializer_list<cudf::test::fixed_width_column_wrapper<T>> elements, ValidityIterator v) : column_wrapper{}
  {
    printf("GGG\n");
    std::vector<column_view> cols;
    std::transform(elements.begin(), elements.end(), std::back_inserter(cols), [](cudf::test::fixed_width_column_wrapper<T> const& w){
      return static_cast<column_view>(w);
    });  
    std::vector<bool> validity;
    std::transform(elements.begin(),
                   elements.end(),
                   v,
                   std::back_inserter(validity),
                   [](cudf::test::fixed_width_column_wrapper<T> const& l, bool valid) { return valid; });
    build(cols, validity);
  } 
  */     
  
  /*
  template <typename Elements = T, typename std::enable_if_t<!cudf::is_fixed_width<Elements>()>* = nullptr>
  lists_column_wrapper(cudf::test::strings_column_wrapper && w) : column_wrapper{}
  {
    build_from_non_nested(std::move(w.release()));
  } 
  */  

  /**
   * @brief Construct a lists column of nested lists from an initializer list of values
   * and a validity iterator.
   *
   * Example:
   * @code{.cpp}
   * // Creates a LIST column with 3 lists
   * auto validity = make_counting_transform_iterator(0, [](auto i){return i%2;});
   * // [{0, 1}, NULL, {4, 5}]
   * lists_column_wrapper l{ {{0, 1}, {2, 3}, {4, 5}, validity} };
   * @endcode
   *
   * Automatically handles nesting
   * Example:
   * @code{.cpp}
   * // Creates a LIST of LIST columns with 2 lists on the top level and
   * // 4 below
   * auto validity = make_counting_transform_iterator(0, [](auto i){return i%2;});
   * // [ {{0, 1}, NULL}, {{4, 5}, NULL} ]
   * lists_column_wrapper l{ {{{0, 1}, {2, 3}}, validity}, {{{4, 5}, {6, 7}}, validity} };
   * @endcode
   *
   * @param elements The list of elements
   * @param v The validity iterator
   */  
  /*
  template <typename Elements = T, typename ValidityIterator>
  lists_column_wrapper(std::initializer_list<lists_column_wrapper<T>> elements, ValidityIterator v)
  // lists_column_wrapper(std::initializer_list<detail::column_wrapper> elements, ValidityIterator v)
    : column_wrapper{}
  {       
    printf("JJJ\n");
  
    std::vector<column_view> cols;
    std::transform(elements.begin(), elements.end(), std::back_inserter(cols), [](lists_column_wrapper<T> const& col){
      return static_cast<column_view>(col);
    });        
    std::vector<bool> validity;
    std::transform(elements.begin(),
                   elements.end(),
                   v,
                   std::back_inserter(validity),
                   [](lists_column_wrapper<T> const& l, bool valid) { return valid; });
    build(cols, validity);     
  }  
  */
 #endif  
 /*
  template <typename Elements = T, std::enable_if_t<is_fixed_width<T>()>* = nullptr>
  lists_column_wrapper(std::initializer_list<T> elements) : column_wrapper{}
  {
    printf("1111\n");
    auto c = cudf::test::fixed_width_column_wrapper<T>(elements).release();

    build({static_cast<column_view>(*c)}, {});
  }
  */  

  // template <typename Elements = T, std::enable_if_t<cudf::is_fixed_width<Elements>()>* = nullptr>
  /*
  template <std::enable_if_t<is_fixed_width<T>()>* = nullptr>
  lists_column_wrapper(std::initializer_list<T> elements) : column_wrapper{}
  {
    printf("1111\n");
    auto c = cudf::test::fixed_width_column_wrapper<T>(elements).release();
    build({static_cast<column_view>(*c)}, {});
    // root = true;
  }
  */ 
  // template <std::enable_if_t<cudf::is_fixed_width<T>()>* = nullptr>
  // lists_column_wrapper(cudf::test::fixed_width_column_wrapper<T> const& col)
  lists_column_wrapper(cudf::test::fixed_width_column_wrapper<T> const&)
  {
    printf("FFF1\n");    
    //build({static_cast<column_view>(col)}, {}); 
  }  
  // template <std::enable_if_t<cudf::is_fixed_width<T>()>* = nullptr>  
  // lists_column_wrapper(std::initializer_list<cudf::test::fixed_width_column_wrapper<T>> elements)
  lists_column_wrapper(std::initializer_list<cudf::test::fixed_width_column_wrapper<T>>)
  {
    printf("FFF\n");
    /*
    std::vector<column_view> cols;
    std::transform(elements.begin(), elements.end(), std::back_inserter(cols), [](cudf::test::fixed_width_column_wrapper<T> const& col){
      return static_cast<column_view>(col);
    });        
    build(cols, {}); 
    */
  }
  // template <typename ValidityIterator, std::enable_if_t<cudf::is_fixed_width<T>()>* = nullptr>
  template<typename ValidityIterator>
  // lists_column_wrapper(std::initializer_list<cudf::test::fixed_width_column_wrapper<T>> elements, ValidityIterator v)
  lists_column_wrapper(std::initializer_list<cudf::test::fixed_width_column_wrapper<T>>, ValidityIterator)
  {
    printf("GGG\n");
    /*
    std::vector<column_view> cols;
    std::transform(elements.begin(), elements.end(), std::back_inserter(cols), [](cudf::test::fixed_width_column_wrapper<T> const& w){
      return static_cast<column_view>(w);
    });  
    std::vector<bool> validity;
    std::transform(elements.begin(),
                   elements.end(),
                   v,
                   std::back_inserter(validity),
                   [](cudf::test::fixed_width_column_wrapper<T> const& l, bool valid) { return valid; });
    build(cols, validity);
    */
  } 

  // lists_column_wrapper(std::initializer_list<lists_column_wrapper<T>> elements)
  lists_column_wrapper(std::initializer_list<lists_column_wrapper<T>>)
  {
    printf("2222\n");
    /*
    std::vector<column_view> cols;
    std::transform(elements.begin(), elements.end(), std::back_inserter(cols), [](lists_column_wrapper<T> const& col){
      return static_cast<column_view>(col);
    });
    
    build(cols, {});
    */
  }  
  template <typename ValidityIterator>
  // lists_column_wrapper(std::initializer_list<lists_column_wrapper<T>> elements, ValidityIterator v)
  lists_column_wrapper(std::initializer_list<lists_column_wrapper<T>> , ValidityIterator )
  {
    printf("DDD\n");
    /*
    std::vector<column_view> cols;
    std::transform(elements.begin(), elements.end(), std::back_inserter(cols), [](lists_column_wrapper<T> const& w){
      return static_cast<column_view>(w);
    });  
    std::vector<bool> validity;
    std::transform(elements.begin(),
                   elements.end(),
                   v,
                   std::back_inserter(validity),
                   [](lists_column_wrapper<T> const& l, bool valid) { return valid; });
    build(cols, validity);
    */
  }

  lists_column_wrapper(std::initializer_list<lists_column_wrapper<T>> elements,
                             std::initializer_list<bool> validity)
  {
     printf("AAA\n");
  }

 private:
  /**
   * @brief Initialize as a nested list column composed of other columns (which may themselves be lists)
   *   
   *
   * List<int>      = { 0, 1 }
   * List<int>      = { {0, 1} }
   *
   * while at the same time, allowing further nesting
   * List<List<int> = { {{0, 1}} }
   *
   * @param c Input column to be wrapped
   *
   */
  void build(std::vector<column_view> const& elements,
                                              std::vector<bool> const& v)
  {
    auto valids = cudf::test::make_counting_transform_iterator(
      0, [&v](auto i) { return v.empty() ? true : v[i]; });

    // preprocess the incoming lists. unwrap any "root" lists and just use their
    // underlying non-list data.
    // also, sanity check everything to make sure the types of all the columns are the same
    std::vector<column_view> cols;
    type_id child_id = EMPTY;
    std::transform(elements.begin(),
                  elements.end(),
                  std::back_inserter(cols),
                  [&child_id](column_view const& col) {
                    // verify all children are of the same type (C++ allows you to use initializer
                    // lists that could construct an invalid list column type)
                    if (child_id == EMPTY) {
                      child_id = col.type().id();
                    } else {
                      CUDF_EXPECTS(child_id == col.type().id(), "Mismatched list types");
                    }

                    return col;
                  });

    // generate offsets column and do some type checking to make sure the user hasn't passed an
    // invalid initializer list
    size_type count = 0;
    std::vector<size_type> offsetv;
    std::transform(cols.begin(),
                  cols.end(),
                  valids,
                  std::back_inserter(offsetv),
                  [&](cudf::column_view const& col, bool valid) {
                    // nulls are represented as a repeated offset
                    size_type ret = count;
                    if (valid) { count += col.size(); }
                    return ret;
                  });
    // add the final offset
    offsetv.push_back(count);
    auto offsets =
      cudf::test::fixed_width_column_wrapper<size_type>(offsetv.begin(), offsetv.end()).release();

    // concatenate them together, skipping data for children that are null
    std::vector<column_view> children;
    for (int idx = 0; idx < cols.size(); idx++) {
      if (valids[idx]) { children.push_back(cols[idx]); }
    }
    auto data = concatenate(children);

    // construct the list column
    wrapped = make_lists_column(
      cols.size(),
      std::move(offsets),
      std::move(data),
      v.size() <= 0 ? 0 : cudf::UNKNOWN_NULL_COUNT,
      v.size() <= 0 ? rmm::device_buffer{0} : detail::make_null_mask(v.begin(), v.end()));
  }
};
#endif

#if 0
void list_slice_test()
{ 
  auto valids = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });

  {
    cudf::test::lists_column_wrapper<int> list{{{{1, 2, 3}, valids}, {4, 5}},
                                               {{LCW{}, LCW{}, {7, 8}, LCW{}}, valids},
                                               {{LCW{6}}},
                                               {{{7, 8}, {{9, 10, 11}, valids}, LCW{}}, valids},
                                               {{LCW{}, {-1, -2, -3, -4, -5}}, valids},
                                               {LCW{}},
                                               {{-10}, {-100, -200}}};

    std::vector<cudf::size_type> splits{1, 3, 4};

    std::vector<cudf::test::lists_column_wrapper<int>> expected;
    expected.push_back(LCW{{{{1, 2, 3}, valids}, {4, 5}}});
    expected.push_back(LCW{{{LCW{}, LCW{}, {7, 8}, LCW{}}, valids}, {{LCW{6}}} });
    expected.push_back(LCW{{{{7, 8}, {{9, 10, 11}, valids}, LCW{}}, valids}});
    expected.push_back(
      LCW{{{LCW{}, {-1, -2, -3, -4, -5}}, valids}, {LCW{}}, {{-10}, {-100, -200}}});

    std::vector<cudf::column_view> result = cudf::split(list, splits);
    EXPECT_EQ(expected.size(), result.size());

    for (unsigned long index = 0; index < result.size(); index++) {
      cudf::test::print(expected[index]);
      printf("\n");
      cudf::test::print(result[index]);
      printf("\n\n\n");
      cudf::test::expect_columns_equivalent(expected[index], result[index]);
    }
  }

  {
    cudf::test::lists_column_wrapper<int> list{{1, 2, 3},
                                               {4, 5},
                                               {6},
                                               {{7, 8}, valids},
                                               {9, 10, 11},
                                               LCW{},
                                               LCW{},
                                               {{-1, -2, -3, -4, -5}, valids},
                                               {-10},
                                               {{-100, -200}, valids}};

    std::vector<cudf::size_type> splits{0, 1, 4, 5, 6, 9};

    std::vector<cudf::test::lists_column_wrapper<int>> expected;
    expected.push_back(LCW{});
    expected.push_back(LCW{{1, 2, 3}});
    expected.push_back(LCW{{4, 5}, {6}, {{7, 8}, valids}});
    expected.push_back(LCW{{9, 10, 11}});
    expected.push_back(LCW{{LCW{}}});
    expected.push_back(LCW{LCW{}, {{-1, -2, -3, -4, -5}, valids}, {-10}});
    expected.push_back(LCW{{{-100, -200}, valids}});

    std::vector<cudf::column_view> result = cudf::split(list, splits);
    EXPECT_EQ(expected.size(), result.size());

    for (unsigned long index = 0; index < result.size(); index++) {
      cudf::test::print(expected[index]);
      printf("\n");
      cudf::test::print(result[index]);
      printf("\n\n\n");
      cudf::test::expect_columns_equivalent(expected[index], result[index]);
    }
  }

  {
    cudf::test::lists_column_wrapper<int> list{{{{1, 2, 3}, valids}, {4, 5}},
                                               {{LCW{}, LCW{}, {7, 8}, LCW{}}, valids},
                                               {{{6}}},
                                               {{{7, 8}, {{9, 10, 11}, valids}, LCW{}}, valids},
                                               {{LCW{}, {-1, -2, -3, -4, -5}}, valids},
                                               {LCW{}},
                                               {{-10}, {-100, -200}}};

    std::vector<cudf::size_type> splits{1, 3, 4};

    std::vector<cudf::test::lists_column_wrapper<int>> expected;
    expected.push_back(LCW{{{{1, 2, 3}, valids}, {4, 5}}});
    expected.push_back(LCW{{{LCW{}, LCW{}, {7, 8}, LCW{}}, valids}, {{{6}}}});
    expected.push_back(LCW{{{{7, 8}, {{9, 10, 11}, valids}, LCW{}}, valids}});
    expected.push_back(
      LCW{{{LCW{}, {-1, -2, -3, -4, -5}}, valids}, {LCW{}}, {{-10}, {-100, -200}}});

    std::vector<cudf::column_view> result = cudf::split(list, splits);
    EXPECT_EQ(expected.size(), result.size());

    for (unsigned long index = 0; index < result.size(); index++) {
      cudf::test::expect_columns_equivalent(expected[index], result[index]);
    }
  }




  {
    cudf::test::lists_column_wrapper<int> list{ {{1, 2, 3}, {4, 5}}, 
                                                {LCW{}, LCW{}, {7, 8}, LCW{}},
                                                {{LCW{6}}},
                                                {{7, 8}, {9, 10, 11}, LCW{}}, 
                                                {LCW{}, {-1, -2, -3, -4, -5}}, 
                                                {LCW{}},
                                                {{-10}, {-100, -200}} };
    
    std::vector<cudf::size_type> splits{1, 3, 4};

    std::vector<cudf::test::lists_column_wrapper<int>> expected;    
    expected.push_back(LCW{ {{1, 2, 3}, {4, 5}} });
    expected.push_back(LCW{ {LCW{}, LCW{}, {7, 8}, LCW{}}, {{LCW{6}}} });
    expected.push_back(LCW{ {{7, 8}, {9, 10, 11}, LCW{}} });
    expected.push_back(LCW{ {LCW{}, {-1, -2, -3, -4, -5}}, {LCW{}}, {{-10}, {-100, -200}} });
    
    std::vector<cudf::column_view> result = cudf::split(list, splits);
    EXPECT_EQ(expected.size(), result.size());

    for (unsigned long index = 0; index < result.size(); index++) {
      cudf::test::print(expected[index]);
      cudf::test::print(result[index]);
      cudf::test::expect_columns_equivalent(expected[index], result[index]);
    }
  }    

  {    
    cudf::test::lists_column_wrapper<int> a{0, 1, 2, 3};
    cudf::test::lists_column_wrapper<int> b{4, 5, 6, 7, 8, 9, 10};
    cudf::test::lists_column_wrapper<int> expected{{0, 1, 2, 3}, {4, 5, 6, 7, 8, 9, 10}};

    auto result = cudf::concatenate({a, b});
    cudf::test::print(expected);
    cudf::test::print(*result);    

    cudf::test::expect_columns_equal(*result, expected);  
  }

  {
     auto valids = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });
  //cudf::test::lists_column_wrapper<int> { {{1, 2, 3}, {4, 5}, {6}, {7, 8}, {9, 10, 11},
                                          //LCW{}, LCW{}, {-1, -2, -3, -4, -5}, {-10}, {-100, -200}}, valids };

  cudf::test::lists_column_wrapper<int> list{ {1, 2, 3}, {4, 5}, {6}, {7, 8}, {9, 10, 11},
                                          LCW{}, LCW{}, {-1, -2, -3, -4, -5}, {-10}, {-100, -200} };
  
  // std::vector<cudf::size_type> indices{1, 3, 2, 4, 1, 9};
  std::vector<cudf::size_type> indices{1, 3, 2, 4, 1, 9};

  std::vector<cudf::test::lists_column_wrapper<int>> expected;
  expected.push_back(LCW{ {4, 5}, {6} });  
  expected.push_back(LCW{ {6}, {7, 8} });
  expected.push_back(LCW{ {4, 5}, {6}, {7, 8}, {9, 10, 11}, LCW{}, LCW{}, {-1, -2, -3, -4, -5}, {-10} });
  
  std::vector<cudf::column_view> result = cudf::slice(list, indices);
  cudf::test::print(result[0]);
  cudf::test::print(result[1]);
  cudf::test::print(result[2]);

  EXPECT_EQ(expected.size(), result.size());

  for (unsigned long index = 0; index < result.size(); index++) {
    cudf::test::expect_columns_equal(expected[index], result[index]);
  }  
  }

  { 
    cudf::test::lists_column_wrapper<int> list{ 
      {{0, 1, 2, 3, 4, 5}, {10, 20}}, 
      {{6, 7}, {11}},
      {{7, 8}, {0}}
    };
    auto sliced = cudf::split(list, {1});
    cudf::test::print(list);
    for(size_t idx=0; idx<sliced.size(); idx++){
      printf("S %lu\n", idx);
      cudf::test::print(sliced[idx]); 
    }
  }

  { 
    cudf::test::lists_column_wrapper<int> a{ 
      {{1, 1, 1}, {2, 2}, {3, 3}},
      {{4, 4, 4}, {5, 5}, {6, 6}},
      {{7, 7, 7}, {8, 8}, {9, 9}},
      {{10, 10, 10}, {11, 11}, {12, 12}},
    };
    auto sliced_a = cudf::split(a, {2});

    cudf::test::lists_column_wrapper<int> b{ 
      {{-1, -1, -1, -1}, {-2}},
      {{-3, -3, -3, -3}, {-4}},
      {{-5, -5, -5, -5}, {-6}},
      {{-7, -7, -7, -7}, {-8}},
    };
    auto sliced_b = cudf::split(b, {2});

    auto cat0 = cudf::concatenate({sliced_a[0], sliced_b[0]});
    cudf::test::print(*cat0);

    auto cat1 = cudf::concatenate({sliced_a[0], sliced_b[1]});
    cudf::test::print(*cat1);

    auto cat2 = cudf::concatenate({sliced_a[1], sliced_b[0]});
    cudf::test::print(*cat2);
    
    auto cat3 = cudf::concatenate({sliced_a[1], sliced_b[1]});
    cudf::test::print(*cat3);
  }

  {
    cudf::test::lists_column_wrapper<int> a{ 
      {{1, 1, 1}, {2, 2}, {3, 3}},
      {{4, 4, 4}, {5, 5}, {6, 6}},
      {{7, 7, 7}, {8, 8}, {9, 9}},
      {{10, 10, 10}, {11, 11}, {12, 12}},
      {{-1, -1, -1, -1}, {-2}},
      {{-3, -3, -3, -3}, {-4}},
      {{-5, -5, -5, -5}, {-6, -13}},
      {{-7, -7, -7, -7}, {-8}},
    };
    auto sliced_a = cudf::split(a, {3});   
    cudf::table_view tbl0({sliced_a[0]});
    cudf::table_view tbl1({sliced_a[1]});

    auto result0 = cudf::gather(tbl0, cudf::test::fixed_width_column_wrapper<int>{1, 2});
    cudf::test::print(result0->view().column(0));

    auto result1 = cudf::gather(tbl1, cudf::test::fixed_width_column_wrapper<int>{0, 3});
    cudf::test::print(result1->view().column(0));
  }
}
#endif

#if 0
void hierarchy_test()
{  
  /*
  // to disambiguate between {} == 0 and {} == List{0}
  // Also, see note about compiler issues when declaring nested
  // empty lists in lists_column_wrapper documentation
  using LCW = cudf::test::lists_column_wrapper<int>;
  {
    cudf::test::lists_column_wrapper<int> a;
    cudf::test::lists_column_wrapper<int> b{4, 5, 6, 7};
    cudf::test::lists_column_wrapper<int> expected{4, 5, 6, 7};

    auto result = cudf::concatenate({a, b});

    cudf::test::expect_columns_equal(*result, expected);
  }

  {
    cudf::test::lists_column_wrapper<int> a, b, c;
    cudf::test::lists_column_wrapper<int> d{4, 5, 6, 7};
    cudf::test::lists_column_wrapper<int> expected{4, 5, 6, 7};

    auto result = cudf::concatenate({a, b, c, d});

    cudf::test::expect_columns_equal(*result, expected);
  }

  {
    cudf::test::lists_column_wrapper<int> a{LCW{}};
    cudf::test::lists_column_wrapper<int> b{4, 5, 6, 7};
    cudf::test::lists_column_wrapper<int> expected{LCW{}, {4, 5, 6, 7}};

    cudf::test::print(a);
    cudf::test::print(aa);
    cudf::test::print(expected);

    auto result = cudf::concatenate({a, b});

    cudf::test::print(*result);

    cudf::test::expect_columns_equal(*result, expected);
  }

  {
    cudf::test::lists_column_wrapper<int> a, b, c;
    cudf::test::lists_column_wrapper<int> d{4, 5, 6, 7};
    cudf::test::lists_column_wrapper<int> expected{LCW{}, LCW{}, LCW{}, {4, 5, 6, 7}};

    auto result = cudf::concatenate({a, b, c, d});

    cudf::test::expect_columns_equal(*result, expected);
  }

  {
    cudf::test::lists_column_wrapper<int> a{1, 2};
    cudf::test::lists_column_wrapper<int> b, c;
    cudf::test::lists_column_wrapper<int> d{4, 5, 6, 7};
    cudf::test::lists_column_wrapper<int> expected{{1, 2}, LCW{}, LCW{}, {4, 5, 6, 7}};

    auto result = cudf::concatenate({a, b, c, d});

    cudf::test::expect_columns_equal(*result, expected);
  }
  /*
  // #define ELL {empty_list<int>{}}
  cudf::test::lists_column_wrapper<int>::empty        ELL{};

  {
     cudf::test::lists_column_wrapper<int> a{
      {{{0, 1, 2}, ELL}, {{5}, {6, 7}}, {{8, 9}} },
      {{ELL}, {{17, 18}, {19, 20}} },
      {{ELL}},
      { {{50}, {51, 52}}, {{53, 54}, {55, 16, 17}}, {{59, 60}}} };

    cudf::test::lists_column_wrapper<int> b{
      {{{21, 22}, {23, 24}}, {ELL, {26, 27}}, {{28, 29, 30}}},
      {{{31, 32}, {33, 34}}, {{35, 36}, {37, 38}, {1, 2}}, {{39, 40}}},
      {{ELL}} };

    cudf::test::lists_column_wrapper<int> expected{
      {{{0, 1, 2}, ELL}, {{5}, {6, 7}}, {{8, 9}}},
      {{ELL}, {{17, 18}, {19, 20}}},
      {{ELL}},
      {{{50}, {51, 52}}, {{53, 54}, {55, 16, 17}}, {{59, 60}}},
      {{{21, 22}, {23, 24}}, {ELL, {26, 27}}, {{28, 29, 30}}},
      {{{31, 32}, {33, 34}}, {{35, 36}, {37, 38}, {1, 2}}, {{39, 40}}},
      {{ELL}}};

  }
  {     
    // cudf::test::lists_column_wrapper<int> aa{ {{LCW{}}} , {{0, 1}, {2, 3}} };
    cudf::test::lists_column_wrapper<int> ab{ {{{LCW{}}}} , ELL };
    cudf::test::print(ab);

    cudf::test::lists_column_wrapper<int> a;
    cudf::test::print(a); 
  
    cudf::test::lists_column_wrapper<int> c{ ELL };
    cudf::test::print(c);

    cudf::test::lists_column_wrapper<int> d{ LCW{} };
    cudf::test::print(d); 

    cudf::test::lists_column_wrapper<int> e{ {ELL} };
    cudf::test::print(e); 

    // cudf::test::lists_column_wrapper<int> e{{EL}};
    //cudf::test::lists_column_wrapper<int> e{LCW{LCW{}}};
    //cudf::test::print(e);
  }

  {
    cudf::test::lists_column_wrapper<int> expected{{{EL}}, {EL}, EL};
    cudf::test::print(expected);

    std::vector<bool> valids{false};
    test::lists_column_wrapper<int> list{ {{{EL}}, valids.begin()}, {EL}, EL };
    cudf::test::print(list);
  }

  {
    std::vector<bool> valids{false};
    cudf::test::lists_column_wrapper<int> a{{{EL}}};
    cudf::test::lists_column_wrapper<int> b{{EL}};
    cudf::test::lists_column_wrapper<int> c{EL};
    auto result = cudf::concatenate({a, b, c});

    cudf::test::lists_column_wrapper<int> expected{{{EL}}, {EL}, EL};

    cudf::test::expect_columns_equal(*result, expected);
  }

  {
    cudf::test::lists_column_wrapper<int> a{ {{{{{EL}}}}}, std::vector<bool>{false}.begin() };
    cudf::test::print(a);
  }

  {
    cudf::test::lists_column_wrapper<int> a{ {{{{{EL}}}}}, std::vector<bool>{false}.begin() };
    cudf::test::print(a);    
    cudf::test::lists_column_wrapper<int> b{ {{{{{EL}}}}}, std::vector<bool>{false}.begin() };
    cudf::test::print(b);
    auto c = cudf::concatenate({a, b});
    cudf::test::print(*c);
  }

  { 
    cudf::test::lists_column_wrapper<int> list{ {{{{EL}}}, {}}, std::vector<bool>{false, false}.begin() };
    cudf::test::print(list);  
    auto whee = cudf::empty_like(list);
    cudf::test::print(*whee);  
  }

  { 
    cudf::test::lists_column_wrapper<int> list{ {{{{EL}}}}, std::vector<bool>{false}.begin() };
    cudf::test::print(list);  
    auto whee = cudf::empty_like(list);
    cudf::test::print(*whee);  
  }
  */
}
#endif

#if 0
void expect_columns_equal_test()
{
  /*
  cudf::test::fixed_width_column_wrapper<int> a { 1, 2, 3, 4, 5, 6 };
  cudf::test::fixed_width_column_wrapper<int> b { 1, 2, 3, 4, 5, 6 };
  cudf::test::expect_columns_equal(a, b);
  
  cudf::test::list_column_wrapper lista { {{0, 1}, {2, 3}}, {{4}, {6, 7, 8}} };
  cudf::test::list_column_wrapper listb { {{0, 1}, {2, 3}}, {{4}, {6, 7, 8}} };
  cudf::test::expect_columns_equal(lista, listb);
  */

 /*
  
  test::list_column_wrapper list{
      {{{1, 2}, {3, 4}}, valids}, {{{5, 6, 7}, {0}, {8}}, valids}, {{{9, 10}}, valids}};
  cudf::test::print(list);
  */
 auto valids = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });

  // test::lists_column_wrapper list{ {{{1, 2}, {3, 4}}, valids}, {{{5, 6}, {7, 8}}, valids} };
  {
    test::lists_column_wrapper list{ {{12, -7, 25}, {0}, {0, -127, 127, 50}}, valids };
    test::print(list);
  }

  {
    test::lists_column_wrapper list{ {{1, 2}, {3, 4}}, {{{5, 6, 7}, {0}, {8}}, valids}, {{9, 10}} };
    //test::lists_column_wrapper a{ {{5, 6, 7}, {0}, {8}}, valids };
    //test::lists_column_wrapper b{ {9, 10} };
    //auto list = cudf::concatenate({a, b});
    test::print(list);
  }
}
#endif

#if 0
void parquet_speed_test()
{
  namespace cudf_io = cudf::io;
    
  using column         = cudf::column;
  using table          = cudf::table;
  using table_view     = cudf::table_view;
  using TypeParam = int;

  {
    std::vector<char> mm_buf;
    mm_buf.reserve(4 * 1024 * 1024 * 16);
    custom_test_memmap_sink<false> custom_sink(&mm_buf);

    namespace cudf_io = cudf::io;

    // exercises multiple rowgroups
    srand(31337);
    auto expected = create_compressible_fixed_table<int>(16, 4 * 1024 * 1024, 6, false);

    // write out using the custom sink (which uses device writes)
    cudf_io::write_parquet_args args{cudf_io::sink_info{&custom_sink}, *expected};
    cudf_io::write_parquet(args);

    cudf_io::read_parquet_args custom_args{cudf_io::source_info{mm_buf.data(), mm_buf.size()}};
    //auto custom_tbl = cudf_io::read_parquet(custom_args);
    //expect_tables_equal(custom_tbl.tbl->view(), expected->view());
  }

  {
    std::vector<char> mm_buf;
    mm_buf.reserve(4 * 1024 * 1024 * 16);
    custom_test_memmap_sink<false> custom_sink(&mm_buf);

    namespace cudf_io = cudf::io;

    // exercises multiple rowgroups
    srand(31337);
    auto expected = create_compressible_fixed_table<int>(16, 4 * 1024 * 1024, 6, true);

    // write out using the custom sink (which uses device writes)
    cudf_io::write_parquet_args args{cudf_io::sink_info{&custom_sink}, *expected};
    cudf_io::write_parquet(args);

    cudf_io::read_parquet_args custom_args{cudf_io::source_info{mm_buf.data(), mm_buf.size()}};
    auto custom_tbl = cudf_io::read_parquet(custom_args);
    expect_tables_equal(custom_tbl.tbl->view(), expected->view());
  }
}
#endif

/*
struct chunk_row_output_iter : public thrust::iterator_facade<chunk_row_output_iter, int32_t, thrust::device_system_tag, thrust::forward_device_iterator_tag, int32_t&, int32_t> {
  PageInfo *p;      
  __host__ __device__ chunk_row_output_iter(PageInfo *_p){p = _p;}
  //__host__ __device__ chunk_row_output_iter() {}
  // __host__ __device__ chunk_row_output_iter(chunk_row_output_iter const& iter) { p = iter.p; }
private:
  friend class thrust::iterator_core_access;  
  __host__ __device__ void advance(size_type n) { p += n; }
  __host__ __device__ void increment() { p++; }
  __device__ reference dereference() const { return p->chunk_row; }
};

struct start_offset_output_iterator : public thrust::iterator_facade<start_offset_output_iterator, int32_t, thrust::device_system_tag, thrust::forward_device_iterator_tag, int32_t&, int32_t> {
  PageInfo *p;
  int col_index;
  int nesting_depth;
  int32_t empty;
  __host__ __device__ start_offset_output_iterator(PageInfo *_p, int _col_index, int _nesting_depth){
    p = _p;
    col_index = _col_index;
    nesting_depth = _nesting_depth;      
  }
private:
  friend class thrust::iterator_core_access;  
  __host__ __device__ void advance(size_type n) { p = p + n; }
  __host__ __device__ void increment() { p = p + 1; }
  __device__ reference dereference() const { 
    if (p->column_idx != col_index || p->flags & PAGEINFO_FLAGS_DICTIONARY) { 
      return const_cast<int32_t&>(empty);
    }
    return p->nesting[nesting_depth].page_start_value; 
  }
};
*/

/*
void print_names(std::vector<column_name_info> const &schema_info, std::string const& indent = "")
{
  for(size_t idx=0; idx<schema_info.size(); idx++){
    printf("%s%s\n", indent.c_str(), schema_info[idx].name.c_str());
    print_names(schema_info[idx].children, indent + "   ");
  }
}
*/