#include "../resources/kernels/common/pipeline_kernels.cl"

//computes the sum of absolute differences between a 16x16 tile and the reference.
float4 compute_block_delta(uint x_loc, uint y_loc, __local float* refference, __local float const *buffer){
  // uint x_vec_loc = 4*(x_loc/2);//[0,7] //where this wi starts
  // uint y_vec_loc = y_loc*2+x_loc&1;//[0,31]
  uint i_loc = x_loc+y_loc*16;//[0, 16x16 = 256 = 2^8]
  uint x_vec_loc = 4*(i_loc%8);//4*[0,7] -> 28 max, final vec element = 31
  uint y_vec_loc = (i_loc/8);//2^8/2^3 = 32.
  //TODO: try mod/rem above to keep pattern 'nicer'
  float4 delta = {0,0,0,0};
  for(uint y = 0; y < 16; y++){
    uint y_buf = y_vec_loc+y;
    for(uint x = 0; x < 16; x++){
      uint x_buf = x_vec_loc+x;
      float4 ref = refference[x+16*y];
      float4 buff = *((__local float4*)(buffer+x_buf+48*y_buf));
      delta += fabs(buff-ref);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  return delta;
}

//performs bounds checking
void load_block(uint x_glob, uint y_glob, uint glob_stride, __global float const *global_source, __local float *local_buffer){
  uint glob_i = x_glob+glob_stride*y_glob;
  uint i_loc = get_local_id(0)+48*get_local_id(1);

  //row one
  local_buffer[i_loc+00+48*0] = global_source[glob_i+00+glob_stride*0];
  local_buffer[i_loc+16+48*0] = global_source[glob_i+16+glob_stride*0];
  local_buffer[i_loc+32+48*0] = global_source[glob_i+32+glob_stride*0];
  //row 2
  local_buffer[i_loc+00+48*16] = global_source[glob_i+00+glob_stride*16];
  local_buffer[i_loc+16+48*16] = global_source[glob_i+16+glob_stride*16];
  local_buffer[i_loc+32+48*16] = global_source[glob_i+32+glob_stride*16];
  //row 3
  local_buffer[i_loc+00+48*32] = global_source[glob_i+00+glob_stride*32];
  local_buffer[i_loc+16+48*32] = global_source[glob_i+16+glob_stride*32];
  local_buffer[i_loc+32+48*32] = global_source[glob_i+32+glob_stride*32];
  barrier(CLK_LOCAL_MEM_FENCE);
}

//performs a search over a specific channel and returns a float4 corresponding to the results.
//TODO: have each WG proccess a row instead of one tile. reduces global memory 'waste ratio'.
float4 motionVectorSearch_subroutine(
  uint x_glob, uint y_glob, uint i_glob, uint stride_glob,
  uint x_loc, uint y_loc, uint i_loc,
  __global float const  *channel, __global float const *ref_channel,
  __local float *ref_block, //The buffer block for our old reference array.
  __local float *tile_buffer //buffers for our tiles
  ){
    float4 score;//each work-item has 4 out of 1024 its responsible for.
    //buffer our reference block into local memorchannel. (4bchanneltes per float, 3 floats per pixel, 256 pixels)
    ref_block[i_loc] = ref_channel[i_glob];

    //Buffer the compared blocks
    load_block(x_glob-16, y_glob -16, stride_glob, channel, tile_buffer);
    score = compute_block_delta(x_loc, y_loc, ref_block, tile_buffer);

    return score;
  }

__kernel void motionVectorSearch(
    __global float const *ref_Y, __global float const *ref_Cb, __global float const *ref_Cr,//the channels of our new frame
    __global float const *Y, __global float const *Cb, __global float const *Cr, //channels of out reference frame
    __global int *indices
  ){
    __local float ref_block[16*16];
    __local float tile_buffer[48*48];
    //10KiB total ^
    //TODO: verify
    // Get the index of the current element to be processed
    int x_glob = get_global_id(0)+16;//no borders,
    int y_glob = get_global_id(1)+16;//so shift by 16.
    int stride_glob = get_global_size(0)+32;//we also have to jump an additional 16 places on each side.
    int i_glob = x_glob+y_glob*stride_glob;//our index in global memory

    //Get the local index
    int x_loc = get_local_id(0);
    int y_loc = get_local_id(1);
    int stride_loc = get_local_size(0);
    int i_loc = x_loc+y_loc*stride_loc;

    float4 score = {0,0,0,0};//score here tracks which offset gives the best motion vector, lower is better.

    score += 0.50f*motionVectorSearch_subroutine(x_glob, y_glob, i_glob, stride_glob, x_loc, y_loc, i_loc, Y, ref_Y, ref_block, tile_buffer);
    score += 0.25f*motionVectorSearch_subroutine(x_glob, y_glob, i_glob, stride_glob, x_loc, y_loc, i_loc, Cr, ref_Cr, ref_block, tile_buffer);
    score += 0.25f*motionVectorSearch_subroutine(x_glob, y_glob, i_glob, stride_glob, x_loc, y_loc, i_loc, Cb, ref_Cb, ref_block, tile_buffer);

    //find the minimum element of our vector and it's place.
    int my_min_i = (min(score[0], score[1]) <= min(score[2], score[3])) ? (score[1] <= score[0]) : 2+(score[3] <= score[2]);
    float my_min = min(min(score.x, score.y), min(score.z, score.w));

    //reuse one of the buffers as an index buffer
    __local int* index_buffer = (__local int*) tile_buffer;
    //should be [-16,15]
    //reshift to [0,32]x[0,32] and flatten
    //conver back to proper coordinates on cpu.
    //uint i_loc = x_loc+y_loc*16;
    uint x_vec_loc = 4*(i_loc%8);
    uint y_vec_loc = (i_loc/8);
    int my_index = (x_vec_loc+my_min_i)+32*(y_vec_loc);

    ref_block[i_loc] = my_min;//reuse the ref_block for storing our minimum values, then reduce in local memory to get actual minium
    index_buffer[i_loc] = my_index;//reuse the tile_buffer_1 as an interger storage for the index.
    barrier(CLK_LOCAL_MEM_FENCE);

    // TODO: test, vectorize, compare with readback, unroll loop.
    // 16x16 = 256 elements, we start at 128. so log2(128) iterations = 128->64->32->16->8->4->2->1 = 7 iterations. Less with vectorization.
    for(uint participants = 128; participants >= 1; participants /= 2){
      if(i_loc < participants){
        float my_new = ref_block[i_loc+participants];
        int my_new_index = index_buffer[i_loc+participants];
        //no barrier needed here, because all the writes in this section are not read until the next one.
        int selection_mask = my_min < my_new;
        my_min = select(my_new, my_min, selection_mask);
        my_index = select(my_new_index, my_index, selection_mask);
        ref_block[i_loc] = my_min;
        index_buffer[i_loc] = my_index;
      }
      barrier(CLK_LOCAL_MEM_FENCE);//barrier outside if so everyone hits it.
    }
    //now the first work-item will have computed the minium.
    if(i_loc ==0){
      int group_id = get_group_id(0)+get_group_id(1)*get_num_groups(0);
      indices[group_id] = my_index;
    }
}
