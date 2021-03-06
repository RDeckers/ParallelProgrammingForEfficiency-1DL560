#include "../resources/kernels/common/pipeline_kernels.cl"

//computes the sum of absolute differences between a 16x16 tile and the reference.
float compute_block_delta(int x_loc, int y_loc, int start_x, int start_y, __local float* refference, __local float* buffer){
  float delta = 0;
  x_loc += start_x;
  y_loc += start_y;
  for(int y = 0; y < 16; y++){
    int y_buf = (y_loc+y)%32;
    for(int x = 0; x < 16; x++){
      int x_buf = (x_loc+x)%32;
      delta += fabs(buffer[x_buf+32*y_buf]-refference[x+16*y]);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  return delta;
}

//performs bounds checking
void load_block(int loc_id, int block_x, int block_y, int target_offset_x, int target_offset_y,__global float const *global_source, __local float *local_buffer){
  int block_start = 16*(block_x+block_y*(get_global_size(0)+32));
  local_buffer[get_local_id(0)+16*target_offset_x + 32*(get_local_id(1)+16*target_offset_y)]
    = global_source[block_start+get_local_id(0)+get_local_id(1)*(get_global_size(0)+32)];
}

float4 motionVectorSearch_subroutine(
  uint x_glob, uint y_glob, uint i_glob, uint stride_glob,
  uint x_loc, uint y_loc, uint i_loc,
  __global float const  *channel, __global float const *ref_channel,
  __local float *ref_block, //The buffer block for our old reference array.
  __local float *tile_buffer //buffers for our tiles
  ){
    int block_x = get_group_id(0)+1;
    int block_y = get_group_id(1)+1;
    float4 score;//each work-item has 4 out of 1024 its responsible for.
    //buffer our reference block into local memorchannel. (4bchanneltes per float, 3 floats per pixel, 256 pixels)
    ref_block[i_loc] = ref_channel[i_glob];
    //Buffer the compared blocks
    //TODO: make one function call.
    load_block(i_loc, block_x-1, block_y-1, 0, 0, channel, tile_buffer);//0
    load_block(i_loc, block_x-1, block_y-0, 0, 1, channel, tile_buffer);//1
    load_block(i_loc, block_x-0, block_y-1, 1, 0, channel, tile_buffer);//2
    load_block(i_loc, block_x-0, block_y-0, 1, 1, channel, tile_buffer);//3
    barrier(CLK_LOCAL_MEM_FENCE);
    score[0] = compute_block_delta(x_loc, y_loc, 0,0, ref_block, tile_buffer);

    load_block(i_loc, block_x+1, block_y-1, 0, 0, channel, tile_buffer);//4
    load_block(i_loc, block_x+1, block_y-0, 0, 1, channel, tile_buffer);//5
    barrier(CLK_LOCAL_MEM_FENCE);
    score[1] = compute_block_delta(x_loc, y_loc, 16,0, ref_block, tile_buffer);

    load_block(i_loc, block_x+0, block_y+1, 1, 0, channel, tile_buffer);//6
    load_block(i_loc, block_x+1, block_y+1, 0, 0, channel, tile_buffer);//7
    barrier(CLK_LOCAL_MEM_FENCE);
    score[3] = compute_block_delta(x_loc, y_loc, 16,16, ref_block, tile_buffer);

    load_block(i_loc, block_x-1, block_y+1, 0, 0, channel, tile_buffer);//8
    load_block(i_loc, block_x-1, block_y-0, 0, 1, channel, tile_buffer);//1
    barrier(CLK_LOCAL_MEM_FENCE);
    score[2] = compute_block_delta(x_loc, y_loc, 0, 16, ref_block, tile_buffer);

    return score;
}

__kernel void motionVectorSearch(
    __global float const *ref_Y, __global float const *ref_Cb, __global float const *ref_Cr,//the channels of our new frame
    __global float const *Y, __global float const *Cb, __global float const *Cr, //channels of out reference frame
    __global int *indices
  ){
    __local float ref_block[16*16];
    __local float tile_buffer[32*32];
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
    int my_index = (x_loc+16*(my_min_i&1))+32*(y_loc+8*(my_min_i&2));

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
    if(i_loc == 0){
      int group_id = get_group_id(0)+get_group_id(1)*get_num_groups(0);
      indices[group_id] = my_index;
    }
}
