#include "../resources/kernels/common/pipeline_kernels.cl"

//computes the sum of absolute differences between a 16x16 tile and the reference.
float compute_block_delta(uint x_glob, uint y_glob, uint stride_glob, __local float* refference, __global float const *buffer){
  float delta = 0;
  for(uint y = 0; y < 16; y++){
    uint y_buf = (y_glob+y);
    for(uint x = 0; x < 16; x++){
      uint x_buf = (x_glob+x);
      delta += fabs(buffer[x_buf+y_buf*stride_glob]-refference[x+16*y]);
    }
  }
  return delta;
}

//performs a search over a specific channel and returns a float4 corresponding to the results.
//TODO: have each WG proccess a row instead of one tile. reduces global memory 'waste ratio'.
float4 motionVectorSearch_subroutine(
  uint x_glob, uint y_glob, uint stride_glob,
  uint x_loc, uint y_loc,
  __global float const  *channel, __global float const *ref_channel,
  __local float *ref_block//The buffer block for our old reference array.
  ){
    float4 score;//each work-item has 4 out of 1024 its responsible for.
    //buffer our reference block into local memorchannel. (4bchanneltes per float, 3 floats per pixel, 256 pixels)
    ref_block[x_loc+16*y_loc] = ref_channel[x_glob+stride_glob*y_glob];
    barrier(CLK_LOCAL_MEM_FENCE);
    //Buffer t he compared blocks
    score[0] = compute_block_delta(x_glob-16, y_glob-16, stride_glob, ref_block, channel);
    score[1] = compute_block_delta(x_glob, y_glob-16, stride_glob, ref_block, channel);
    score[2] = compute_block_delta(x_glob-16, y_glob, stride_glob, ref_block, channel);
    score[3] = compute_block_delta(x_glob, y_glob, stride_glob,  ref_block, channel);
    barrier(CLK_LOCAL_MEM_FENCE); //might otherwise overwrite ref_block in the next computation
    return score;
  }

__kernel void motionVectorSearch(
    __global float const *ref_Y, __global float const *ref_Cb, __global float const *ref_Cr,//the channels of our new frame
    __global float const *Y, __global float const *Cb, __global float const *Cr, //channels of out reference frame
    __global int *indices
  ){
    __local uint index_buffer[16*16];
    __local float ref_block[16*16];
    //TODO: verify
    // Get the index of the current element to be processed
    uint x_glob = get_global_id(0)+16;
    uint y_glob = get_global_id(1)+16;
    uint stride_glob = get_global_size(0)+32;

    //Get the local index
    uint x_loc = get_local_id(0);
    uint y_loc = get_local_id(1);

    float4 score = {0,0,0,0};//score here tracks which offset gives the best motion vector, lower is better.

    score += 0.50f*motionVectorSearch_subroutine(x_glob, y_glob, stride_glob, x_loc, y_loc, Y, ref_Y, ref_block);
    score += 0.25f*motionVectorSearch_subroutine(x_glob, y_glob, stride_glob, x_loc, y_loc, Cr, ref_Cr, ref_block);
    score += 0.25f*motionVectorSearch_subroutine(x_glob, y_glob, stride_glob, x_loc, y_loc, Cb, ref_Cb, ref_block);

    //find the minimum element of our vector and it's place.
    int my_min_i = (min(score[0], score[1]) <= min(score[2], score[3])) ? (score[1] <= score[0]) : 2+(score[3] <= score[2]);
    float my_min = min(min(score.x, score.y), min(score.z, score.w));

    //reuse one of the buffers as an index buffer

    //should be [-16,15]
    //reshift to [0,32]x[0,32] and flatten
    //conver back to proper coordinates on cpu.
    uint my_index = (x_loc+16*(my_min_i&1))+32*(y_loc+8*(my_min_i&2));

    uint i_loc = x_loc+get_local_size(0)*y_loc;
    ref_block[i_loc] = my_min;//reuse the ref_block for storing our minimum values, then reduce in local memory to get actual minium
    index_buffer[i_loc] = my_index;//reuse the tile_buffer_1 as an interger storage for the index.

    // TODO: test, vectorize, compare with readback, unroll loop.
    // 16x16 = 256 elements, we start at 128. so log2(128) iterations = 128->64->32->16->8->4->2->1 = 7 iterations. Less with vectorization.
    for(uint participants = 128; participants >= 1; participants /= 2){
      barrier(CLK_LOCAL_MEM_FENCE);
      if(i_loc < participants){
        float my_new = ref_block[i_loc+participants];
        uint my_new_index = index_buffer[i_loc+participants];
        //no barrier needed here, because all the writes in this section are not read until the next one.
        uint selection_mask = my_min < my_new;
        my_min = select(my_new, my_min, selection_mask);
        my_index = select(my_new_index, my_index, selection_mask);
        ref_block[i_loc] = my_min;
        index_buffer[i_loc] = my_index;
      }
      //barrier(CLK_LOCAL_MEM_FENCE);//barrier outside if so everyone hits it.
    }
    //now the first work-item will have computed the minium.
    if(i_loc == 0){
      int group_id = get_group_id(0)+get_group_id(1)*get_num_groups(0);
      indices[group_id] = my_index;
    }
}
