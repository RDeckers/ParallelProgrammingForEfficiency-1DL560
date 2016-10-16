__kernel void convertRGBtoYCbCr(
    __global float *R, __global float *G, __global float *B,
    __global float *Y, __global float *Cb, __global float *Cr
  ){
    // Get the index of the current element to be processed
    int i = get_global_id(0);
    float r = R[i];
    float g = G[i];
    float b = B[i];
    Y[i] = 0.0f +      (0.299f*r) +    (0.587f*g) +    (0.113f*b);
    Cb[i] = 128.0f - (0.168736f*r) - (0.331264f*g) +      (0.5f*b);
    Cr[i] = 128.0f +      (0.5f*r) - (0.418688f*g) - (0.081312f*b);
}

__kernel void lowPass_Y(__global const float *in, __global float *out){
   float a = 0.25f;
  float b = 0.5f;
  float c = 0.25f;

  // Get the index of the current element to be processed
  int y_glob = get_global_id(0) + 1;
  int x_glob = get_global_id(1) + 1;
  int stride_glob = get_global_size(0) + 2;

  int i_glob = x_glob + y_glob*stride_glob;
  int i_glob_up = x_glob + (y_glob-1)*stride_glob;
  int i_glob_down = x_glob + (y_glob+1)*stride_glob;
  //
  float up= in[i_glob_up];
  float middle = in[i_glob];
  float down = in[i_glob_down];

  out[i_glob] = a*up + b*middle + c*down;
}

__kernel void lowPass_X(__global const float *in, __global float *out){
  float a = 0.25f;
  float b = 0.5f;
  float c = 0.25f;

  // Get the index of the current element to be processed
  int y_glob = get_global_id(0) + 1;
  int x_glob = get_global_id(1) + 1;
  int stride_glob = get_global_size(0) + 2;

  int i_glob  = x_glob + y_glob*stride_glob;
  int i_left  = x_glob + y_glob*stride_glob-1;
  int i_right = x_glob + y_glob*stride_glob+1;
  //
  float left = in[i_left];
  float middle = in[i_glob];
  float right = in[i_right];

  out[i_glob] = a*left + b*middle + c*right;
}

//computes the sum of absolute differences between a 16x16 tile and the reference.
float compute_block_delta(int x_loc, int y_loc, __local float* refference, __local float *NW, __local float *SW, __local float *NE, __local float *SE){
  float *tiles[4] = {NW, SW, NE, SE};//hacky way to select the write tile to read from without if's.
  float delta = 0;
  for(int y = 0; y < 16; y++){
    int y_buf = y_loc+y;
    int y_overflow = (y_buf >= 16);//return 1 or 0.
    for(int x = 0; x < 16; x++){
      int x_buf x_loc+x;
      int x_overflow = (x_buf >= 16);
      float *current_tile = tiles[y_overflow+2*x_overflow];//gives an index in [0,1,2,3] corresponding to the overflow in x and y
      delta += fabs(current_tile[(x_buf&15)+16*(y_buf&16)]-refference[x+16*y]);
    }
  }
  return delta;
}

//performs a search over a specific channel and returns a float4 corresponding to the results.
float4 motionVectorSearch_subroutine(
  int i_glob, int stride_glob,
  int x_loc, int y_loc, int i_loc,
  __global float *channel, __global float *ref_channel,
  __local float *ref_block, //The buffer block for our old reference array.
  __local float *tile_buffer_1, __local float *tile_buffer_2, __local float *tile_buffer_3, __local float *tile_buffer_4, __local float *tile_buffer_5 //buffers for our tiles
  ){
    float4 score;//each work-item has 4 out of 1024 its responsible for.
    //buffer our reference block into local memorchannel. (4bchanneltes per float, 3 floats per pixel, 256 pixels)
    ref_block[i_loc] = ref_channel[i_glob];
    //Buffer the compared blocks
    tile_buffer_1[i_loc] = channel[i_glob]; //NW
    tile_buffer_2[i_loc] = channel[i_glob+16*stride_glob];//SW
    tile_buffer_3[i_loc] = channel[i_glob+16];//NE
    tile_buffer_4[i_loc] = channel[i_glob+16+16*stride_glob];//SE
    //make sure all writes to local memory have finished
    barrier(CLK_LOCAL_MEM_FENCE);
    score.0= compute_block_delta(x_loc, channel_loc, ref_block, tile_buffer_1, tile_buffer_2, tile_buffer_3, tile_buffer_4);
    tile_buffer_5[i_loc] = channel[i_glob+32];
    tile_buffer_1[i_loc] = channel[i_glob+32+16*stride_glob];
    barrier(CLK_LOCAL_MEM_FENCE);
    score.1 = compute_block_delta(x_loc, channel_loc, ref_block, tile_buffer_3, tile_buffer_4, tile_buffer_5, tile_buffer_1);
    tile_buffer_3[i_loc] = channel[i_glob+32*stride_glob];
    tile_buffer_5[i_loc] = channel[i_glob+16+32*stride_glob];
    barrier(CLK_LOCAL_MEM_FENCE);
    score.2 = compute_block_delta(x_loc, channel_loc, ref_block, tile_buffer_2, tile_buffer_3, tile_buffer_4, tile_buffer_5);
    tile_buffer_2[i_loc] = channel[i_glob+32+32*stride_glob];
    barrier(CLK_LOCAL_MEM_FENCE);
    score.3 = compute_block_delta(x_loc, channel_loc, ref_block, tile_buffer_4, tile_buffer_5, tile_buffer_1, tile_buffer_5);
    return score;
  }

__kernel void motionVectorSearch(
    __global float *Y, __global float *Cr, __global float *Cb,//the channels of our new frame
    __global float *ref_Y, __global float *ref_Cr, __global float *ref_Cb, //channels of out reference frame
    __local float *ref_block, //The buffer block for our old reference array.
    __local float *tile_buffer_1, __local float *tile_buffer_2, __local float *tile_buffer_3, __local float *tile_buffer_4, __local float *tile_buffer_5, //buffers for our tiles
    __global float4 *scores;
  ){
    //TODO: add edge-case conditions.
    //TODO: verify
    // Get the index of the current element to be processed
    int x_glob = get_global_id(0);
    int y_glob = get_global_id(1);
    int stride_glob = get_global_size(0);
    int i_glob = x_glob+y_glob*stride_glob;

    //Get the local index
    int x_loc = get_local_id(0);
    int y_loc = get_local_id(1);
    int stride_loc = get_local_size(0);
    int i_loc = x_loc+y_loc*stride_loc;

    float4 score = {0,0,0,0};//score here tracks which offset gives the best motion vector, lower is better.
    score += 0.50f*motionVectorSearch_subroutine(i_glob, stride_glob, x_loc, y_loc, i_loc, Y, ref_Y, ref_block, tile_buffer_1, tile_buffer_2, tile_buffer_3, tile_buffer_4, tile_buffer_5);
    score += 0.25f*motionVectorSearch_subroutine(i_glob, stride_glob, x_loc, y_loc, i_loc, Cr, ref_Cr, ref_block, tile_buffer_1, tile_buffer_2, tile_buffer_3, tile_buffer_4, tile_buffer_5);
    score += 0.25f*motionVectorSearch_subroutine(i_glob, stride_glob, x_loc, y_loc, i_loc, Cb, ref_Cb, ref_block, tile_buffer_1, tile_buffer_2, tile_buffer_3, tile_buffer_4, tile_buffer_5);
    //now find the minimum score of all the scores in the workgroup, and write it out.
    my_min_i = (min(score.0, score.1) <= min(score.2, score.3)) ? (score.1 <= score.1) : 2+(score.3 <= score.3);
    my_min = score[my_min_i];
    ref_block[i_loc] = my_min;//reuse the ref_block for storing our minimum values, then reduce in local memory to get actual minium
    barrier(CLK_LOCAL_MEM_FENCE);
    int loc_size  = stride_loc*get_local_size(1);
    //TODO: reduction to find minimum element. store it's index in global memory

}
