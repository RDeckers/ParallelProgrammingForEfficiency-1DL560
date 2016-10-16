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

// Motion vector search:
//  Takes most time.
//  Load the relevant area around the current block in local mem.
//   *have each compute unit handle a starting pixel of a block (or several if needed).
//   *OR have each pixel in a tile be handled by a compute unit (16x16 block = 256 units)
//     bad idea.
//   then, for each tile/or compute unit we have a score. We need the best match. So we need to either sort them in some way on the GPU
//   or move back to cpu at this point.

//512 compute units,
//16x16 blocks.
//search space is -16 to +16 relative, so 32*32 pixels = 1024 pixels
//actually (16*3)^2 = 1024*9
//load 2 pixels per wi.
//optimize:
  //vectorize. Compare offset with swivels.
  //when done with our currentblock, don't clear local memory, but step right (keep blocks, left right center.)
  //experiment with block-sizes.
//12 bytes per pixel = ~128 KiB, local memory size = 64 Kib.
//have to separate channels.
//load
//native vector format is 4 float, SIMD
//difference computation is symmetric.

//instead
//load to be tested block into local memory. 16x16 = 256
//load two top left blocks into local memory. Compute first block comparison.

float compute_block_delta(int x_loc, int y_loc, __local float* refference, __local float *NW, __local float *SW, __local float *NE, __local float *SE){
  float *tiles[4] = {NW, SW, NE, SE};
  float delta = 0;
  for(int y = 0; y < 16; y++){
    int y_buf = y_loc+y;
    int y_overflow = (y_buf >= 16);
    for(int x = 0; x < 16; x++){
      int x_buf x_loc+x;
      int x_overflow = (x_buf >= 16);
      float *current_tile = tiles[y_overflow+2*x_overflow];
      delta += fabs(current_tile[(x_buf&15)+16*(y_buf&16)]-refference[x+16*y]);
    }
  }
  return delta;
}

__kernel void motionVectorSearch_subroutine(
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
    //make sure we're consistent
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
    int loc_size  = stride_loc*get_local_size(1);
    float4 score = {0,0,0,0};
    score += 0.50f*motionVectorSearch_subroutine(i_glob, stride_glob, x_loc, y_loc, i_loc, Y, ref_Y, ref_block, tile_buffer_1, tile_buffer_2, tile_buffer_3, tile_buffer_4, tile_buffer_5);
    score += 0.25f*motionVectorSearch_subroutine(i_glob, stride_glob, x_loc, y_loc, i_loc, Cr, ref_Cr, ref_block, tile_buffer_1, tile_buffer_2, tile_buffer_3, tile_buffer_4, tile_buffer_5);
    score += 0.25f*motionVectorSearch_subroutine(i_glob, stride_glob, x_loc, y_loc, i_loc, Cb, ref_Cb, ref_block, tile_buffer_1, tile_buffer_2, tile_buffer_3, tile_buffer_4, tile_buffer_5);
    //now find the minimum score of all the scores in the workgroup, and write it out.
    my_min_i = (min(score.0, score.1) <= min(score.2, score.3)) ? (score.1 <= score.1) : 2+(score.3 <= score.3);
    my_min = score[my_min_i];
    ref_block[i_loc] = my_min;
    barrier(CLK_LOCAL_MEM_FENCE);
    //TODO: reduction to find minimum element.

}
