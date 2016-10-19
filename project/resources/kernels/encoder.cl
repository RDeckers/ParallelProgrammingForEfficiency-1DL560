#include "../resources/kernels/PipeLineConstants.h"


void d_RGB2YCbCr( float R,
                  float G,
                  float B,
                  float* Y,
                  float* Cb,
                  float* Cr)
{
    Y [0] = 0.0f +      (0.299f*R) +    (0.587f*G) +    (0.113f*B);
    Cb[0] = 128.0f - (0.168736f*R) - (0.331264f*G) +      (0.5f*B);
    Cr[0] = 128.0f +      (0.5f*R) - (0.418688f*G) - (0.081312f*B);
}

void d_conv( float tl,
	     float tm,
	     float tr,
	     float ll,
	     float mm,
	     float rr,
	     float bl,
	     float bm,
	     float br,
	     float* res)
{
  float corner =  1.0f/16.0f ;
  float neighb =  1.0f/8.0f ;
  float self =  1.0f/4.0f ;
  res[0] = corner * tl + neighb * tm +  corner * tr
    + neighb * ll +  self * mm + neighb * rr
    + corner * bl + neighb * bm +  corner * br;
}

__kernel void RGB2YCbCr_LowPassFilter_pipeline( __global float *g_R,
                                                __global float *g_G,
                                                __global float *g_B,
                                                __global float *g_Y,
                                                __global float *g_Cb,
                                                __global float *g_Cr,
                                                int rows,
                                                int cols)
{

  __local float lmem_Y [LMEM_Y][LMEM_X];
  __local float lmem_Cb[LMEM_Y][LMEM_X];
  __local float lmem_Cr[LMEM_Y][LMEM_X];

  // Global X&Y index:
  // One thread/WORK_ITEM per input and output element
  int g_tx = get_global_id(0);
  int g_ty = get_global_id(1);
  // Memory pitch:
  int pitch = get_global_size(0);
  // Group local X&Y index
  int l_tx = get_local_id(0);
  int l_ty = get_local_id(1);

  //
  // Read RGB and convert 2 YCbCr  --> LMEM
  //
  // Loop to include border values
  for(int j = 0; j < 2; j++)
  {
    for(int i = 0; i < 2; i++)
    {

      // First read top left corder (i-1, j-1)
      int x_idx = g_tx - 1 + j*DIM_X;
      int y_idx = g_ty - 1 + i*DIM_Y;

      if(   l_tx+j*DIM_X < LMEM_X   // Block border condition X
          &&l_ty+i*DIM_Y < LMEM_Y   // Block border condition Y
          )
      {

        // Check halo (border) values for entire frame,
        // if outside set to zero
        bool index_ok = (x_idx >=0 && x_idx < cols) && (y_idx >=0 && y_idx < rows);
        float R = 0.0f;
        float G = 0.0f;
        float B = 0.0f;

        float Y = 0.0f;
        float Cb = 0.0f;
        float Cr = 0.0f;

        if( index_ok)
        {
          R = g_R[x_idx + y_idx * pitch];
          G = g_G[x_idx + y_idx * pitch];
          B = g_B[x_idx + y_idx * pitch];
        }
        //
        // Convert to YCbCr and place in local memory directly
        d_RGB2YCbCr( R,G,B,
                    &Y ,
                    &Cb,
                    &Cr);

        // All threads write to a unique position in LMEM (no syncing needed)
        lmem_Y [ l_ty + i*DIM_Y ][ l_tx + j*DIM_X] = Y ;
        lmem_Cb[ l_ty + i*DIM_Y ][ l_tx + j*DIM_X] = Cb;
        lmem_Cr[ l_ty + i*DIM_Y ][ l_tx + j*DIM_X] = Cr;
      }
    } // END FOR 'i'
  } // END FOR 'j'
  //
  // Synchronize (ensure all threads in work group are done)
  //
  barrier(CLK_LOCAL_MEM_FENCE);
  //
  // Low pass filter X-direction
  //
  // Offset to area of intereens in LMEM (center part)
  const int offset = 1;
  // Cb
  float cb_tl = lmem_Cb[ (l_ty+offset) - 1 ][ (l_tx+offset) - 1];
  float cb_tm = lmem_Cb[ (l_ty+offset) - 1 ][ (l_tx+offset) ];
  float cb_tr = lmem_Cb[ (l_ty+offset) - 1 ][ (l_tx+offset) + 1];
  float cb_ll = lmem_Cb[ l_ty+offset ][ (l_tx+offset) - 1];
  float cb_mm = lmem_Cb[ l_ty+offset ][ (l_tx+offset) ];
  float cb_rr = lmem_Cb[ l_ty+offset ][ (l_tx+offset) + 1];
  float cb_bl = lmem_Cb[ (l_ty+offset) + 1 ][ (l_tx+offset) - 1];
  float cb_bm = lmem_Cb[ (l_ty+offset) + 1 ][ (l_tx+offset) ];
  float cb_br = lmem_Cb[ (l_ty+offset) + 1 ][ (l_tx+offset) + 1];

  float low_pass_cb = 0.0f;
  d_conv(cb_tl, cb_tm, cb_tr, cb_ll, cb_mm, cb_rr, cb_bl, cb_bm, cb_br, &low_pass_cb);

  // Cr
  float cr_tl = lmem_Cr[ (l_ty+offset) - 1 ][ (l_tx+offset) - 1];
  float cr_tm = lmem_Cr[ (l_ty+offset) - 1 ][ (l_tx+offset) ];
  float cr_tr = lmem_Cr[ (l_ty+offset) - 1 ][ (l_tx+offset) + 1];
  float cr_ll = lmem_Cr[ l_ty+offset ][ (l_tx+offset) - 1];
  float cr_mm = lmem_Cr[ l_ty+offset ][ (l_tx+offset) ];
  float cr_rr = lmem_Cr[ l_ty+offset ][ (l_tx+offset) + 1];
  float cr_bl = lmem_Cr[ (l_ty+offset) + 1 ][ (l_tx+offset) - 1];
  float cr_bm = lmem_Cr[ (l_ty+offset) + 1 ][ (l_tx+offset) ];
  float cr_br = lmem_Cr[ (l_ty+offset) + 1 ][ (l_tx+offset) + 1];

  float low_pass_cr = 0.0f;
  d_conv(cr_tl, cr_tm, cr_tr, cr_ll, cr_mm, cr_rr, cr_bl, cr_bm, cr_br, &low_pass_cr);

  barrier(CLK_LOCAL_MEM_FENCE);
  lmem_Cr[ (l_ty+offset) ][(l_tx+offset)] = low_pass_cr;
  lmem_Cb[ (l_ty+offset) ][(l_tx+offset)] = low_pass_cb;
  // Synchronize for good measure
  barrier(CLK_LOCAL_MEM_FENCE);
  // Output result to global memory
  // Only central portion is area of interest:
  g_Y[ g_tx + g_ty * pitch ]  = lmem_Y  [ (l_ty+offset) ][(l_tx+offset)];
  g_Cr[ g_tx + g_ty * pitch ] = lmem_Cr [ (l_ty+offset) ][(l_tx+offset)];
  g_Cb[ g_tx + g_ty * pitch ] = lmem_Cb [ (l_ty+offset) ][(l_tx+offset)];
}

//computes the sum of absolute differences between a 16x16 tile and the reference.
float compute_block_delta(uint x_loc, uint y_loc, __local float* refference, __local float const *buffer){
  float delta = 0;
  for(uint y = 0; y < 16; y++){
    uint y_buf = (y_loc+y);
    for(uint x = 0; x < 16; x++){
      uint x_buf = (x_loc+x);
      delta += fabs(buffer[x_buf+32*y_buf]-refference[x+16*y]);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  return delta;
}

//performs bounds checking
void load_block(int glob_id, int block_x, int block_y, __global float const *global_source, __local float *local_buffer){
  block_x = clamp(block_x, (int)0, (int)get_num_groups(0)-1);
  block_y = clamp(block_y, (int)0, (int)get_num_groups(1)-1);
  uint block_start = 16*(block_x+block_y*get_global_size(0));
  uint loc_adrress = get_local_id(0) + 32*get_local_id(1);
  uint glob_adress = glob_id;
  local_buffer[loc_adrress] = global_source[glob_adress];
  local_buffer[loc_adrress+16] = global_source[glob_adress+16];
  local_buffer[loc_adrress+00+32*16] = global_source[glob_adress+00+get_global_size(0)*16];
  local_buffer[loc_adrress+16+32*16] = global_source[glob_adress+16+get_global_size(0)*16];
  barrier(CLK_LOCAL_MEM_FENCE);
}

//performs a search over a specific channel and returns a float4 corresponding to the results.
//TODO: have each WG proccess a row instead of one tile. reduces global memory 'waste ratio'.
float4 motionVectorSearch_subroutine(
  int i_glob, int stride_glob,
  int x_loc, int y_loc, int i_loc,
  __global float const  *channel, __global float const *ref_channel,
  __local float *ref_block, //The buffer block for our old reference array.
  __local float *tile_buffer //buffers for our tiles
  ){
    int block_x = get_group_id(0);
    int block_y = get_group_id(1);

    float4 score;//each work-item has 4 out of 1024 its responsible for.
    //buffer our reference block into local memorchannel. (4bchanneltes per float, 3 floats per pixel, 256 pixels)
    ref_block[i_loc] = ref_channel[i_glob];

    //Buffer the compared blocks
    load_block(i_glob, block_x-1, block_y-1,channel, tile_buffer);
    score[0] = compute_block_delta(x_loc, y_loc, ref_block, tile_buffer);

    load_block(i_glob, block_x, block_y-1,channel, tile_buffer);
    score[1] = compute_block_delta(x_loc, y_loc,ref_block, tile_buffer);

    load_block(i_glob, block_x-1, block_y,channel, tile_buffer);
    score[2] = compute_block_delta(x_loc, y_loc, ref_block, tile_buffer);

    load_block(i_glob, block_x, block_y,channel, tile_buffer);
    score[3] = compute_block_delta(x_loc, y_loc, ref_block, tile_buffer);

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

    score += 0.50f*motionVectorSearch_subroutine(i_glob, stride_glob, x_loc, y_loc, i_loc, Y, ref_Y, ref_block, tile_buffer);
    score += 0.25f*motionVectorSearch_subroutine(i_glob, stride_glob, x_loc, y_loc, i_loc, Cr, ref_Cr, ref_block, tile_buffer);
    score += 0.25f*motionVectorSearch_subroutine(i_glob, stride_glob, x_loc, y_loc, i_loc, Cb, ref_Cb, ref_block, tile_buffer);

    score = score.xywz;
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
