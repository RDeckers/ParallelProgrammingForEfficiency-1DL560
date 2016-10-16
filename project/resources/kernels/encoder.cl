
#include "PipeLineConstants.h"
//
// Hard code kernel work group size, useful for static local memory allocation
//
//#define DIM_X = 64;
//#define DIM_Y = 8;
//// LMEM size must account for work filter overlap due to 
//#define LMEM_X = DIM_X+2;
//#define LMEM_Y = DIM_Y+2;


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
/*
  RGB2YCbCr_LowPassFilter
  Fused RGB conversion and low pass filter kernel

    ->Reads in 64x8 RGB + neighboor values, 
    ->computes conversion, places in local memory
    -> Computes low pass result for 64x8 values, data
      is shared via local memory
    -> Writes out 64x8 low pass filtered values 

    Most of this code is unfortunately not divided into device side functions
    as that seems to freak OpenCL out at times.

  @param g_R input
  @param g_G input
  @param g_B input
  @param g_Y  output
  @param g_Cb output
  @param g_Cr output
*/
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
  // Iterate Y to compute the 2 halo rows at the top and bottom
  for(int i = l_ty; i < LMEM_Y; i+=DIM_Y)
  {
    // Cb
    float cb_left   = lmem_Cb[ i ][ (l_tx+offset) - 1];
    float cb_middle = lmem_Cb[ i ][ (l_tx+offset) ];
    float cb_right  = lmem_Cb[ i ][ (l_tx+offset) + 1];
    // Cr
    float cr_left   = lmem_Cr[ i ][ (l_tx+offset) - 1];
    float cr_middle = lmem_Cr[ i ][ (l_tx+offset) ];
    float cr_right  = lmem_Cr[ i ][ (l_tx+offset) + 1];

    float low_pass_cr = 0.25f*cr_left + 0.5f*cr_middle + 0.25f*cr_right;
    float low_pass_cb = 0.25f*cb_left + 0.5f*cb_middle + 0.25f*cb_right;
    // Make sure all threads / WORK-items have read their values before update:
    barrier(CLK_LOCAL_MEM_FENCE);
    // Update LMEM ( central portion )
    lmem_Cr[ i ][ (l_tx+offset) ] = low_pass_cr;
    lmem_Cb[ i ][ (l_tx+offset) ] = low_pass_cb;
    // Synchronize for nex iter
    // FIXME: probably not need as we are switching rows (and they are data-independent here)
    // barrier(CLK_LOCAL_MEM_FENCE);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  //
  // Low pass filter Y-direction
  //
  // NOTE: we no longer need to care about halo left/right sides as no other 
  // steps require them
  //
  // Cb
  float cb_top      = lmem_Cb[ (l_ty+offset) - 1  ][ (l_tx+offset) ];
  float cb_middle   = lmem_Cb[ (l_ty+offset)      ][ (l_tx+offset) ];
  float cb_bottom   = lmem_Cb[ (l_ty+offset) + 1  ][ (l_tx+offset) ];
  // Cr
  float cr_top      = lmem_Cr[ (l_ty+offset) - 1  ][ (l_tx+offset) ];
  float cr_middle   = lmem_Cr[ (l_ty+offset)      ][ (l_tx+offset) ];
  float cr_bottom   = lmem_Cr[ (l_ty+offset) + 1  ][ (l_tx+offset) ];

  float low_pass_cr = 0.25f*cr_top + 0.5f*cr_middle + 0.25f*cr_bottom;
  float low_pass_cb = 0.25f*cb_top + 0.5f*cb_middle + 0.25f*cb_bottom;
  // Make sure all threads / WORK-items have read their values before update:
  barrier(CLK_LOCAL_MEM_FENCE);
  // Update LMEM ( central portion )
  lmem_Cr[ (l_ty+offset) ][(l_tx+offset)] = low_pass_cr;
  lmem_Cb[ (l_ty+offset) ][(l_tx+offset)] = low_pass_cb;
  // 
  // Synchronize for good measure
  //
  barrier(CLK_LOCAL_MEM_FENCE);
  //
  // Output result to global memory
  //
  // Only central portion is area of interest:
  g_Y[ g_tx + g_ty * pitch ]  = lmem_Y  [ (l_ty+offset) ][(l_tx+offset)];
  g_Cr[ g_tx + g_ty * pitch ] = lmem_Cr [ (l_ty+offset) ][(l_tx+offset)];
  g_Cb[ g_tx + g_ty * pitch ] = lmem_Cb [ (l_ty+offset) ][(l_tx+offset)];

} // END KERNEL


__kernel void convertRGBtoYCbCr(
    __global float *g_R, __global float *g_G, __global float *g_B,
    __global float *g_Y, __global float *g_Cb, __global float *g_Cr
  ){
    // Get the index of the current element to be processed
    int i = get_global_id(0);
    //
    // Read from global memory to registers
    //
    float R = g_R[i];
    float G = g_G[i];
    float B = g_B[i];
    float Y = 0.0f;
    float Cb = 0.0f;
    float Cr = 0.0f;
    //
    // Perform conversion
    // 
    d_RGB2YCbCr(R,G,B, &Y, &Cb, &Cr);
    //
    // Write to global memory
    //
    g_Y[i]  = Y ;
    g_Cb[i] = Cb;
    g_Cr[i] = Cr;

    /*
    Y[i] = 0.0f +      (0.299f*r) +    (0.587f*g) +    (0.113f*b);
    Cb[i] = 128.0f - (0.168736f*r) - (0.331264f*g) +      (0.5f*b);
    Cr[i] = 128.0f +      (0.5f*r) - (0.418688f*g) - (0.081312f*b);
    */
}

__kernel void lowPass_Y(__global const float *in, __global float *out/*, __local float *buffer*/){
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
