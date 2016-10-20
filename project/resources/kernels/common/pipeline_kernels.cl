#include "./PipeLineConstants.h"

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
