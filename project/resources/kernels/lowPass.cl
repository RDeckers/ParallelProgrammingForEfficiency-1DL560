__kernel void lowPass_Y(__global const float *in, __global float *out){
   float a = 0.25;
  float b = 0.5;
  float c = 0.25;

  // Get the index of the current element to be processed
  int y_glob = get_global_id(0) + 1;
  int x_glob = get_global_id(1) + 1;
  int stride_glob = get_global_size(0) + 2;

  int i_glob = x_glob + y_glob*stride_glob;
  int i_glob_up = x_glob + (y_glob-1)*stride_glob;
  int i_glob_down = x_glob + (y_glob+1*stride_glob;
  //
  float up= in[i_glob_up];
  float middle = in[i_glob];
  float down = in[i_glob_down];

  out[i] = a*up + b*middle + c*down;
}

__kernel void lowPass_X(__global const float *in, __global float *out){
  float a = 0.25;
  float b = 0.5;
  float c = 0.25;

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

  out[i] = a*left + b*middle + c*right;
}
