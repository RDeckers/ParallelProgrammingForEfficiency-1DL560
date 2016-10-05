__kernel void lowPass_Y(__global const float *IN, __global float *OUT){
   float a = 0.25;
  float b = 0.5;
  float c = 0.25;
  
  // Get the index of the current element to be processed
  int y_glob = get_global_id(0) + 1;
  int x_glob = get_global_id(1) + 1;
  int stride_glob = get_global_size(0) + 2;
  
  int i_glob = y_glob + x_glob*stride_glob;  
  int i_up = y_glob + x_glob*stride_glob - 1;
  int i_down = y_glob + x_glob*stride_glob + 1;
  //    
  float upp = IN[i_up];
  float mid = IN[i_glob];
  float dow = IN[i_down];
  
  OUT[i] = a*upp + b*mid + c*dow;
}

__kernel void lowPass_X(__global const float *IN, __global float *OUT){
  float a = 0.25;
  float b = 0.5;
  float c = 0.25;
  
  // Get the index of the current element to be processed
  int y_glob = get_global_id(0) + 1;
  int x_glob = get_global_id(1) + 1;
  int stride_glob = get_global_size(0) + 2;
  
  int i_glob = y_glob + x_glob*stride_glob;  
  int i_left = y_glob + (x_glob-1)*stride_glob;
  int i_right = y_glob + (x_glob+1)*stride_glob;
  //    
  float lef = IN[i_left];
  float mid = IN[i_glob];
  float rig = IN[i_right];
  
  OUT[i] = a*lef + b*mid + c*rig;
}
