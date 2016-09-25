__kernel void set(
    __global float *A
  ){
    // Get the index of the current element to be processed
    int i = get_global_id(0);
    A[i] = i;
}
