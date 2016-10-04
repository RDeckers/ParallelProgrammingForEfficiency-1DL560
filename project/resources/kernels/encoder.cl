__kernel void convertRGBtoYCbCr(
    __global float *R, __global float *G, __global float *B
    //__global float *Y, __global float *Cb, __global float *Cr
  ){
    // Get the index of the current element to be processed
    int i = get_global_id(0);
    float r = R[i];
    float g = G[i];
    float b = B[i];
     R[i] = 0.0f +      (0.299f*r) +    (0.587f*g) +    (0.113f*b);
    G[i] = 128.0f - (0.168736f*r) - (0.331264f*g) +      (0.5f*b);
    B[i] = 128.0f +      (0.5f*r) - (0.418688f*g) - (0.081312f*b);
}
