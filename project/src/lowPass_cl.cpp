
Channel* lowPass_X_cl(Chanenl* in, Channel *out){
  // Applies a simple 3-tap low-pass filter in the X- and Y- dimensions.
  // E.g., blur
  // weights for neighboring pixels

   //enque writting the input
  cl_int ret = clEnqueueWriteBuffer(com_qs[0], mem_IN, CL_FALSE, 0, in->size_in_bytes(), in->data, 0, NULL, NULL);
  if(CL_SUCCESS != ret){
    report(FAIL, "clEnqueueWriteBuffer returned: %s (%d)", cluErrorString(ret), ret);
  }
 clFinish(com_qs[0]);

  //enque the kernel
  if(CL_SUCCESS != (ret = clEnqueueNDRangeKernel(com_qs[0], kernel, 1, NULL, &work_dim, &work_item_dim, 0, NULL, NULL))){
        report(FAIL, "enqueue kernel[0] returned: %s (%d)",cluErrorString(ret), ret);
        return;
  }
  clFinish(com_qs[0]);

  //enque reading the output
  ret = clEnqueueReadBuffer(com_qs[0], mem_OUT, CL_FALSE, 0, out->size_in_bytes(), out->data, 0, NULL, NULL);
  if(CL_SUCCESS != ret){
    report(FAIL, "clEnqueueReadBuffer returned: %s (%d)", cluErrorString(ret), ret);
  }
  clFinish(com_qs[0]);
  //wait for it to finish.
}

Channel* lowPass_Y_cl(Chanenl* in, Channel *out){
  // Applies a simple 3-tap low-pass filter in the X- and Y- dimensions.
  // E.g., blur
  // weights for neighboring pixels

   //enque writting the input
  cl_int ret = clEnqueueWriteBuffer(com_qs[0], mem_IN, CL_FALSE, 0, in->size_in_bytes(), in->data, 0, NULL, NULL);
  if(CL_SUCCESS != ret){
    report(FAIL, "clEnqueueWriteBuffer returned: %s (%d)", cluErrorString(ret), ret);
  }
 clFinish(com_qs[0]);

  //enque the kernel
  if(CL_SUCCESS != (ret = clEnqueueNDRangeKernel(com_qs[0], kernel, 1, NULL, &work_dim, &work_item_dim, 0, NULL, NULL))){
        report(FAIL, "enqueue kernel[0] returned: %s (%d)",cluErrorString(ret), ret);
        return;
  }
  clFinish(com_qs[0]);

  //enque reading the output
  ret = clEnqueueReadBuffer(com_qs[0], mem_OUT, CL_FALSE, 0, out->size_in_bytes(), out->data, 0, NULL, NULL);
  if(CL_SUCCESS != ret){
    report(FAIL, "clEnqueueReadBuffer returned: %s (%d)", cluErrorString(ret), ret);
  }
  clFinish(com_qs[0]);
  //wait for it to finish.
}
