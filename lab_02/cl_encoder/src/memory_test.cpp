#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <string>
#include <tiff.h>
#include <tiffio.h>
#include "custom_types.h"
#include <vector>
#include "config.h"
#include <clu.h>
#include <utilities/file.h>
#include <utilities/logging.h>
#include <utilities/benchmarking.h>
#include <algorithm>

using namespace std;

cl_command_queue *com_qs = NULL;
cl_kernel kernel;
size_t work_dim;
size_t work_item_dim = 256;
cl_mem mem_A;

//can be optimized by saving bytes instead of floats and converting dynamically.
//also should load next frame async in the background.
void fill_buffer(cl_float *buffer) {
  for(unsigned u = 0; u < work_dim; u++){
    buffer[u] = 1;
  }
}

void upload_to_GPU_blocking(cl_float *buffer){
  fill_buffer(buffer);
  cl_int ret = clEnqueueWriteBuffer(com_qs[0], mem_A, CL_TRUE, 0, work_dim*sizeof(buffer[0]), buffer, 0, NULL, NULL);
  if(CL_SUCCESS != ret){
    report(FAIL, "clEnqueueWriteBuffer returned: %s (%d)", cluErrorString(ret), ret);
  }
  //clFinish(com_qs[0]);
}
void upload_to_GPU_pin(cl_float *buffer){
  cl_int ret;
  float* mappedBuffer = (float *)clEnqueueMapBuffer(com_qs[0], mem_A, CL_TRUE, CL_MAP_WRITE, 0, work_dim*sizeof(float), 0, NULL, NULL, &ret);
  if(CL_SUCCESS != ret){
    report(FAIL, "clEnqueueMapBuffer returned: %s (%d)", cluErrorString(ret), ret);
  }
  fill_buffer(mappedBuffer);
  clEnqueueUnmapMemObject(com_qs[0], mem_A, mappedBuffer, 0, NULL, NULL);
}

void run_kernel(){
  //enque the kernel
  cl_int ret;
  if(CL_SUCCESS != (ret = clEnqueueNDRangeKernel(com_qs[0], kernel, 1, NULL, &work_dim, &work_item_dim, 0, NULL, NULL))){
        report(FAIL, "enqueue kernel[0] returned: %s (%d)",cluErrorString(ret), ret);
        return;
  }
  clFinish(com_qs[0]);
}

void readback_blocking(cl_float *buffer){
    //enque reading the output
    cl_int ret;
    ret = clEnqueueReadBuffer(com_qs[0], mem_A, CL_TRUE, 0, work_dim*sizeof(buffer[0]), buffer, 0, NULL, NULL);
    if(CL_SUCCESS != ret){
      report(FAIL, "clEnqueueReadBuffer returned: %s (%d)", cluErrorString(ret), ret);
    }
    for(unsigned c = 0; c < work_dim; c++){
      if(c != buffer[c]){
        report(FAIL, "failed at %u", c);
      }
    }
    //clFinish(com_qs[0]);
    //wait for it to finish.
}

void readback_pin(cl_float *buffer){
  cl_int ret;
  float* mappedBuffer = (float *)clEnqueueMapBuffer(com_qs[0], mem_A, CL_TRUE, CL_MAP_READ, 0, work_dim*sizeof(float), 0, NULL, NULL, &ret);
  if(CL_SUCCESS != ret){
    report(FAIL, "clEnqueueMapBuffer returned: %s (%d)", cluErrorString(ret), ret);
  }
  for(unsigned c = 0; c < work_dim; c++){
    if(c != mappedBuffer[c]){
      report(FAIL, "failed at %u", c);
    }
  }
  clEnqueueUnmapMemObject(com_qs[0], mem_A, mappedBuffer, 0, NULL, NULL);
}


int encode() {
  REPORT_W_COLORS = 1;
  REPORT_W_TIMESTAMPS = 1;
  set_cwdir_to_bin_dir();

  size_t iterations = 100;
  size_t min_buffer_size_log2 = 10;
  size_t max_buffer_size_log2 = 29;
  size_t max_buffer_size = 1 << max_buffer_size_log2;
  cl_float *buffer = (cl_float*) _mm_malloc(sizeof(cl_float)*max_buffer_size, 4096);
  if(!buffer){
    report(FAIL, "failed to allocate buffer of %u elements!", max_buffer_size);
    return -1;
  }

  //struct timeval starttime, endtime;
  struct timespec clock;
  double upload_t[iterations] ={0};
  double kernel_t[iterations] = {0};
  double readback_t[iterations] = {0};
  double total_t[iterations] = {0};



/*////////////////////
// CL INIT
////////////////////*/
  tick(&clock);
  cl_platform_id *platforms = NULL;
    cl_device_id *devices = NULL;
    cl_uint n_platforms = cluGetPlatforms(&platforms, CLU_DYNAMIC);
    if(!n_platforms){
      report(FAIL, "No OpenCL platforms found!");
      return -1;
    }
    cl_context context;
    int device_count = 0;
    for(unsigned p = 0; p < n_platforms; p++){
      if(device_count = cluGetDevices(platforms[p], CL_DEVICE_TYPE_GPU, CLU_DYNAMIC, &devices)){
        context = cluCreateContextFromTypes(platforms[p], CL_DEVICE_TYPE_GPU);
        break;
      }
    }
    if(!device_count){
      report(WARN, "No GPU system found, falling back to CPU");
      for(unsigned p = 0; p < n_platforms; p++){
        if(device_count = cluGetDevices(platforms[p], CL_DEVICE_TYPE_CPU, CLU_DYNAMIC, &devices)){
          context = cluCreateContextFromTypes(platforms[p], CL_DEVICE_TYPE_CPU);
          break;
        }
      }
    }
    if(!device_count){
      report(FAIL, "No GPU or CPUs found, but a platform was found...");
      return -1;
    }
    report(PASS, "Created a context with %d device(s)", device_count);
    cluCreateCommandQueues(context, devices, device_count, &com_qs);
    report(PASS, "Command queues created");

    cl_int ret;
    mem_A = clCreateBuffer(context, CL_MEM_READ_WRITE, max_buffer_size*sizeof(float), nullptr, &ret);
    if(CL_SUCCESS != ret){
      report(FAIL, "clCreateBuffer (R) returned: %s (%d)", cluErrorString(ret), ret);
    }

    char *program_src = NULL;
    cl_program program = cluProgramFromFilename(context, "../resources/kernels/set.cl");
    ret = clBuildProgram(program, device_count, devices, NULL, NULL, NULL);
    if(CL_SUCCESS != ret){
      report(FAIL, "clBuildProgram returned: %s (%d)", cluErrorString(ret), ret);
      char *log = NULL;
    for(int i = 0; i < device_count; i++){
      cluGetProgramLog(program, devices[0], CLU_DYNAMIC, &log);
      report(INFO, "log device[%d]\n==============\n%s",i, log);
    }
    free(log);
    }

    kernel = clCreateKernel(program, "set", &ret);
    if(CL_SUCCESS != ret){
      report(FAIL, "clCreateKernel returned: %s (%d)", cluErrorString(ret), ret);
    }

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&mem_A);
    if(CL_SUCCESS != ret){
      report(FAIL, "clSetKernelArg returned: %s (%d)", cluErrorString(ret), ret);
    }

    /*/////////////////
    // END OPENCL INIT
    ///////////////*/

    report(PASS, "Setup OpenCL");
    for(unsigned buffer_size_log2 = min_buffer_size_log2; buffer_size_log2 < max_buffer_size_log2; buffer_size_log2++){
      report(INFO, "starting on log2 buffer: %u", buffer_size_log2);
      work_dim = 1 << buffer_size_log2;
      for(unsigned i = 0; i < iterations; i++){
        tick(&clock);
        upload_to_GPU_blocking(buffer);
        upload_t[i] = elapsed_since(&clock);
        tick(&clock);
        run_kernel();
        kernel_t[i] = elapsed_since(&clock);
        tick(&clock);
        readback_blocking(buffer);
        readback_t[i] = elapsed_since(&clock);
        total_t[i] = upload_t[i] + kernel_t[i] + readback_t[i];
      }
      std::sort(upload_t, upload_t+iterations);
      std::sort(kernel_t, kernel_t+iterations);
      std::sort(readback_t, readback_t+iterations);
      std::sort(total_t, total_t+iterations);
      printf("%u %.4e %.4e %.4e %.4e\n", work_dim, upload_t[0], kernel_t[0], readback_t[0], total_t[0]);
    }printf("\n\n");
    report(INFO, "Effective bandwith: %3.2f GB/s", (max_buffer_size*4*2)/(total_t[0]));
    report(INFO, "Real bandwith: %3.2f GB/s up,  %3.2f GB/s down", (max_buffer_size*4)/(upload_t[0]), (max_buffer_size*4)/(readback_t[0]));

    clReleaseMemObject(mem_A);
    mem_A = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, max_buffer_size*sizeof(float), buffer, &ret);
    if(CL_SUCCESS != ret){
      report(FAIL, "clCreateBuffer (R) returned: %s (%d)", cluErrorString(ret), ret);
    }
    for(unsigned buffer_size_log2 = min_buffer_size_log2; buffer_size_log2 < max_buffer_size_log2; buffer_size_log2++){
      report(INFO, "starting on log2 buffer: %u", buffer_size_log2);
      work_dim = 1 << buffer_size_log2;
      for(unsigned i = 0; i < iterations; i++){
        tick(&clock);
        upload_to_GPU_pin(buffer);
        upload_t[i] = elapsed_since(&clock);
        tick(&clock);
        run_kernel();
        kernel_t[i] = elapsed_since(&clock);
        tick(&clock);
        readback_pin(buffer);
        readback_t[i] = elapsed_since(&clock);
        total_t[i] = upload_t[i] + kernel_t[i] + readback_t[i];
      }
      std::sort(upload_t, upload_t+iterations);
      std::sort(kernel_t, kernel_t+iterations);
      std::sort(readback_t, readback_t+iterations);
      std::sort(total_t, total_t+iterations);
      printf("%u %.4e %.4e %.4e %.4e\n", work_dim, upload_t[0], kernel_t[0], readback_t[0], total_t[0]);
    }printf("\n");
    report(INFO, "Effective bandwith: %3.2f GB/s", (max_buffer_size*4*2)/(total_t[0]));
    report(INFO, "Real bandwith: %3.2f GB/s up,  %3.2f GB/s down", (max_buffer_size*4)/(upload_t[0]), (max_buffer_size*4)/(readback_t[0]));

    return 0;
  }


  int main(int args, char** argv){
    encode();
    return 0;
  }
