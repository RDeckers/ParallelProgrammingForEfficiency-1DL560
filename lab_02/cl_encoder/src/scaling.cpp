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
size_t work_item_dim;

cl_mem mem_R;
cl_mem mem_G;
cl_mem mem_B;

cl_mem mem_Y;
cl_mem mem_Cr;
cl_mem mem_Cb;

//#define SIMD_OPT
#define MEMORY_OPT
//#define PARALLEL_OPT

//can be optimized by saving bytes instead of floats and converting dynamically.
//also should load next frame async in the background.
void loadImage(int number, string path, Image** photo) {
  string filename;
  TIFFRGBAImage img;
  char emsg[1024];

  filename = path + to_string(number) + ".tiff";
  TIFF* tif = TIFFOpen(filename.c_str(), "r");
  if(tif==NULL) fprintf(stderr,"Failed opening image: %s\n", filename);
  if (!(TIFFRGBAImageBegin(&img, tif, 0, emsg))) TIFFError(filename.c_str(), emsg);

  uint32 w, h;
  size_t npixels;
  uint32* raster;

  TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
  TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
  npixels = w * h;
  raster = (uint32*) _TIFFmalloc(npixels * sizeof (uint32));

  TIFFReadRGBAImage(tif, w, h, raster, 0);

  if(*photo==NULL)
  *photo = new Image((int)w, (int)h, FULLSIZE);

  //Matlab and LibTIFF store the image diferently.
  //Necessary to mirror the image horizonatly to be consistent
  for (int j=0; j<(int)w; j++) {
    for (int i=0; i<(int)h; i++) {
      // The inversion is ON PURPOSE
      (*photo)->rc->set(j, h - 1 - i, (float)TIFFGetR(raster[i*w+j]));
      (*photo)->gc->set(j, h - 1 - i, (float)TIFFGetG(raster[i*w + j]));
      (*photo)->bc->set(j, h - 1 - i, (float)TIFFGetB(raster[i*w + j]));
    }
  }

  _TIFFfree(raster);
  TIFFRGBAImageEnd(&img);
  TIFFClose(tif);

}

void upload_to_GPU(Image *in){
  cl_int ret = clEnqueueWriteBuffer(com_qs[0], mem_R, CL_FALSE, 0, work_dim, in->rc->data, 0, NULL, NULL);
  if(CL_SUCCESS != ret){
    report(FAIL, "clEnqueueWriteBuffer returned: %s (%d)", cluErrorString(ret), ret);
  }
  ret = clEnqueueWriteBuffer(com_qs[0], mem_G, CL_FALSE, 0, work_dim, in->gc->data, 0, NULL, NULL);
  if(CL_SUCCESS != ret){
    report(FAIL, "clEnqueueWriteBuffer returned: %s (%d)", cluErrorString(ret), ret);
  }
  ret = clEnqueueWriteBuffer(com_qs[0], mem_B, CL_FALSE, 0, work_dim, in->bc->data, 0, NULL, NULL);
  if(CL_SUCCESS != ret){
    report(FAIL, "clEnqueueWriteBuffer returned: %s (%d)", cluErrorString(ret), ret);
  }
  clFinish(com_qs[0]);
}

void convertRGBtoYCbCr_cl(){
  //enque the kernel
  cl_int ret;
  if(CL_SUCCESS != (ret = clEnqueueNDRangeKernel(com_qs[0], kernel, 1, NULL, &work_dim, &work_item_dim, 0, NULL, NULL))){
        report(FAIL, "enqueue kernel[0] returned: %s (%d)",cluErrorString(ret), ret);
        return;
  }
  clFinish(com_qs[0]);
}

void readback(Image *out){
    //enque reading the output
    cl_int ret;
    ret = clEnqueueReadBuffer(com_qs[0], mem_Y, CL_FALSE, 0, work_dim, out->rc->data, 0, NULL, NULL);
    if(CL_SUCCESS != ret){
      report(FAIL, "clEnqueueReadBuffer returned: %s (%d)", cluErrorString(ret), ret);
    }
    ret = clEnqueueReadBuffer(com_qs[0], mem_Cb, CL_FALSE, 0, work_dim, out->gc->data, 0, NULL, NULL);
    if(CL_SUCCESS != ret){
      report(FAIL, "clEnqueueReadBuffer returned: %s (%d)", cluErrorString(ret), ret);
    }
    ret = clEnqueueReadBuffer(com_qs[0], mem_Cr, CL_FALSE, 0, work_dim, out->bc->data, 0, NULL, NULL);
    if(CL_SUCCESS != ret){
      report(FAIL, "clEnqueueReadBuffer returned: %s (%d)", cluErrorString(ret), ret);
    }
    clFinish(com_qs[0]);
    //wait for it to finish.
}

void convertRGBtoYCbCr(Image* in, Image *out) {
  int width = in->width;
  int height = in->height;
  for (int i = 0; i < width*height; i+=8) {
    v8f R = in->rc->get_8(i);
    v8f G = in->gc->get_8(i);
    v8f B = in->bc->get_8(i);
    v8f Y = 0.0f + (0.299f*R) + (0.587f*G) + (0.113f*B);
    v8f Cb = 128.0f - (0.168736f*R) - (0.331264f*G) + (0.5f*B);
    v8f Cr = 128.0f + (0.5f*R) - (0.418688f*G) - (0.081312f*B);
    _mm256_stream_ps(out->rc->data+i, Y);
    _mm256_stream_ps(out->gc->data+i, Cb);
    _mm256_stream_ps(out->bc->data+i, Cr);
  }
  _mm_mfence();
}

int encode() {
  REPORT_W_COLORS = 1;
  REPORT_W_TIMESTAMPS = 1;
  set_cwdir_to_bin_dir();

  Image *frame_rgb = nullptr;
  string image_path =  "../../../inputs/" + string(image_name) + "/" + image_name + ".";
  report(INFO, "using image_path = %s", image_path.c_str());
  loadImage(0, image_path, &frame_rgb);
  int width = frame_rgb->width;
  int height = frame_rgb->height;
  int npixels = width*height;
  report(INFO, "Image dimensions: %dx%d (%d pixels)", width, height, npixels);
  delete frame_rgb;

  int end_frame = int(N_FRAMES);
  //struct timeval starttime, endtime;
  struct timespec clock;
  double setup_t = 0;
  double getTIFF_t[N_FRAMES] = {0};
  double upload_t[N_FRAMES] ={0};
  double convert_t[N_FRAMES] = {0};
  double conversion_total_t[N_FRAMES] = {0};
  double readback_t[N_FRAMES] = {0};

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

    work_dim = npixels;
    work_item_dim = 512;
    cl_int ret;
    mem_R = clCreateBuffer(context, CL_MEM_READ_ONLY, npixels*sizeof(float), nullptr, &ret);
    if(CL_SUCCESS != ret){
      report(FAIL, "clCreateBuffer (R) returned: %s (%d)", cluErrorString(ret), ret);
    }
    mem_G = clCreateBuffer(context, CL_MEM_READ_ONLY, npixels*sizeof(float), nullptr, &ret);
    if(CL_SUCCESS != ret){
      report(FAIL, "clCreateBuffer (G) returned: %s (%d)", cluErrorString(ret), ret);
    }
    mem_B = clCreateBuffer(context, CL_MEM_READ_ONLY, npixels*sizeof(float), nullptr, &ret);
    if(CL_SUCCESS != ret){
      report(FAIL, "clCreateBuffer (B) returned: %s (%d)", cluErrorString(ret), ret);
    }

    mem_Y = clCreateBuffer(context, CL_MEM_WRITE_ONLY, npixels*sizeof(float), nullptr, &ret);
    if(CL_SUCCESS != ret){
      report(FAIL, "clCreateBuffer (Y) returned: %s (%d)", cluErrorString(ret), ret);
    }
    mem_Cb = clCreateBuffer(context, CL_MEM_WRITE_ONLY, npixels*sizeof(float), nullptr, &ret);
    if(CL_SUCCESS != ret){
      report(FAIL, "clCreateBuffer (Cb) returned: %s (%d)", cluErrorString(ret), ret);
    }
    mem_Cr = clCreateBuffer(context, CL_MEM_WRITE_ONLY, npixels*sizeof(float), nullptr, &ret);
    if(CL_SUCCESS != ret){
      report(FAIL, "clCreateBuffer (Cr) returned: %s (%d)", cluErrorString(ret), ret);
    }


    char *program_src = NULL;
    cl_program program = cluProgramFromFilename(context, "../resources/kernels/encoder.cl");
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

    kernel = clCreateKernel(program, "convertRGBtoYCbCr", &ret);
    if(CL_SUCCESS != ret){
      report(FAIL, "clCreateKernel returned: %s (%d)", cluErrorString(ret), ret);
    }

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&mem_R);
    if(CL_SUCCESS != ret){
      report(FAIL, "clSetKernelArg returned: %s (%d)", cluErrorString(ret), ret);
    }
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&mem_G);
    if(CL_SUCCESS != ret){
      report(FAIL, "clSetKernelArg returned: %s (%d)", cluErrorString(ret), ret);
    }
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&mem_B);
    if(CL_SUCCESS != ret){
      report(FAIL, "clSetKernelArg returned: %s (%d)", cluErrorString(ret), ret);
    }

    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&mem_Y);
    if(CL_SUCCESS != ret){
      report(FAIL, "clSetKernelArg returned: %s (%d)", cluErrorString(ret), ret);
    }
    ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&mem_Cb);
    if(CL_SUCCESS != ret){
      report(FAIL, "clSetKernelArg returned: %s (%d)", cluErrorString(ret), ret);
    }
    ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&mem_Cr);
    if(CL_SUCCESS != ret){
      report(FAIL, "clSetKernelArg returned: %s (%d)", cluErrorString(ret), ret);
    }
    setup_t = elapsed_since(&clock);
    /*/////////////////
    // END OPENCL INIT
    ///////////////*/

    report(PASS, "Setup OpenCL");
    printf("#image_dimensions setup load/decode_image upload convert readback upload+convert+readback\n");
    for(size_t max_dim_log2 = 5; max_dim_log2 <= 12; max_dim_log2++){
      work_dim = (1 << max_dim_log2)*(1 << max_dim_log2);
      report(INFO,"starting on work_dim = %u (%ux%u)", work_dim, (1 << max_dim_log2), (1 << max_dim_log2));
    for (int frame_number = 0 ; frame_number < end_frame ; frame_number++) {
      frame_rgb = nullptr;
      tick(&clock);
      loadImage(frame_number, image_path, &frame_rgb);
      getTIFF_t[frame_number] = elapsed_since(&clock);
      Image* frame_ycbcr = new Image(width, height, FULLSIZE);
      report(PASS, "Loaded image %d", frame_number);
      tick(&clock);
      upload_to_GPU(frame_rgb);
      upload_t[frame_number] = tock(&clock);
      convertRGBtoYCbCr_cl();
      convert_t[frame_number] = tock(&clock);
      readback(frame_ycbcr);
      readback_t[frame_number] = elapsed_since(&clock);
      conversion_total_t[frame_number] = readback_t[frame_number]+convert_t[frame_number]+upload_t[frame_number];
      delete frame_rgb;
      delete frame_ycbcr;
    }
    std::sort(getTIFF_t, getTIFF_t+N_FRAMES);
    std::sort(upload_t, upload_t+N_FRAMES);
    std::sort(convert_t, convert_t+N_FRAMES);
    std::sort(readback_t, readback_t+N_FRAMES);
    std::sort(conversion_total_t, conversion_total_t+N_FRAMES);

    printf("%u %.4e %.4e %.4e %.4e %.4e %.4e\n", 1 << max_dim_log2, setup_t, getTIFF_t[0], upload_t[0], convert_t[0], readback_t[0], conversion_total_t[0]);
  }
    return 0;
  }


  int main(int args, char** argv){
    encode();
    return 0;
  }
