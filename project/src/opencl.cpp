#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <string>
#include <tiff.h>
#include <tiffio.h>
#include "custom_types.h"
#include "xml_aux.h"
#include "dct8x8_block.h"
#include <vector>
#include "config.h"
#include <clu.h>
#include <utilities/file.h>
#include <utilities/logging.h>
#include <utilities/benchmarking.h>
#include <algorithm>

#include "../resources/kernels/PipeLineConstants.h"

using namespace std;


 double setup_t = 0; //time for setting up openCL
 double loadImage_t[N_FRAMES];//time spend getting image from disc
 double transfer_t[N_FRAMES];//Time spend transfering data to and from the GPU
 double convert_t[N_FRAMES];//Time spend on color conversion
 double lowpass_t[N_FRAMES];
 double mvs_t[N_FRAMES];//motionVectorSearch
 double computeDelta_t[N_FRAMES];
 double downsample_t[N_FRAMES];
 double convertFreq_t[N_FRAMES];
 double quant_t[N_FRAMES];
 double extract_t[N_FRAMES];
 double zigzag_t[N_FRAMES];
 double encode_t[N_FRAMES];
 double total_i_t[N_FRAMES]; //total time for iframes
 double total_p_t[N_FRAMES]; //total time for pframes

cl_command_queue *com_qs = NULL;
cl_kernel kernel;
cl_kernel pipeline_kernel;
cl_kernel kernel_lowPass_X;
cl_kernel kernel_lowPass_Y;
cl_kernel kernel_mvs;
size_t work_dim[2];
size_t work_item_dim[2];

cl_mem mem_R;
cl_mem mem_G;
cl_mem mem_B;

int frame_number;
cl_mem mem_Y[2];
cl_mem mem_Cr[2];
cl_mem mem_Cb[2];

// cl_mem mem_Y_ref;
// cl_mem mem_Cr_ref;
// cl_mem mem_Cb_ref;

cl_mem mem_lowPass;

cl_mem mem_indices;

//can be optimized by saving bytes instead of floats and converting dynamically.
//also should load next frame async in the background.
void loadImage(int number, string path, Image** photo) {
  string filename;
  TIFFRGBAImage img;
  char emsg[1024];

  filename = path + to_string(number) + ".tiff";
  TIFF* tif = TIFFOpen(filename.c_str(), "r");
  if(tif==NULL){
    report(FAIL,"Failed opening image: %s", filename.c_str());
    exit(-1);
  };
  if (!(TIFFRGBAImageBegin(&img, tif, 0, emsg))){
    report(FAIL, "Can't decode %s: %s", filename.c_str(), emsg);
    exit(-1);
  }

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


void uploadData_cl(Image* in){
  cl_int ret = clEnqueueWriteBuffer(com_qs[0], mem_R, CL_FALSE, 0, in->rc->size_in_bytes(), in->rc->data, 0, NULL, NULL);
  if(CL_SUCCESS != ret){
    report(FAIL, "clEnqueueWriteBuffer returned: %s (%d)", cluErrorString(ret), ret);
  }
  ret = clEnqueueWriteBuffer(com_qs[0], mem_G, CL_FALSE, 0, in->gc->size_in_bytes(), in->gc->data, 0, NULL, NULL);
  if(CL_SUCCESS != ret){
    report(FAIL, "clEnqueueWriteBuffer returned: %s (%d)", cluErrorString(ret), ret);
  }
  ret = clEnqueueWriteBuffer(com_qs[0], mem_B, CL_FALSE, 0, in->bc->size_in_bytes(), in->bc->data, 0, NULL, NULL);
  if(CL_SUCCESS != ret){
    report(FAIL, "clEnqueueWriteBuffer returned: %s (%d)", cluErrorString(ret), ret);
  }
  clFinish(com_qs[0]);
}

void readbackData_cl(Frame *out){
  //enque reading the output
  cl_int ret = clEnqueueReadBuffer(com_qs[0], mem_Y[frame_number&1], CL_FALSE, 0, out->Y->size_in_bytes(), out->Y->data, 0, NULL, NULL);
  if(CL_SUCCESS != ret){
    report(FAIL, "clEnqueueReadBuffer returned: %s (%d)", cluErrorString(ret), ret);
  }
  ret = clEnqueueReadBuffer(com_qs[0], mem_Cb[frame_number&1], CL_FALSE, 0, out->Cb->size_in_bytes(), out->Cb->data, 0, NULL, NULL);
  if(CL_SUCCESS != ret){
    report(FAIL, "clEnqueueReadBuffer returned: %s (%d)", cluErrorString(ret), ret);
  }
  ret = clEnqueueReadBuffer(com_qs[0], mem_Cr[frame_number&1], CL_FALSE, 0, out->Cr->size_in_bytes(), out->Cr->data, 0, NULL, NULL);
  if(CL_SUCCESS != ret){
    report(FAIL, "clEnqueueReadBuffer returned: %s (%d)", cluErrorString(ret), ret);
  }
  clFinish(com_qs[0]);
}

std::vector<mVector>* motionVectorSearch_cl(Frame* in, int32_t *indices){
  std::vector<mVector> *motion_vectors = new std::vector<mVector>(); // empty list of ints
  int index_cur = frame_number&1;
  int index_prev = index_cur^1;
  cl_int ret = clSetKernelArg(kernel_mvs, 0, sizeof(cl_mem), (void *)&mem_Y[index_cur]);
  if(CL_SUCCESS != ret){
    report(FAIL, "clSetKernelArg returned: %s (%d)", cluErrorString(ret), ret);
  }
  ret = clSetKernelArg(kernel_mvs, 1, sizeof(cl_mem), (void *)&mem_Cb[index_cur]);
  if(CL_SUCCESS != ret){
    report(FAIL, "clSetKernelArg returned: %s (%d)", cluErrorString(ret), ret);
  }
  ret = clSetKernelArg(kernel_mvs, 2, sizeof(cl_mem), (void *)&mem_Cr[index_cur]);
  if(CL_SUCCESS != ret){
    report(FAIL, "clSetKernelArg returned: %s (%d)", cluErrorString(ret), ret);
  }

  ret = clSetKernelArg(kernel_mvs, 3, sizeof(cl_mem), (void *)&mem_Y[index_prev]);
  if(CL_SUCCESS != ret){
    report(FAIL, "clSetKernelArg returned: %s (%d)", cluErrorString(ret), ret);
  }
  ret = clSetKernelArg(kernel_mvs, 4, sizeof(cl_mem), (void *)&mem_Cb[index_prev]);
  if(CL_SUCCESS != ret){
    report(FAIL, "clSetKernelArg returned: %s (%d)", cluErrorString(ret), ret);
  }
  ret = clSetKernelArg(kernel_mvs, 5, sizeof(cl_mem), (void *)&mem_Cr[index_prev]);
  if(CL_SUCCESS != ret){
    report(FAIL, "clSetKernelArg returned: %s (%d)", cluErrorString(ret), ret);
  }
  //run the kernel


  work_item_dim[0] = work_item_dim[1] = 16;
  if(CL_SUCCESS != (ret = clEnqueueNDRangeKernel(com_qs[0], kernel_mvs, 2, NULL, work_dim, work_item_dim, 0, NULL, NULL))){
        report(FAIL, "enqueue kernel (mvs) returned: %s (%d)",cluErrorString(ret), ret);
        return nullptr;
  }
  clFinish(com_qs[0]);

  int num_groups = work_dim[0]*work_dim[1]/(work_item_dim[0]*work_item_dim[1]);
  //read back the data
  ret = clEnqueueReadBuffer(com_qs[0], mem_indices, CL_FALSE, 0, num_groups*sizeof(float), indices, 0, NULL, NULL);
  if(CL_SUCCESS != ret){
    report(FAIL, "clEnqueueReadBuffer returned: %s (%d)", cluErrorString(ret), ret);
  }
  clFinish(com_qs[0]);

  //report(INFO,"there are %d groups", num_groups);
  for(int x = 1; x < work_dim[1]/work_item_dim[1]-1; x++){
    for(int y = 1; y < work_dim[0]/work_item_dim[0]-1; y++){
      int i = x + y*(work_dim[0]/work_item_dim[0]);
      int index = indices[i];
      int offset_y = (index%32)-16;
      int offset_x = (index/32)-16;
      mVector v;
      v.a=offset_x;
      v.b=offset_y;
      motion_vectors->push_back(v);
    }
  }
  return motion_vectors;
}

Frame* computeDelta(Frame* i_frame_ycbcr, Frame* p_frame_ycbcr, std::vector<mVector>* motion_vectors){
  Frame *delta = new Frame(p_frame_ycbcr);

  int width = i_frame_ycbcr->width;
  int height = i_frame_ycbcr->height;
  int window_size = 16;
  int block_size = 16;
  // How far from the edge we can go since we don't special case the edges
  int inset = (int) max((float) window_size, (float)block_size);

  int current_block = 0;
  for(int my=inset; my<width-(inset+window_size)+1; my+=block_size) {
    for(int mx=inset; mx<height-(inset+window_size)+1; mx+=block_size) {
      int vector[2];
      vector[0]=(int)motion_vectors->at(current_block).a;
      vector[1]=(int)motion_vectors->at(current_block).b;

      // copy the block
      for(int y=0; y<block_size; y++) {
        for(int x=0; x<block_size; x++) {

          int src_x = mx+vector[0]+x;
          int src_y = my+vector[1]+y;
          int dst_x = mx+x;
          int dst_y = my+y;
          //report(INFO, "computing delta for %d, %d -> %d, %d",src_x, src_y, dst_x, dst_y);
          delta->Y->get_ref(dst_y, dst_x) -= i_frame_ycbcr->Y->get(src_y, src_x);
          delta->Cb->get_ref(dst_y, dst_x) -= i_frame_ycbcr->Cb->get(src_y, src_x);
          delta->Cr->get_ref(dst_y, dst_x) -= i_frame_ycbcr->Cr->get(src_y, src_x);
        }
      }

      current_block = current_block + 1;
    }
  }
  return delta;
}

//mem but not simd
Frame* downSample_frame(Frame* in, Frame* out){
  int w2 = out->Cb->width;
  int h2 = out->Cb->height;
  for(int y2=0; y2<h2; y2++) {
    for (int x2 = 0; x2<w2; x2++) {
      out->Cb->set(x2, y2, in->Cb->get(x2*2,y2*2));
      out->Cr->set(x2, y2, in->Cr->get(x2*2,y2*2));
    }
  }
  return out;
}

void dct8x8(Channel* in, Channel* out){
  int width = in->width;
  int height = in -> height;

  // 8x8 block dct on each block
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      in->get_ref(j, i) -= 128;
      out->get_ref(j, i) = 0;
    }
  }

  for(int y=0; y<width; y+=8) {
    for(int x=0; x<height; x+=8) {
      dct8x8_block(&(in->data[x*width + y]), &(out->data[x*width + y]), width);
      //dct8x8_block(&(in->get_ref(y, x)),&(out->get_ref(y, x)), width);
    }
  }
}


void round_block(float* in, float* out, int stride){
  float quantMatrix[8][8] ={
    {16, 11, 10, 16,  24,  40,  51,  61},
    {12, 12, 14, 19,  26,  58,  60,  55},
    {14, 13, 16, 24,  40,  57,  69,  56},
    {14, 17, 22, 29,  51,  87,  80,  62},
    {18, 22, 37, 56,  68, 109, 103,  77},
    {24, 35, 55, 64,  81, 104, 113,  92},
    {49, 64, 78, 87, 103, 121, 120, 101},
    {72, 92, 95, 98, 112, 100, 103, 99},
  };

  for(int y=0; y<8; y++) {
    for(int x=0; x<8; x++) {
      quantMatrix[x][y] = ceil(quantMatrix[x][y]/QUALITY);
      out[x*stride+y] = (float)round(in[x*stride+y]/quantMatrix[x][y]);
    }
  }
}


void quant8x8(Channel* in, Channel* out) {
  int width = in->width;
  int height = in->height;

  for (int i = 0; i<width*height; i++) {
    out->set(i,0); //zeros
  }

  for (int y=0; y<width; y+=8) {
    for (int x=0; x<height; x+=8) {
      round_block(&(in->data[x*width + y]), &(out->data[x*width + y]), width);
    }
  }
}


void dcDiff(Channel* in, Channel* out) {
  int width = in->width;
  int height = in->height;

  int number_of_dc = width*height/64;
  double* dc_values = new double[number_of_dc];

  int iter = 0;
  for(int j=0; j<width; j+=8){
    for(int i=0; i<height; i+=8) {
      dc_values[iter] = in->get(j,i);
      iter++;
    }
  }

  int new_w = (int) max((float)(width/8), 1);
  int new_h = (int) max((float)(height/8), 1);

  out->get_ref(0) = (float)dc_values[0];

  double prev = 0.;
  iter = 0;
  for (int j=0; j<new_w; j++) {
    for (int i=0; i<new_h; i++) {
      out->get_ref(iter) = (float)(dc_values[i*new_w+j] - prev);
      prev = dc_values[i*new_w+j];
      iter++;
    }
  }
  delete dc_values;

}

void cpyBlock(float* in, float* out, int blocksize, int stride) {
  for (int i = 0; i<blocksize; i++){
    for (int j=0; j<blocksize; j++) {
      out[i*blocksize+j] = in[i*stride+j];
    }
  }
}


void zigZagOrder(Channel* in, Channel* ordered) {
  int width = in->width;
  int height = in->height;
  int zigZagIndex[64]={0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,12,19,26,33,40,
    48,41,34,27,20,13,6,7,14,21,28,35,42,49,56,57,50,43,36,29,22,15,23,30,37,
    44,51,58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63};

    int blockNumber=0;
    float _block[MPEG_CONSTANT];

    for(int x=0; x<height; x+=8) {
      for(int y=0; y<width; y+=8) {
        cpyBlock(&(in->data[x*width+y]), _block, 8, width); //block = in(x:x+7,y:y+7);
        //Put the coefficients in zig-zag order
        for (int i=0; i<MPEG_CONSTANT; i++)
        ordered->set(blockNumber*MPEG_CONSTANT + i, _block[zigZagIndex[i]]);//converst a block, zigzagged, into a line of 64
        blockNumber++;
      }
    }
  }


  void encode8x8(Channel* ordered, SMatrix* encoded){
    int width = encoded->height;//dafuq?
    int height = encoded->width;
    int num_blocks = height;

    for(int i=0; i<num_blocks; i++) {
      std::string block_encode[MPEG_CONSTANT];
      for (int j=0; j<MPEG_CONSTANT; j++) {
        block_encode[j]="\0"; //necessary to initialize every string position to empty string
      }

      double* block = new double[width];
      for(int y=0; y<width; y++)
      block[y] = ordered->get(y,i);
      int num_coeff = MPEG_CONSTANT; //width
      int encoded_index = 0;
      int in_zero_run = 0;
      int zero_count = 0;

      // Skip DC coefficient
      for(int c=1; c<num_coeff; c++){
        double coeff = block[c];
        if (coeff == 0){
          if (in_zero_run == 0){
            zero_count = 0;
            in_zero_run = 1;
          }
          zero_count = zero_count + 1;
        }
        else {
          if (in_zero_run == 1){
            in_zero_run = 0;
            block_encode[encoded_index] = "Z" + std::to_string(zero_count);
            encoded_index = encoded_index+1;
          }
          block_encode[encoded_index] = std::to_string((int)coeff);
          encoded_index = encoded_index+1;
        }
      }

      // If we were in a zero run at the end attach it as well.
      if (in_zero_run == 1) {
        if (zero_count > 1) {
          block_encode[encoded_index] = "Z" + std::to_string(zero_count);
        } else {
          block_encode[encoded_index] = "0";
        }
      }


      for(int it=0; it < MPEG_CONSTANT; it++) {
        if (block_encode[it].length() > 0)
        encoded->data[i*width+it] = new std::string(block_encode[it]);
        else
        it = MPEG_CONSTANT;
      }
      delete block;
    }
  }



bool ErrorCheck(cl_int err_code)
{
  if(CL_SUCCESS != err_code)
    {
      report(FAIL, "clCreateKernel returned: %s (%d)", cluErrorString(err_code), err_code);
      return false;
    }
  return true;
}

void setupPipelineKernel(cl_program program,  int image_rows, int image_cols)
{

  //
  // get pipeline kernel reference
  //
  cl_int ret = CL_SUCCESS;
  pipeline_kernel = clCreateKernel(program, "RGB2YCbCr_LowPassFilter_pipeline", &ret);
  ErrorCheck(ret);
  //
  // Setup the arguments:
  //
  //
  // Inputs:
  //
  ErrorCheck(clSetKernelArg(pipeline_kernel, 0, sizeof(cl_mem), (void *)&mem_R) );
  ErrorCheck(clSetKernelArg(pipeline_kernel, 1, sizeof(cl_mem), (void *)&mem_G) );
  ErrorCheck(clSetKernelArg(pipeline_kernel, 2, sizeof(cl_mem), (void *)&mem_B) );
  //
  // Outputs
  //
  // ErrorCheck(clSetKernelArg(pipeline_kernel, 3, sizeof(cl_mem), (void *)&mem_Y) );
  // ErrorCheck(clSetKernelArg(pipeline_kernel, 4, sizeof(cl_mem), (void *)&mem_Cb) );
  // ErrorCheck(clSetKernelArg(pipeline_kernel, 5, sizeof(cl_mem), (void *)&mem_Cr) );
  //
  // Constants
  //
  ErrorCheck(clSetKernelArg(pipeline_kernel, 6, sizeof(int), (void *)&image_rows) ); //image_rows == image-in->height
  ErrorCheck(clSetKernelArg(pipeline_kernel, 7, sizeof(int), (void *)&image_cols) );  //image_clos == image-in->width

}
void RunPipelineKernel(unsigned rows, unsigned cols)
{

  work_item_dim[0] = DIM_X; // ex 64 - in fast x-dimension
  work_item_dim[1] = DIM_Y; // ex 8
  work_dim[0] = cols;   // x-direction
  work_dim[1] = rows;
  report(INFO, "trying to run kernel with [%u, %u] and [%u, %u]", DIM_X, DIM_Y, cols, rows);
  //enque the kernel
  cl_int ret;
  ErrorCheck(clSetKernelArg(pipeline_kernel, 3, sizeof(cl_mem), (void *)&mem_Y[frame_number&1]) );
  ErrorCheck(clSetKernelArg(pipeline_kernel, 4, sizeof(cl_mem), (void *)&mem_Cb[frame_number&1]) );
  ErrorCheck(clSetKernelArg(pipeline_kernel, 5, sizeof(cl_mem), (void *)&mem_Cr[frame_number&1]) );
  if(CL_SUCCESS != (ret = clEnqueueNDRangeKernel(com_qs[0], pipeline_kernel, 2, NULL, work_dim, work_item_dim, 0, NULL, NULL))){
    report(FAIL, "clEnqueueNDRangeKernel(pipeline) returned: %s (%d)", cluErrorString(ret), ret);
    exit(-1);
  }
  clFinish(com_qs[0]);
}
  std::vector<mVector>* motionVectorSearch(Frame* source, Frame* match, int width, int height) {
    std::vector<mVector> *motion_vectors = new std::vector<mVector>(); // empty list of ints

    float Y_weight = 0.5;
    float Cr_weight = 0.25;
    float Cb_weight = 0.25;

    //Window size is how much on each side of the block we search
    int window_size = 16;
    int block_size = 16;

    //How far from the edge we can go since we don't special case the edges
    int inset = (int) max((float)window_size, (float)block_size);
    int iter=0;
    //Loop over all the blocks in the image.
    for (int my=inset; my<height-(inset+window_size)+1; my+=block_size) {
      for (int mx=inset; mx<width-(inset+window_size)+1; mx+=block_size) {

        float best_match_sad = 1e10;
        int best_match_location[2] = {0, 0};
        //Tile block_size by block_size blocks, in a window_size by window_size area around the current block
        for(int sy=my-window_size; sy<my+window_size; sy++) {
          for(int sx=mx-window_size; sx<mx+window_size; sx++) {
            float current_match_sad = 0;
            // Do the SAD
            //compute current delta.
            for (int y=0; y<block_size; y++) {
              for (int x=0; x<block_size; x++) {
                int match_x = mx+x;
                int match_y = my+y;
                int search_x = sx+x;
                int search_y = sy+y;
                float diff_Y  = fabs(match->Y->get(match_y, match_x) - source->Y->get(search_y, search_x));
                float diff_Cb = fabs(match->Cb->get(match_y, match_x) - source->Cb->get(search_y, search_x));
                float diff_Cr = fabs(match->Cr->get(match_y, match_x) - source->Cr->get(search_y, search_x));

                float diff_total = Y_weight*diff_Y + Cb_weight*diff_Cb + Cr_weight*diff_Cr;
                current_match_sad = current_match_sad + diff_total;
              }
            } //end SAD
            //if this tile has the best match, remember it.
            if (current_match_sad <= best_match_sad){
              best_match_sad = current_match_sad;
              best_match_location[0] = sx-mx;
              best_match_location[1] = sy-my;
            }
          }
        }

        //store the best.
        mVector v;
        v.a=best_match_location[0];
        v.b=best_match_location[1];
        motion_vectors->push_back(v);

      }
    }

    return motion_vectors;
  }

  int encode(const char* kernel_name) {
    REPORT_W_COLORS = 1;
    REPORT_W_TIMESTAMPS = 1;
    set_cwdir_to_bin_dir();
    string image_path =  "../../inputs/" + string(image_name) + "/" + image_name + ".";
    string stream_path = "../outputs/stream_c_" + string(image_name) + ".xml";

    xmlDocPtr stream = NULL;
    Image* frame_rgb = NULL;
    Image* previous_frame_rgb = NULL;
    Frame* previous_frame_lowpassed = NULL;

    loadImage(0, image_path, &frame_rgb);

    int width = frame_rgb->width;
    int height = frame_rgb->height;
    int npixels = width*height;
    int npixels_lowPass = npixels;
    //int npixels_lowPass = (width-2)*(height-2);

    delete frame_rgb;
    int32_t *indices = (int32_t*) malloc(4*npixels/(16*16));
    int end_frame = int(N_FRAMES);
    int i_frame_frequency = int(I_FRAME_FREQ);
    //struct timeval starttime, endtime;
    struct timespec clock, total_clock;
    //double runtime[10] = {0};
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

    mem_Y[0] = clCreateBuffer(context, CL_MEM_READ_WRITE, npixels*sizeof(float), nullptr, &ret);
    if(CL_SUCCESS != ret){
      report(FAIL, "clCreateBuffer (Y) returned: %s (%d)", cluErrorString(ret), ret);
    }
    mem_Cb[0] = clCreateBuffer(context, CL_MEM_READ_WRITE, npixels*sizeof(float), nullptr, &ret);
    if(CL_SUCCESS != ret){
      report(FAIL, "clCreateBuffer (Cb) returned: %s (%d)", cluErrorString(ret), ret);
    }
    mem_Cr[0] = clCreateBuffer(context, CL_MEM_READ_WRITE, npixels*sizeof(float), nullptr, &ret);
    if(CL_SUCCESS != ret){
      report(FAIL, "clCreateBuffer (Cr) returned: %s (%d)", cluErrorString(ret), ret);
    }

    mem_Y[1] = clCreateBuffer(context, CL_MEM_READ_ONLY, npixels*sizeof(float), nullptr, &ret);
    if(CL_SUCCESS != ret){
      report(FAIL, "clCreateBuffer (Y_ref) returned: %s (%d)", cluErrorString(ret), ret);
    }
    mem_Cb[1] = clCreateBuffer(context, CL_MEM_READ_ONLY, npixels*sizeof(float), nullptr, &ret);
    if(CL_SUCCESS != ret){
      report(FAIL, "clCreateBuffer (Cb_ref) returned: %s (%d)", cluErrorString(ret), ret);
    }
    mem_Cr[1] = clCreateBuffer(context, CL_MEM_READ_ONLY, npixels*sizeof(float), nullptr, &ret);
    if(CL_SUCCESS != ret){
      report(FAIL, "clCreateBuffer (Cr_ref) returned: %s (%d)", cluErrorString(ret), ret);
    }


    //buffer for lowPass
    mem_lowPass = clCreateBuffer(context, CL_MEM_READ_WRITE, npixels_lowPass*sizeof(float), nullptr, &ret);
    if(CL_SUCCESS != ret){
      report(FAIL, "clCreateBuffer (lowPass) returned: %s (%d)", cluErrorString(ret), ret);
    }

    mem_indices = clCreateBuffer(context, CL_MEM_WRITE_ONLY, npixels*sizeof(float), nullptr, &ret);
    if(CL_SUCCESS != ret){
      report(FAIL, "clCreateBuffer (indices) returned: %s (%d)", cluErrorString(ret), ret);
    }

    char *program_src = NULL;
    cl_program program = cluProgramFromFilename(context, kernel_name);
    ret = clBuildProgram(program, device_count, devices, NULL, NULL, NULL);
    if(CL_SUCCESS != ret){
      report(FAIL, "clBuildProgram returned: %s (%d)", cluErrorString(ret), ret);
    }
    char *log = NULL;
    for(int i = 0; i < device_count; i++){
      cluGetProgramLog(program, devices[0], CLU_DYNAMIC, &log);
      report(INFO, "log device[%d]\n==============\n%s",i, log);
    }
    free(log);
    report(PASS, "program build");

    setupPipelineKernel(program, height, width); //convertRGB + lowPass

    kernel_mvs = clCreateKernel(program, "motionVectorSearch", &ret);
    if(CL_SUCCESS != ret){
      report(FAIL, "clCreateKernel returned: %s (%d)", cluErrorString(ret), ret);
    }
    report(PASS, "kernel created");

    ret = clSetKernelArg(kernel_mvs, 6, sizeof(cl_mem), (void *)&mem_indices);
    if(CL_SUCCESS != ret){
      report(FAIL, "clSetKernelArg returned: %s (%d)", cluErrorString(ret), ret);
    }

    report(PASS, "kernel setup");


    /*/////////////////
    // END OPENCL INIT
    ///////////////*/
    setup_t = elapsed_since(&clock);

    stream = create_xml_stream(width, height, QUALITY, WINDOW_SIZE, BLOCK_SIZE);
    vector<mVector>* motion_vectors = NULL;

    for (frame_number = 0 ; frame_number < end_frame ; frame_number++) {
      frame_rgb = NULL;
      tick(&total_clock);
      tick(&clock);
      loadImage(frame_number, image_path, &frame_rgb);
      loadImage_t[frame_number] = elapsed_since(&clock);

      tick(&clock);
      uploadData_cl(frame_rgb);
      transfer_t[frame_number] = elapsed_since(&clock);

      //convertRGB + lowPass
      report(INFO, "Covert to YCbCr + Low pass filter...");
      tick(&clock);
      RunPipelineKernel(height, width);
      convert_t[frame_number] = 0;
      lowpass_t[frame_number] = elapsed_since(&clock); //each get half time?

      Frame *frame_lowpassed = new Frame(width, height, FULLSIZE);
      tick(&clock);
      readbackData_cl(frame_lowpassed);
      transfer_t[frame_number] += elapsed_since(&clock);

      dump_frame(frame_lowpassed, "frame_ycbcr_lowpass", frame_number);
      delete frame_rgb;

      Frame *frame_lowpassed_final = NULL;
      //uses Cb/Cr at full resolution
      if (frame_number % i_frame_frequency != 0) {
        // We have a P frame
        // Note that in the first iteration we don't enter this branch!

        //Compute the motion vectors
        report(INFO, "Motion Vector Search...");

        tick(&clock);
        motion_vectors = motionVectorSearch_cl(previous_frame_lowpassed, indices);
        mvs_t[frame_number] = elapsed_since(&clock);
        // vector<mVector>* motion_vectors_old = motionVectorSearch(previous_frame_lowpassed, frame_lowpassed, frame_lowpassed->width, frame_lowpassed->height);
        // for(int i = 0; i < motion_vectors_old->size();i++){
        //   report(WARN, "(%d, %d) (%d, %d)", motion_vectors_old->at(i).a, motion_vectors_old->at(i).b,  motion_vectors->at(i).a,  motion_vectors->at(i).b);
        // }
        report(INFO, "Compute Delta...");
        tick(&clock);
        frame_lowpassed_final = computeDelta(previous_frame_lowpassed, frame_lowpassed, motion_vectors);
        computeDelta_t[frame_number] = elapsed_since(&clock);

      } else {
        // We have a I frame
        motion_vectors = NULL;
        frame_lowpassed_final = new Frame(frame_lowpassed);
      }
      delete frame_lowpassed; frame_lowpassed=NULL;

      if (frame_number > 0) delete previous_frame_lowpassed;
      previous_frame_lowpassed = new Frame(frame_lowpassed_final);


      // Downsample the difference
      report(INFO, "Downsample...");

      Frame* frame_downsampled = new Frame(width, height, DOWNSAMPLE);
      // We don't touch the Y frame
      tick(&clock);
      frame_downsampled->Y->copy(frame_lowpassed_final->Y);
      downSample_frame(frame_lowpassed_final, frame_downsampled);
      downsample_t[frame_number] = elapsed_since(&clock);

      dump_frame(frame_downsampled, "frame_downsampled", frame_number);
      delete frame_lowpassed_final;
      //delete frame_downsampled_cb;
      //delete frame_downsampled_cr;

      // Convert to frequency domain
      report(INFO, "Convert to frequency domain...");

      Frame* frame_dct = new Frame(width, height, DOWNSAMPLE);

      tick(&clock);
      dct8x8(frame_downsampled->Y, frame_dct->Y);
      dct8x8(frame_downsampled->Cb, frame_dct->Cb);
      dct8x8(frame_downsampled->Cr, frame_dct->Cr);
      convertFreq_t[frame_number] = elapsed_since(&clock);

      dump_frame(frame_dct, "frame_dct", frame_number);
      delete frame_downsampled;

      //Quantize the data
      report(INFO, "Quantize...");

      Frame* frame_quant = new Frame(width, height, DOWNSAMPLE);

      tick(&clock);
      quant8x8(frame_dct->Y, frame_quant->Y);
      quant8x8(frame_dct->Cb, frame_quant->Cb);
      quant8x8(frame_dct->Cr, frame_quant->Cr);
      quant_t[frame_number] = elapsed_since(&clock);

      dump_frame(frame_quant, "frame_quant", frame_number);
      delete frame_dct;

      //Extract the DC components and compute the differences
      report(INFO, "Compute DC differences...");


      Frame* frame_dc_diff = new Frame(1, (width/8)*(height/8), DCDIFF); //dealocate later

      tick(&clock);
      dcDiff(frame_quant->Y, frame_dc_diff->Y);
      dcDiff(frame_quant->Cb, frame_dc_diff->Cb);
      dcDiff(frame_quant->Cr, frame_dc_diff->Cr);
      extract_t[frame_number] = elapsed_since(&clock);

      dump_dc_diff(frame_dc_diff, "frame_dc_diff", frame_number);

      // Zig-zag order for zero-counting
      report(INFO, "Zig-zag order...");
      Frame* frame_zigzag = new Frame(MPEG_CONSTANT, width*height/MPEG_CONSTANT, ZIGZAG);

      tick(&clock);
      zigZagOrder(frame_quant->Y, frame_zigzag->Y);
      zigZagOrder(frame_quant->Cb, frame_zigzag->Cb);
      zigZagOrder(frame_quant->Cr, frame_zigzag->Cr);
      zigzag_t[frame_number] = elapsed_since(&clock);

      dump_zigzag(frame_zigzag, "frame_zigzag", frame_number);
      delete frame_quant;

      // Encode coefficients
      report(INFO, "Encode coefficients...");

      FrameEncode* frame_encode = new FrameEncode(width, height, MPEG_CONSTANT);

      tick(&clock);
      encode8x8(frame_zigzag->Y, frame_encode->Y);
      encode8x8(frame_zigzag->Cb, frame_encode->Cb);
      encode8x8(frame_zigzag->Cr, frame_encode->Cr);
      encode_t[frame_number] = elapsed_since(&clock);

      delete frame_zigzag;
      if (frame_number % i_frame_frequency != 0){
        total_p_t[frame_number] = elapsed_since(&total_clock);
        total_p_t[frame_number] += -loadImage_t[frame_number]-transfer_t[frame_number]-convert_t[frame_number]-lowpass_t[frame_number]-mvs_t[frame_number]-computeDelta_t[frame_number]-downsample_t[frame_number]-convertFreq_t[frame_number]-quant_t[frame_number]-extract_t[frame_number]-zigzag_t[frame_number]-encode_t[frame_number];
      }else{
        total_i_t[frame_number] = elapsed_since(&total_clock);
        total_i_t[frame_number] += -loadImage_t[frame_number]-transfer_t[frame_number]-convert_t[frame_number]-lowpass_t[frame_number]-downsample_t[frame_number]-convertFreq_t[frame_number]-quant_t[frame_number]-extract_t[frame_number]-zigzag_t[frame_number]-encode_t[frame_number];
      }

      stream_frame(stream, frame_number, motion_vectors, frame_number-1, frame_dc_diff, frame_encode);
      write_stream(stream_path, stream);

      delete frame_dc_diff;
      delete frame_encode;

      if (motion_vectors != NULL) {
        free(motion_vectors);
        motion_vectors = NULL;
      }
    }

    std::sort(loadImage_t, loadImage_t+N_FRAMES);
    std::sort(transfer_t, transfer_t+N_FRAMES);
    std::sort(convert_t, convert_t+N_FRAMES);
    std::sort(lowpass_t, lowpass_t+N_FRAMES);
    std::sort(mvs_t, mvs_t+N_FRAMES);
    std::sort(computeDelta_t, computeDelta_t+N_FRAMES);
    std::sort(downsample_t, downsample_t+N_FRAMES);
    std::sort(convertFreq_t, convertFreq_t+N_FRAMES);
    std::sort(quant_t, quant_t+N_FRAMES);
    std::sort(extract_t, extract_t+N_FRAMES);
    std::sort(zigzag_t, zigzag_t+N_FRAMES);
    std::sort(encode_t, encode_t+N_FRAMES);
    std::sort(total_i_t, total_i_t+N_FRAMES);
    std::sort(total_p_t, total_p_t+N_FRAMES);
    printf("#pixels: %u, frames: %d, i_frame_frequency: %d\n", width*height, N_FRAMES, i_frame_frequency);
    printf("setup %.4e\n", setup_t);
    printf("loadImage %.4e\n", loadImage_t[0]);
    printf("transfer %.4e\n", transfer_t[0]);
    printf("convert %.4e\n", convert_t[0]);
    printf("lowpass %.4e\n", lowpass_t[0]);
    printf("mvs %.4e\n", mvs_t[0]);
    printf("computeDelta %.4e\n", computeDelta_t[0]);
    printf("downsample %.4e\n", downsample_t[0]);
    printf("convertFreq %.4e\n", convertFreq_t[0]);
    printf("quant %.4e\n", quant_t[0]);
    printf("extract %.4e\n", extract_t[0]);
    printf("zigzag %.4e\n", zigzag_t[0]);
    printf("encode %.4e\n", encode_t[0]);
    printf("overhead_i %.4e\n", total_i_t[0]);
    printf("overhead_p %.4e\n", total_p_t[0]);
    // printf("%u %.4e %.4e %.4e %.4e %.4e %.4e  %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e\n",
    //  width*height,
    //   setup_t,
    //   loadImage_t[0],
    //   transfer_t[0],
    //   convert_t[0],
    //   lowpass_t[0],
    //   mvs_t[0],
    //   computeDelta_t[0],
    //   downsample_t[0],
    //   convertFreq_t[0],
    //   quant_t[0],
    //   extract_t[0],
    //   zigzag_t[0],
    //   encode_t[0],
    //   total_i_t[0],
    //   total_p_t[0]
    // );
    return 0;
  }


  int main(int args, char** argv){
    for(int i = 0; i < N_FRAMES; i++){
      loadImage_t[i] = 1e200;
      transfer_t[i] = 1e200;
      convert_t[i] = 1e200;
      lowpass_t[i] = 1e200;
      mvs_t[i] = 1e200;
      computeDelta_t[i] = 1e200;
      downsample_t[i] = 1e200;
      convertFreq_t[i] = 1e200;
      quant_t[i] = 1e200;
      extract_t[i] = 1e200;
      zigzag_t[i] = 1e200;
      encode_t[i] = 1e200;
      total_i_t[i] = 1e200;
      total_p_t[i] = 1e200;
    }
    encode(argv[1]);
    return 0;
  }
