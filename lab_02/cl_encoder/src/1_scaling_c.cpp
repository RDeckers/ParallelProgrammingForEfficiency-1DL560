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

size_t work_dim;
size_t work_item_dim;

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

void convertRGBtoYCbCr(Image* in) {
  int width = work_dim;
  int height = work_dim;
  //report(INFO,"working on an %dx%d image", work_item_dim, work_item_dim);
  #pragma omp parallel for
  for (int i = 0; i < width*height; i+=8) {
    v8f R = in->rc->get_8(i);
    v8f G = in->gc->get_8(i);
    v8f B = in->bc->get_8(i);
    v8f Y = 0.0f + (0.299f*R) + (0.587f*G) + (0.113f*B);
    v8f Cb = 128.0f - (0.168736f*R) - (0.331264f*G) + (0.5f*B);
    v8f Cr = 128.0f + (0.5f*R) - (0.418688f*G) - (0.081312f*B);
    in->rc->set_8(i, Y);
    in->gc->set_8(i, Cb);
    in->bc->set_8(i, Cr);
  }
  // _mm_mfence();
}

int encode() {
  REPORT_W_COLORS = 1;
  REPORT_W_TIMESTAMPS = 1;
  set_cwdir_to_bin_dir();

  Image *frames_rgb[N_FRAMES] = {nullptr};
  string image_path =  "../../../inputs/" + string(image_name) + "/" + image_name + ".";
  report(INFO, "using image_path = %s", image_path.c_str());

  int end_frame = int(N_FRAMES);
  //struct timeval starttime, endtime;
  struct timespec clock;
  double setup_t = 0;
  double getTIFF_t[N_FRAMES] = {0};
  double upload_t[N_FRAMES] ={0};
  double convert_t[N_FRAMES] = {0};
  double conversion_total_t[N_FRAMES] = {0};
  double readback_t[N_FRAMES] = {0};

  for(int frame_number = 0 ; frame_number < end_frame ; frame_number++){
    tick(&clock);
    loadImage(frame_number, image_path, frames_rgb+frame_number);
    getTIFF_t[frame_number] = elapsed_since(&clock);
    report(PASS, "Loaded image %d", frame_number);
  }
  int width = frames_rgb[0]->width;
  int height = frames_rgb[0]->height;
  int npixels = width*height;
  report(INFO, "Image dimensions: %dx%d (%d pixels)", width, height, npixels);

/*////////////////////
// CL INIT
////////////////////*/
  tick(&clock);
  setup_t = elapsed_since(&clock);
    /*/////////////////
    // END OPENCL INIT
    ///////////////*/
    printf("#image_dimensions setup load/decode_image upload convert readback upload+convert+readback\n");
    for(size_t max_dim_log2 = 4; max_dim_log2 <= 12; max_dim_log2++){
      work_dim = (1 << max_dim_log2);
      report(INFO,"starting on work_dim = %u (%ux%u)", work_dim, (1 << max_dim_log2), (1 << max_dim_log2));
    for (int frame_number = 0 ; frame_number < end_frame ; frame_number++) {
      Image* frame_rgb = frames_rgb[frame_number];
      //Image* frame_ycbcr = new Image(work_dim, work_dim, FULLSIZE);
      tick(&clock);
      convertRGBtoYCbCr(frame_rgb);
      conversion_total_t[frame_number] = elapsed_since(&clock);
      //delete frame_ycbcr;
    }
    std::sort(getTIFF_t, getTIFF_t+N_FRAMES);
    std::sort(upload_t, upload_t+N_FRAMES);
    std::sort(convert_t, convert_t+N_FRAMES);
    std::sort(readback_t, readback_t+N_FRAMES);
    std::sort(conversion_total_t, conversion_total_t+N_FRAMES);

    printf("%u %.4e %.4e %.4e %.4e %.4e %.4e\n", 1 << max_dim_log2, setup_t, getTIFF_t[0], upload_t[0], convert_t[0], readback_t[0], conversion_total_t[0]);
  }
    report(INFO, "Effective bandwith: %3.2f GB/s", ((4096*4096)*3*4*2)/(conversion_total_t[0]));
    return 0;
  }


  int main(int args, char** argv){
    encode();
    return 0;
  }
