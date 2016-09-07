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
#include "gettimeofday.h"
#include "config.h"
#include "bench.h"

using namespace std;
#define OPTIM
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

void convertRGBtoYCbCr_inplace(Image* in) {
	int width = in->width;
	int height = in->height;

	for (int y = 0; y<height; y++) {
		for (int x = 0; x<width; x++) {
			float R = in->rc->get(x, y);
			float G = in->gc->get(x, y);
			float B = in->bc->get(x, y);
			float Y = 0 + ((float)0.299*R) + ((float)0.587*G) + ((float)0.113*B);
			float Cb = 128 - ((float)0.168736*R) - ((float)0.331264*G) + ((float)0.5*B);
			float Cr = 128 + ((float)0.5*R) - ((float)0.418688*G) - ((float)0.081312*B);
			in->rc->set(x, y, Y);
			in->gc->set(x, y, Cb);
			in->bc->set(x, y, Cr);
		}
	}
}

void convertRGBtoYCbCr(Image* in, Image* out){
    int width = in->width;
    int height = in->height;

		for(int y=0; y<width; y++) {
			for (int x = 0; x<height; x++) {

			float R = in->rc->get(y,x);
			float G = in->gc->get(y,x);
			float B = in->bc->get(y,x);
			float Y = 0+((float)0.299*R)+((float)0.587*G)+((float)0.113*B);
			float Cb = 128-((float)0.168736*R)-((float)0.331264*G)+((float)0.5*B);
            float Cr = 128+((float)0.5*R)-((float)0.418688*G)-((float)0.081312*B);
			out->rc->get_ref(y,x) = Y;
			out->gc->get_ref(y,x) = Cb;
			out->bc->get_ref(y,x) = Cr;
        }
    }

    //return out;
}
//
//Channel* lowPass(Channel* in, Channel* out) {
//	// Applies a simple 3-tap low-pass filter in the X- and Y- dimensions.
//	// E.g., blur
//	// weights for neighboring pixels
//	float a = 0.25;
//	float b = 0.5;
//	float c = 0.25;
//
//	int width = in->width;
//	int height = in->height;
//
//	out->copy(in);
//
//	const int block_size = 8;
//
//	const int block_x = (width-2) / block_size;
//	const int block_x_rem = (width-2) % block_size;
//
//	const int block_y = (height-2) / block_size;
//	const int block_y_rem = (height-2) % block_size;
//
//	float block_buff[(block_size+1)*(block_size+1)];
//
//
//	for (int by = 0; by < block_y; by++) {
//		for (int bx = 0; bx < block_x; bx++) {
//			for (int y = 0; y < block_size+1; y++) {
//				for (int x = 0; x < block_size+1; x++) {
//					block_buff[y*block_size + x] = in->get(x + bx*block_size, y + by*block_size);
//				}
//			}
//			//horizonatal blur
//			float prev = block_buff[block_size + 1];
//			float current = block_buff[block_size + 2];
//			for (int y = 1; y < block_size; y++) {
//				for (int x = 1; x < block_size; x++) {
//					float next = block_buff[y*block_size + x+1];
//					block_buff[y*block_size + x] = a*prev + b*current + c*next;
//					prev = current;
//					current = next;
//				}
//			}
//			//vertical blur
//			prev = block_buff[1];
//			current = block_buff[block_size + 2];
//			for (int y = 1; y < block_size; y++) {
//				for (int x = 1; x < block_size; x++) {
//					float next = block_buff[(y+1)*block_size + x];
//					block_buff[y*block_size + x] = a*prev + b*current + c*next;
//					prev = current;
//					current = next;
//				}
//			}
//
//			//WB
//			for (int y = 0; y < block_size; y++) {
//				for (int x = 0; x < block_size; x++) {
//					in->set(x + bx*block_size, y + by*block_size, block_buff[(y+1)*block_size + x + 1]);
//				}
//			}
//		}
//		//TODO: pad with to be multiple of 8.
//		return out;
//	}
//
//	//for (int y = 1; y < height - 1; y++) {
//	//	for (int x = 1; x < width - 1; x++) {
//	//		float c = in->get 
//	//		out->set(x,y )
//	//	}
//	//}
//
//	// In X
//	//for (int y = 1; y<(height - 1); y++) {
//	//	for (int x = 1; x<(width - 1); x++) {
//	//		out->set(x, y, a*in->get(x - 1, y) + b*in->get(x, y) + c*in->get(x + 1, y));
//	//	}
//	//}
//	//// In Y
//	//for (int y = 1; y<(height - 1); y++) {
//	//	for (int x = 1; x<(width - 1); x++) {
//	//		out->set(x, y, a*in->get(x, y - 1) + b*in->get(x, y) + c*in->get(x, y + 1));
//	//	}
//	//}
//
//	//return out;
//}
//
Channel* lowPass(Channel* in, Channel* out){
    // Applies a simple 3-tap low-pass filter in the X- and Y- dimensions.
    // E.g., blur
    // weights for neighboring pixels
    float a=0.25;
    float b=0.5;
    float c=0.25;

	int width = in->width;
	int height = in->height;

    //out = in; TODO Is this necessary?
	out->copy(in);


    // In X
    for (int y=1; y<(width-1); y++) {
        for (int x=1; x<(height-1); x++) {
            out->get_ref(y,x) = a*in->get(y,x-1)+b*in->get(y,x)+c*in->get(y, x+1);
        }
    }
    // In Y
    for (int y=1; y<(width-1); y++) {
        for (int x=1; x<(height-1); x++) {
			out->get_ref(y, x) = a*out->get(y-1, x) + b*out->get(y, x) + c*out->get(y+1, x);
        }
    }

    return out;
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

    for (int my=inset; my<height-(inset+window_size)+1; my+=block_size) {
      for (int mx=inset; mx<width-(inset+window_size)+1; mx+=block_size) {

            float best_match_sad = 1e10;
            int best_match_location[2] = {0, 0};

            for(int sy=my-window_size; sy<my+window_size; sy++) {
                for(int sx=mx-window_size; sx<mx+window_size; sx++) {
                    float current_match_sad = 0;
                    // Do the SAD
                    for (int y=0; y<block_size; y++) {
                        for (int x=0; x<block_size; x++) {
                            int match_x = mx+x;
                            int match_y = my+y;
                            int search_x = sx+x;
                            int search_y = sy+y;
                            float diff_Y = abs(match->Y->get(match_y, match_x) - source->Y->get(search_y, search_x));
							float diff_Cb = abs(match->Cb->get(match_y, match_x) - source->Cb->get(search_y, search_x));
							float diff_Cr = abs(match->Cr->get(match_y, match_x) - source->Cr->get(search_y, search_x));

                            float diff_total = Y_weight*diff_Y + Cb_weight*diff_Cb + Cr_weight*diff_Cr;
                            current_match_sad = current_match_sad + diff_total;

                        }
                    } //end SAD

                    if (current_match_sad <= best_match_sad){
                        best_match_sad = current_match_sad;
                        best_match_location[0] = sx-mx;
                        best_match_location[1] = sy-my;
                    }
                }
            }

            mVector v;
            v.a=best_match_location[0];
            v.b=best_match_location[1];
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


#ifndef OPTIM
Channel* downSample(Channel* in){
	int width = in->width;
	int height = in->height;

	Channel* out = new Channel(width/2,height/2);
	int w2 = out->width;
	int h2 = out->height;
	for(int y2=0,y=0; y2<w2; y2++) {
		for (int x2 = 0, x = 0; x2<h2; x2++) {
           out->set(y2, x2, in->get(y,x));
           x+=2;
       }
      y+=2;
    }
    return out;
}
#else
Channel* downSample(Channel* in) {
	int width = in->width;
	int height = in->height;

	Channel* out = new Channel(width / 2, height / 2);
	int w2 = out->width;
	int h2 = out->height;
	for (int y2 = 0, y = 0; y2<h2; y2++) {
		for (int x2 = 0, x = 0; x2<w2; x2++) {
			out->set(x2, y2, in->get(x, y));
			x += 2;
		}
		y += 2;
	}
	return out;
}
#endif

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


int encode() {
    int end_frame = int(N_FRAMES);
    int i_frame_frequency = int(I_FRAME_FREQ);
	//struct timeval starttime, endtime;
	struct timespec clock;
	double runtime[10] = {0};

    // Hardcoded paths
    string image_path =  "../../inputs/" + string(image_name) + "/" + image_name + ".";
	string stream_path = "../../outputs/stream_c_" + string(image_name) + ".xml";

    xmlDocPtr stream = NULL;

    Image* frame_rgb = NULL;
    Image* previous_frame_rgb = NULL;
    Frame* previous_frame_lowpassed = NULL;

    loadImage(0, image_path, &frame_rgb);

    int width = frame_rgb->width;
    int height = frame_rgb->height;
    int npixels = width*height;

	delete frame_rgb;

	createStatsFile();
    stream = create_xml_stream(width, height, QUALITY, WINDOW_SIZE, BLOCK_SIZE);
    vector<mVector>* motion_vectors = NULL;

    for (int frame_number = 0 ; frame_number < end_frame ; frame_number++) {
		frame_rgb = NULL;
        loadImage(frame_number, image_path, &frame_rgb);

        //  Convert to YCbCr
		print("Covert to YCbCr...");

		
		//Image* frame_ycbcr = new Image(width, height, FULLSIZE);
		tick(&clock);
		//convertRGBtoYCbCr(frame_rgb, frame_ycbcr);
		convertRGBtoYCbCr_inplace(frame_rgb);
		
		runtime[0] = tock(&clock);
		Image* frame_ycbcr = frame_rgb;
		frame_rgb = nullptr;
		dump_image(frame_ycbcr, "frame_ycbcr", frame_number);
		//delete frame_rgb;

        // We low pass filter Cb and Cr channesl
        print("Low pass filter...");

		//TODO: split up channels for better memory locality.
		tick(&clock);
		Channel* frame_blur_cb = new Channel(width, height);
        Channel* frame_blur_cr = new Channel(width, height);
		Frame *frame_lowpassed = new Frame(width, height, FULLSIZE);

		lowPass(frame_ycbcr->gc, frame_blur_cb);
		lowPass(frame_ycbcr->bc, frame_blur_cr);

		frame_lowpassed->Y->copy(frame_ycbcr->rc);
		frame_lowpassed->Cb->copy(frame_blur_cb);
        frame_lowpassed->Cr->copy(frame_blur_cr);
		runtime[1] = tock(&clock);
		

		dump_frame(frame_lowpassed, "frame_ycbcr_lowpass", frame_number);
		delete frame_ycbcr;
		delete frame_blur_cb;
		delete frame_blur_cr;

        Frame *frame_lowpassed_final = NULL;

        if (frame_number % i_frame_frequency != 0) {
            // We have a P frame
            // Note that in the first iteration we don't enter this branch!

			//Compute the motion vectors
			print("Motion Vector Search...");

			tick(&clock);
            motion_vectors = motionVectorSearch(previous_frame_lowpassed, frame_lowpassed, frame_lowpassed->width, frame_lowpassed->height);
			runtime[2] =  tock(&clock);

			print("Compute Delta...");
			tick(&clock);
            frame_lowpassed_final = computeDelta(previous_frame_lowpassed, frame_lowpassed, motion_vectors);
			runtime[3] = tock(&clock);

        } else {
            // We have a I frame
            motion_vectors = NULL;
            frame_lowpassed_final = new Frame(frame_lowpassed);
        }
		delete frame_lowpassed; frame_lowpassed=NULL;

		if (frame_number > 0) delete previous_frame_lowpassed;
        previous_frame_lowpassed = new Frame(frame_lowpassed_final);


        // Downsample the difference
		print("Downsample...");

		tick(&clock);
        Frame* frame_downsampled = new Frame(width, height, DOWNSAMPLE);

        // We don't touch the Y frame
		frame_downsampled->Y->copy(frame_lowpassed_final->Y);
        Channel* frame_downsampled_cb = downSample(frame_lowpassed_final->Cb);
		frame_downsampled->Cb->copy(frame_downsampled_cb);
        Channel* frame_downsampled_cr = downSample(frame_lowpassed_final->Cr);
		frame_downsampled->Cr->copy(frame_downsampled_cr);
  
		runtime[4] = tock(&clock);

        dump_frame(frame_downsampled, "frame_downsampled", frame_number);
		delete frame_lowpassed_final;
		delete frame_downsampled_cb;
		delete frame_downsampled_cr;

        // Convert to frequency domain
		print("Convert to frequency domain...");

		tick(&clock);
		Frame* frame_dct = new Frame(width, height, DOWNSAMPLE);

        dct8x8(frame_downsampled->Y, frame_dct->Y);
        dct8x8(frame_downsampled->Cb, frame_dct->Cb);
        dct8x8(frame_downsampled->Cr, frame_dct->Cr);
		runtime[5] = tock(&clock);

        dump_frame(frame_dct, "frame_dct", frame_number);
		delete frame_downsampled;

        //Quantize the data
		print("Quantize...");

		tick(&clock);
        Frame* frame_quant = new Frame(width, height, DOWNSAMPLE);

        quant8x8(frame_dct->Y, frame_quant->Y);
		quant8x8(frame_dct->Cb, frame_quant->Cb);
		quant8x8(frame_dct->Cr, frame_quant->Cr);
		runtime[6] = tock(&clock);

		dump_frame(frame_quant, "frame_quant", frame_number);
		delete frame_dct;

        //Extract the DC components and compute the differences
		print("Compute DC differences...");

		tick(&clock);
		Frame* frame_dc_diff = new Frame(1, (width/8)*(height/8), DCDIFF); //dealocate later

        dcDiff(frame_quant->Y, frame_dc_diff->Y);
        dcDiff(frame_quant->Cb, frame_dc_diff->Cb);
        dcDiff(frame_quant->Cr, frame_dc_diff->Cr);
		runtime[7] = tock(&clock);

        dump_dc_diff(frame_dc_diff, "frame_dc_diff", frame_number);

		// Zig-zag order for zero-counting
		print("Zig-zag order...");
		tick(&clock);

		Frame* frame_zigzag = new Frame(MPEG_CONSTANT, width*height/MPEG_CONSTANT, ZIGZAG);

        zigZagOrder(frame_quant->Y, frame_zigzag->Y);
        zigZagOrder(frame_quant->Cb, frame_zigzag->Cb);
        zigZagOrder(frame_quant->Cr, frame_zigzag->Cr);
		runtime[8] = tock(&clock);

		dump_zigzag(frame_zigzag, "frame_zigzag", frame_number);
		delete frame_quant;

        // Encode coefficients
		print("Encode coefficients...");

		tick(&clock);
		FrameEncode* frame_encode = new FrameEncode(width, height, MPEG_CONSTANT);

        encode8x8(frame_zigzag->Y, frame_encode->Y);
        encode8x8(frame_zigzag->Cb, frame_encode->Cb);
        encode8x8(frame_zigzag->Cr, frame_encode->Cr);
		runtime[9] = tock(&clock);

		delete frame_zigzag;

        stream_frame(stream, frame_number, motion_vectors, frame_number-1, frame_dc_diff, frame_encode);
        write_stream(stream_path, stream);

		delete frame_dc_diff;
		delete frame_encode;

        if (motion_vectors != NULL) {
            free(motion_vectors);
			motion_vectors = NULL;
        }

		writestats(frame_number, frame_number % i_frame_frequency, runtime);

    }

	closeStats();
	/* Uncoment to prevent visual studio output window from closing */
	//system("pause");

	return 0;
}


int main(int args, char** argv){
    encode();
    return 0;
}
