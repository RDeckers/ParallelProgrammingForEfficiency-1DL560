#ifndef custom_types_h
#define custom_types_h
#include <immintrin.h>
//typedef float __declspec(align(16)) v8f[8];
typedef __m256 v8f;
typedef __m256i v8i;
//typedef float __declspec(align(16)) v8f[8];
//typedef int v8i __attribute__ ((vector_size (32)));
#include <vector>
#include <string>


float max(float a, float b);

//float round(float number);

#include <immintrin.h>

class Channel{
private:

	//int padded_width;
	//int padded_height;
public:
	//TODO: these are padded to optimize blocking, make sure that is what we want everywhere.
	int width;
	int height;
	union{
    float* data;
    v8f* v_data;
  };
	//std::vector<float> *data;

	Channel(int _width, int _height);

	Channel(Channel* in);

	~Channel();
	//void operator=(Channel* c);
	void copy(Channel* c);
	inline float& get_ref(int x, int y) {
		return ((float*)data)[y*width + x];
	}
	inline float get(int x, int y) {
		return ((float*)data)[y*width + x];
	}
  inline v8f get_8(int x, int y) {
		return v_data[y*width/8 + x/8];
	}
	inline v8f get_8(int i) {
		return v_data[i/8];
	}
	inline void set(int x, int y, float value) {
		((float*)data)[y*width + x] = value;
	}
  inline void set_8(int x, int y, v8f value) {
		v_data[y*width/8 + x/8] = value;
	}
	inline void set_8(int i, v8f value) {
		v_data[i/8] = value;
	}
	inline float& get_ref(int i) {
		return ((float*)data)[i];
	}
	inline float get(int i) {
		return ((float*)data)[i];
	}
	inline void set(int i, float value) {
		((float*)data)[i] = value;
	}
};


class Image{
public:
	Channel *rc;
    Channel *gc;
    Channel *bc;
    int width;
    int height;
    // Flag to check if the image is downsampled or not
    int type;

	Image(int _w, int _h, int _type);

	~Image();
};


class Frame{
	public:
	Channel *Y;
    Channel *Cb;
    Channel *Cr;
    int width;
    int height;
    // Flag to check if the image is downsampled or not
    int type;

	Frame(int _w, int _h, int _type);
	Frame(Frame* in);

	~Frame();

};


class SMatrix{
public:
	std::string** data;
	int width;
    int height;

	SMatrix(int _width, int _height);

	~SMatrix();
	//void operator=(SMatrix* c);
};

class FrameEncode{
	public:
	SMatrix *Y;
    SMatrix *Cb;
    SMatrix *Cr;
    int width;
    int height;
    // Flag to check if the image is downsampled or not
    int type;

	FrameEncode(int _w, int _h, int _mpg);

	~FrameEncode();

};

typedef struct smVector{
    int a;
    int b;
} mVector;


#endif
