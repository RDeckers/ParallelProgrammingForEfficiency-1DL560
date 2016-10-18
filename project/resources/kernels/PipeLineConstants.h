//
// Hard code kernel work group size, useful for static local memory allocation
//
#define DIM_X 64
#define DIM_Y 8
// LMEM size must account for work filter overlap due to 
#define LMEM_X (DIM_X+2)
#define LMEM_Y (DIM_Y+2)