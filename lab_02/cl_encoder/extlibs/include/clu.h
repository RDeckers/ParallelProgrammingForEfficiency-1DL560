#ifndef _CLU_H
#define _CLU_H

#ifdef __cplusplus
extern "C" {
#endif

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#define CLU_DYNAMIC -1

cl_uint cluPlatformCount();
cl_uint cluGetPlatforms(cl_platform_id **platforms, int max_platform_count);
size_t cluPlatformGetInfoSize(cl_platform_id platform, cl_platform_info param);
size_t cluGetPlatformInfo(cl_platform_id platform, cl_platform_info param, int max_size, void** output);

cl_uint cluDeviceCount(cl_platform_id platform, cl_device_type device_type);
cl_uint cluGetDevices(cl_platform_id platform, cl_device_type device_type, int max_count, cl_device_id **devices);
size_t cluDeviceGetInfoSize(cl_device_id device, cl_device_info param);
size_t cluGetDeviceInfo(cl_device_id device, cl_device_info param, int max_size, void** output);

cl_context cluCreateContextFromTypes(cl_platform_id platform, cl_device_type device_types);

cl_command_queue cluCreateCommandQueue(cl_context context, cl_device_id device);
int cluCreateCommandQueues(cl_context context, cl_device_id *devices, int n_devices, cl_command_queue **queues);
cl_program cluProgramFromFilename(cl_context context, const char *filename);

int cluGetProgramLog(cl_program program, cl_device_id device, int max_size, char **log);
size_t cluGetLogSize(cl_program program, cl_device_id device);

const char* cluErrorString(cl_int error);

#ifdef __cplusplus
}
#endif
#endif
