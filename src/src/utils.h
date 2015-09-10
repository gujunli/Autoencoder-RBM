#ifndef _UTILS_H_
#define _UTILS_H_

#define _AMD_CPU_
#define _AMD_GPU_

// choose platform to compile
#ifdef _AMD_CPU_

#include<acml.h>

#endif

#ifdef _AMD_GPU_

#include<CL/opencl.h>
#include<clAmdBlas.h>

#endif

#include<iostream>
#include<fstream>
#include<cstdlib>
#include<vector>

using namespace std;

// float points precision
typedef float floatType;

class CL_ENV
{
public:
	cl_int status;
	cl_platform_id platform;
	cl_device_id device;
	cl_context_properties props[3];
	cl_context ctx;
	cl_command_queue queue;
	cl_event event;
	cl_program prog;
	clAmdBlasOrder order;
};

void reset(floatType* a, unsigned int n);

void gpu_reset(CL_ENV gpu_env, cl_kernel biasKernel, cl_mem a, unsigned n, cl_event* event);

void randomInit(floatType* a, unsigned int n, floatType inf, floatType sup);

void gpu_randomInit(CL_ENV gpu_env, cl_kernel kern, cl_mem a, unsigned int n, floatType inf, floatType sup, cl_event* event);

void gaussInit(floatType* a, unsigned int n, floatType E, floatType V);

void gpu_gaussInit(CL_ENV gpu_env, cl_kernel kern, cl_mem a, unsigned int n, floatType E, floatType V, cl_event* event);

floatType gaussRand(floatType E, floatType V);

void gpu_random(CL_ENV gpu_env, cl_kernel biasKernel, cl_mem a, unsigned int n, floatType inf, floatType sup, cl_event* event);

void sigmoid(floatType* a, unsigned int n);

void gpu_sigmoid(CL_ENV gpu_env, cl_kernel biasKernel, cl_mem a, unsigned int n, cl_event* event);

void addBias(floatType* prob, floatType* bias, unsigned int layerSize, unsigned int nVectorPerBatch);

void gpu_addBias(CL_ENV gpu_env, cl_kernel biasKernel, cl_mem prob, cl_mem bias, unsigned int layerSize, unsigned int nVectorPerBatch, cl_event* event);

void sumBatch(floatType* prob, floatType* sum, unsigned int layerSize, unsigned nVectorPerBatch);

void gpu_sumBatch(CL_ENV gpu_env, cl_kernel biasKernel, cl_mem prob, cl_mem bias, unsigned int layerSize, unsigned int nVectorPerBatch, cl_event* event);

void gpu_add(CL_ENV gpu_env, cl_kernel biasKernel, cl_mem a, cl_mem b, unsigned int n, cl_event* event);

void gpu_getStates(CL_ENV gpu_env, cl_kernel biasKernel, cl_mem states, cl_mem prob, unsigned int n, cl_event* event);

void gpu_rounding(CL_ENV gpu_env, cl_kernel kern, cl_mem states, cl_mem prob, unsigned int n, cl_event* event);

void gpu_subtract(CL_ENV gpu_env, cl_kernel kern, cl_mem a, cl_mem b, cl_mem c, unsigned int n, cl_event* event);

void gpu_deriv(CL_ENV gpu_env, cl_kernel kern, cl_mem err, cl_mem act, unsigned int n, cl_event* event);

void gpu_update(CL_ENV gpu_env, cl_kernel kern, cl_mem weight, cl_mem delta_weight, cl_mem bias, cl_mem delta_bias, floatType eps_w, floatType eps_b, unsigned int nBottomLayerSize, unsigned int nUpperLayerSize, cl_event* event);

floatType EuDist(floatType* a, floatType* b, unsigned int n);

void logData(string filename, floatType* data, unsigned int Length, unsigned int stride, unsigned int nImageNum);

void logBinaryData(string filename, floatType* data, unsigned int Length, unsigned int stride, unsigned int nImageNum);

void gpu_init(CL_ENV& cl_env, unsigned int deviceIndex);

void gpu_updateWeights(CL_ENV gpu_env, cl_kernel biasKernel, cl_mem weights, cl_mem delta_weights, cl_mem posProds, cl_mem negProds, floatType momentum, floatType eps_w, floatType weightCost, unsigned int nVisLayerSize, unsigned int nHidLayerSize, unsigned int nVectorPerBatch, cl_event* event);

void gpu_updateBias(CL_ENV gpu_env, cl_kernel biasKernel, cl_mem bias, cl_mem delta_bias, cl_mem posAct, cl_mem negAct, floatType momentum, floatType eps_b, unsigned int nLayerSize, unsigned int nVectorPerBatch, cl_event* event);

void gpu_squareError(CL_ENV gpu_env, cl_kernel biasKernel, cl_mem a, cl_mem b, cl_mem c, unsigned int n);

void loadKernelSource(string filename, char* source);

#endif
