#include <sys/time.h>
#include "autoencoder.h"
#define KERNEL_SOURCE_LENGTH 50000

autoencoder_GPU::autoencoder_GPU():autoencoder(){

	// initialize the OpenCL environment
	gpu_init(gpu_env, 0);

	d_weight0 = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize0 * nLayerSize1 * sizeof(floatType), NULL, &gpu_env.status);
	d_weight1 = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize1 * nLayerSize2 * sizeof(floatType), NULL, &gpu_env.status);
	d_weight2 = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize2 * nLayerSize3 * sizeof(floatType), NULL, &gpu_env.status);
	d_weight3 = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize3 * nLayerSize4 * sizeof(floatType), NULL, &gpu_env.status);
	d_weight4 = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize4 * nLayerSize5 * sizeof(floatType), NULL, &gpu_env.status);
	d_weight5 = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize5 * nLayerSize6 * sizeof(floatType), NULL, &gpu_env.status);
	d_weight6 = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize6 * nLayerSize7 * sizeof(floatType), NULL, &gpu_env.status);
	d_weight7 = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize7 * nLayerSize8 * sizeof(floatType), NULL, &gpu_env.status);

	d_bias0 = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize1 * sizeof(floatType), NULL, &gpu_env.status);
	d_bias1 = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize2 * sizeof(floatType), NULL, &gpu_env.status);
	d_bias2 = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize3 * sizeof(floatType), NULL, &gpu_env.status);
	d_bias3 = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize4 * sizeof(floatType), NULL, &gpu_env.status);
	d_bias4 = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize5 * sizeof(floatType), NULL, &gpu_env.status);
	d_bias5 = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize6 * sizeof(floatType), NULL, &gpu_env.status);
	d_bias6 = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize7 * sizeof(floatType), NULL, &gpu_env.status);
	d_bias7 = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize8 * sizeof(floatType), NULL, &gpu_env.status);

	d_layer0act = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize0 * nVectorPerBatch * sizeof(floatType), NULL, &gpu_env.status);
	d_layer0err = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize0 * nVectorPerBatch * sizeof(floatType), NULL, &gpu_env.status);
	d_layer1act = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize1 * nVectorPerBatch * sizeof(floatType), NULL, &gpu_env.status);
	d_layer1err = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize1 * nVectorPerBatch * sizeof(floatType), NULL, &gpu_env.status);
	d_layer2act = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize2 * nVectorPerBatch * sizeof(floatType), NULL, &gpu_env.status);
	d_layer2err = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize2 * nVectorPerBatch * sizeof(floatType), NULL, &gpu_env.status);
	d_layer3act = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize3 * nVectorPerBatch * sizeof(floatType), NULL, &gpu_env.status);
	d_layer3err = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize3 * nVectorPerBatch * sizeof(floatType), NULL, &gpu_env.status);
	d_layer4act = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize4 * nVectorPerBatch * sizeof(floatType), NULL, &gpu_env.status);
	d_layer4err = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize4 * nVectorPerBatch * sizeof(floatType), NULL, &gpu_env.status);
	d_layer4state = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize4 * nVectorPerBatch * sizeof(floatType), NULL, &gpu_env.status);
	d_layer5act = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize5 * nVectorPerBatch * sizeof(floatType), NULL, &gpu_env.status);
	d_layer5err = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize5 * nVectorPerBatch * sizeof(floatType), NULL, &gpu_env.status);
	d_layer6act = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize6 * nVectorPerBatch * sizeof(floatType), NULL, &gpu_env.status);
	d_layer6err = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize6 * nVectorPerBatch * sizeof(floatType), NULL, &gpu_env.status);
	d_layer7act = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize7 * nVectorPerBatch * sizeof(floatType), NULL, &gpu_env.status);
	d_layer7err = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize7 * nVectorPerBatch * sizeof(floatType), NULL, &gpu_env.status);
	d_layer8act = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize8 * nVectorPerBatch * sizeof(floatType), NULL, &gpu_env.status);
	d_layer8err = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize8 * nVectorPerBatch * sizeof(floatType), NULL, &gpu_env.status);

	d_delta_weight0 = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize0 * nLayerSize1 * sizeof(floatType), NULL, &gpu_env.status);
	d_delta_weight1 = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize1 * nLayerSize2 * sizeof(floatType), NULL, &gpu_env.status);
	d_delta_weight2 = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize2 * nLayerSize3 * sizeof(floatType), NULL, &gpu_env.status);
	d_delta_weight3 = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize3 * nLayerSize4 * sizeof(floatType), NULL, &gpu_env.status);
	d_delta_weight4 = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize4 * nLayerSize5 * sizeof(floatType), NULL, &gpu_env.status);
	d_delta_weight5 = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize5 * nLayerSize6 * sizeof(floatType), NULL, &gpu_env.status);
	d_delta_weight6 = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize6 * nLayerSize7 * sizeof(floatType), NULL, &gpu_env.status);
	d_delta_weight7 = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize7 * nLayerSize8 * sizeof(floatType), NULL, &gpu_env.status);

	d_delta_bias0 = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize1 * sizeof(floatType), NULL, &gpu_env.status);
	d_delta_bias1 = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize2 * sizeof(floatType), NULL, &gpu_env.status);
	d_delta_bias2 = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize3 * sizeof(floatType), NULL, &gpu_env.status);
	d_delta_bias3 = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize4 * sizeof(floatType), NULL, &gpu_env.status);
	d_delta_bias4 = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize5 * sizeof(floatType), NULL, &gpu_env.status);
	d_delta_bias5 = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize6 * sizeof(floatType), NULL, &gpu_env.status);
	d_delta_bias6 = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize7 * sizeof(floatType), NULL, &gpu_env.status);
	d_delta_bias7 = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize8 * sizeof(floatType), NULL, &gpu_env.status);

	// error vector
	d_error = clCreateBuffer(gpu_env.ctx, CL_MEM_READ_WRITE, nLayerSize0 * nVectorPerBatch * sizeof(floatType), NULL, &gpu_env.status);

	// transfer data from CPU to GPU, TO DO

	// build OpenCL kernels
	char* source = new char[KERNEL_SOURCE_LENGTH];
	loadKernelSource("../src/gpu_rbm.cl", source);
	gpu_env.prog = clCreateProgramWithSource(gpu_env.ctx, 1, (const char**)&source, NULL, &gpu_env.status);

	gpu_env.status = clBuildProgram(gpu_env.prog, 0, NULL, NULL, NULL, NULL);

	if (gpu_env.status == CL_BUILD_PROGRAM_FAILURE) {
		// Determine the size of the log
		size_t log_size;
		clGetProgramBuildInfo(gpu_env.prog, gpu_env.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

		// Allocate memory for the log
		char *log = (char *) malloc(log_size);

		// Get the log
		clGetProgramBuildInfo(gpu_env.prog, gpu_env.device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

		// Print the log
		printf("%s\n", log);
		
		exit(0);
	}

	squareError		= clCreateKernel(gpu_env.prog, "squareError", &gpu_env.status);
	sigmoid			= clCreateKernel(gpu_env.prog, "sigmoid", &gpu_env.status);
	addBias			= clCreateKernel(gpu_env.prog, "addBias", &gpu_env.status);
	sumBatch		= clCreateKernel(gpu_env.prog, "sumBatch", &gpu_env.status);
	add				= clCreateKernel(gpu_env.prog, "add", &gpu_env.status);
	getStates		= clCreateKernel(gpu_env.prog, "getStates", &gpu_env.status);
	updateWeights	= clCreateKernel(gpu_env.prog, "updateWeights", &gpu_env.status);
	updateBias		= clCreateKernel(gpu_env.prog, "updateBias", &gpu_env.status);
	randNum			= clCreateKernel(gpu_env.prog, "PRNG_threefry4x32", &gpu_env.status);
	randn			= clCreateKernel(gpu_env.prog, "PRNGn_threefry4x32", &gpu_env.status);
	reset			= clCreateKernel(gpu_env.prog, "reset", &gpu_env.status);
	rounding		= clCreateKernel(gpu_env.prog, "rounding", &gpu_env.status);
	subtract		= clCreateKernel(gpu_env.prog, "subtract", &gpu_env.status);
	deriv			= clCreateKernel(gpu_env.prog, "deriv", &gpu_env.status);
	updateAE		= clCreateKernel(gpu_env.prog, "update", &gpu_env.status);

	gpu_env.status = clEnqueueWriteBuffer(gpu_env.queue, d_weight0, CL_TRUE, 0, nLayerSize0 * nLayerSize1 * sizeof(floatType), (void*)weight0, 0, NULL, NULL);
	gpu_env.status = clEnqueueWriteBuffer(gpu_env.queue, d_weight1, CL_TRUE, 0, nLayerSize1 * nLayerSize2 * sizeof(floatType), (void*)weight1, 0, NULL, NULL);
	gpu_env.status = clEnqueueWriteBuffer(gpu_env.queue, d_weight2, CL_TRUE, 0, nLayerSize2 * nLayerSize3 * sizeof(floatType), (void*)weight2, 0, NULL, NULL);
	gpu_env.status = clEnqueueWriteBuffer(gpu_env.queue, d_weight3, CL_TRUE, 0, nLayerSize3 * nLayerSize4 * sizeof(floatType), (void*)weight3, 0, NULL, NULL);
	gpu_env.status = clEnqueueWriteBuffer(gpu_env.queue, d_weight4, CL_TRUE, 0, nLayerSize4 * nLayerSize5 * sizeof(floatType), (void*)weight4, 0, NULL, NULL);
	gpu_env.status = clEnqueueWriteBuffer(gpu_env.queue, d_weight5, CL_TRUE, 0, nLayerSize5 * nLayerSize6 * sizeof(floatType), (void*)weight5, 0, NULL, NULL);
	gpu_env.status = clEnqueueWriteBuffer(gpu_env.queue, d_weight6, CL_TRUE, 0, nLayerSize6 * nLayerSize7 * sizeof(floatType), (void*)weight6, 0, NULL, NULL);
	gpu_env.status = clEnqueueWriteBuffer(gpu_env.queue, d_weight7, CL_TRUE, 0, nLayerSize7 * nLayerSize8 * sizeof(floatType), (void*)weight7, 0, NULL, NULL);

	gpu_env.status = clEnqueueWriteBuffer(gpu_env.queue, d_bias0, CL_TRUE, 0, nLayerSize1 * sizeof(floatType), (void*)bias0, 0, NULL, NULL);
	gpu_env.status = clEnqueueWriteBuffer(gpu_env.queue, d_bias1, CL_TRUE, 0, nLayerSize2 * sizeof(floatType), (void*)bias1, 0, NULL, NULL);
	gpu_env.status = clEnqueueWriteBuffer(gpu_env.queue, d_bias2, CL_TRUE, 0, nLayerSize3 * sizeof(floatType), (void*)bias2, 0, NULL, NULL);
	gpu_env.status = clEnqueueWriteBuffer(gpu_env.queue, d_bias3, CL_TRUE, 0, nLayerSize4 * sizeof(floatType), (void*)bias3, 0, NULL, NULL);
	gpu_env.status = clEnqueueWriteBuffer(gpu_env.queue, d_bias4, CL_TRUE, 0, nLayerSize5 * sizeof(floatType), (void*)bias4, 0, NULL, NULL);
	gpu_env.status = clEnqueueWriteBuffer(gpu_env.queue, d_bias5, CL_TRUE, 0, nLayerSize6 * sizeof(floatType), (void*)bias5, 0, NULL, NULL);
	gpu_env.status = clEnqueueWriteBuffer(gpu_env.queue, d_bias6, CL_TRUE, 0, nLayerSize7 * sizeof(floatType), (void*)bias6, 0, NULL, NULL);
	gpu_env.status = clEnqueueWriteBuffer(gpu_env.queue, d_bias7, CL_TRUE, 0, nLayerSize8 * sizeof(floatType), (void*)bias7, 0, NULL, NULL);

}

void autoencoder_GPU::fprop(){

	// 1st layer to 2nd layer
	gpu_addBias(gpu_env, addBias, d_layer1act, d_bias0, nLayerSize1, nVectorPerBatch, NULL);
	gpu_env.status = clAmdBlasSgemm(gpu_env.order, clAmdBlasNoTrans, clAmdBlasNoTrans, nLayerSize1, nVectorPerBatch, nLayerSize0, 1.0, d_weight0, nLayerSize1, d_layer0act, nLayerSize0, 1.0, d_layer1act, nLayerSize1, 1, &gpu_env.queue, 0, NULL, NULL);
	gpu_sigmoid(gpu_env, sigmoid, d_layer1act, nLayerSize1 * nVectorPerBatch, NULL);

	// 2nd layer to 3rd layer
	gpu_addBias(gpu_env, addBias, d_layer2act, d_bias1, nLayerSize2, nVectorPerBatch, NULL);
	gpu_env.status = clAmdBlasSgemm(gpu_env.order, clAmdBlasNoTrans, clAmdBlasNoTrans, nLayerSize2, nVectorPerBatch, nLayerSize1, 1.0, d_weight1, nLayerSize2, d_layer1act, nLayerSize1, 1.0, d_layer2act, nLayerSize2, 1, &gpu_env.queue, 0, NULL, NULL);
	gpu_sigmoid(gpu_env, sigmoid, d_layer2act, nLayerSize2 * nVectorPerBatch, NULL);

	// 3rd layer to 4th layer
	gpu_addBias(gpu_env, addBias, d_layer3act, d_bias2, nLayerSize3, nVectorPerBatch, NULL);
	gpu_env.status = clAmdBlasSgemm(gpu_env.order, clAmdBlasNoTrans, clAmdBlasNoTrans, nLayerSize3, nVectorPerBatch, nLayerSize2, 1.0, d_weight2, nLayerSize3, d_layer2act, nLayerSize2, 1.0, d_layer3act, nLayerSize3, 1, &gpu_env.queue, 0, NULL, NULL);
	gpu_sigmoid(gpu_env, sigmoid, d_layer3act, nLayerSize3 * nVectorPerBatch, NULL);

	// 4th layer to 5th layer
	gpu_addBias(gpu_env, addBias, d_layer4act, d_bias3, nLayerSize4, nVectorPerBatch, NULL);
	gpu_env.status = clAmdBlasSgemm(gpu_env.order, clAmdBlasNoTrans, clAmdBlasNoTrans, nLayerSize4, nVectorPerBatch, nLayerSize3, 1.0, d_weight3, nLayerSize4, d_layer3act, nLayerSize3, 1.0, d_layer4act, nLayerSize4, 1, &gpu_env.queue, 0, NULL, NULL);
	gpu_sigmoid(gpu_env, sigmoid, d_layer4act, nLayerSize4 * nVectorPerBatch, NULL);

	// rounding for Layer 4
	gpu_rounding(gpu_env, rounding, d_layer4state, d_layer4act, nLayerSize4 * nVectorPerBatch, NULL);

	// 5th layer to 6th layer
	gpu_addBias(gpu_env, addBias, d_layer5act, d_bias4, nLayerSize5, nVectorPerBatch, NULL);
	gpu_env.status = clAmdBlasSgemm(gpu_env.order, clAmdBlasNoTrans, clAmdBlasNoTrans, nLayerSize5, nVectorPerBatch, nLayerSize4, 1.0, d_weight4, nLayerSize5, d_layer4state, nLayerSize4, 1.0, d_layer5act, nLayerSize5, 1, &gpu_env.queue, 0, NULL, NULL);
	gpu_sigmoid(gpu_env, sigmoid, d_layer5act, nLayerSize5 * nVectorPerBatch, NULL);

	// 6th layer to 7th layer
	gpu_addBias(gpu_env, addBias, d_layer6act, d_bias5, nLayerSize6, nVectorPerBatch, NULL);
	gpu_env.status = clAmdBlasSgemm(gpu_env.order, clAmdBlasNoTrans, clAmdBlasNoTrans, nLayerSize6, nVectorPerBatch, nLayerSize5, 1.0, d_weight5, nLayerSize6, d_layer5act, nLayerSize5, 1.0, d_layer6act, nLayerSize6, 1, &gpu_env.queue, 0, NULL, NULL);
	gpu_sigmoid(gpu_env, sigmoid, d_layer6act, nLayerSize6 * nVectorPerBatch, NULL);

	// 7th layer to 8th layer
	gpu_addBias(gpu_env, addBias, d_layer7act, d_bias6, nLayerSize7, nVectorPerBatch, NULL);
	gpu_env.status = clAmdBlasSgemm(gpu_env.order, clAmdBlasNoTrans, clAmdBlasNoTrans, nLayerSize7, nVectorPerBatch, nLayerSize6, 1.0, d_weight6, nLayerSize7, d_layer6act, nLayerSize6, 1.0, d_layer7act, nLayerSize7, 1, &gpu_env.queue, 0, NULL, NULL);
	gpu_sigmoid(gpu_env, sigmoid, d_layer7act, nLayerSize7 * nVectorPerBatch, NULL);

	// 8th layer to 9th layer
	gpu_addBias(gpu_env, addBias, d_layer8act, d_bias7, nLayerSize8, nVectorPerBatch, NULL);
	gpu_env.status = clAmdBlasSgemm(gpu_env.order, clAmdBlasNoTrans, clAmdBlasNoTrans, nLayerSize8, nVectorPerBatch, nLayerSize7, 1.0, d_weight7, nLayerSize8, d_layer7act, nLayerSize7, 1.0, d_layer8act, nLayerSize8, 1, &gpu_env.queue, 0, NULL, NULL);
	gpu_sigmoid(gpu_env, sigmoid, d_layer8act, nLayerSize8 * nVectorPerBatch, NULL);


}

void autoencoder_GPU::bprop(){
	gpu_subtract(gpu_env, subtract, d_layer8err, d_layer8act, d_layer0act, nLayerSize0 * nVectorPerBatch, NULL);

	// 9th layer to 8th layer
	gpu_env.status = clAmdBlasSgemm(gpu_env.order, clAmdBlasTrans, clAmdBlasNoTrans, nLayerSize7, nVectorPerBatch, nLayerSize8, 1.0, d_weight7, nLayerSize8, d_layer8err, nLayerSize8, 0.0, d_layer7err, nLayerSize7, 1, &gpu_env.queue, 0, NULL, NULL);

	gpu_deriv(gpu_env, deriv, d_layer7err, d_layer7act, nLayerSize7 * nVectorPerBatch, NULL);
	gpu_sumBatch(gpu_env, sumBatch, d_layer8err, d_delta_bias7, nLayerSize8, nVectorPerBatch, NULL);

	gpu_env.status = clAmdBlasSgemm(gpu_env.order, clAmdBlasNoTrans, clAmdBlasTrans, nLayerSize8, nLayerSize7, nVectorPerBatch, 1.0, d_layer8err, nLayerSize8, d_layer7act, nLayerSize7, 0.0, d_delta_weight7, nLayerSize8, 1, &gpu_env.queue, 0, NULL, NULL);

	// 8th layer to 7th layer
	gpu_env.status = clAmdBlasSgemm(gpu_env.order, clAmdBlasTrans, clAmdBlasNoTrans, nLayerSize6, nVectorPerBatch, nLayerSize7, 1.0, d_weight6, nLayerSize7, d_layer7err, nLayerSize7, 0.0, d_layer6err, nLayerSize6, 1, &gpu_env.queue, 0, NULL, NULL);

	gpu_deriv(gpu_env, deriv, d_layer6err, d_layer6act, nLayerSize6 * nVectorPerBatch, NULL);
	gpu_sumBatch(gpu_env, sumBatch, d_layer7err, d_delta_bias6, nLayerSize7, nVectorPerBatch, NULL);

	gpu_env.status = clAmdBlasSgemm(gpu_env.order, clAmdBlasNoTrans, clAmdBlasTrans, nLayerSize7, nLayerSize6, nVectorPerBatch, 1.0, d_layer7err, nLayerSize7, d_layer6act, nLayerSize6, 0.0, d_delta_weight6, nLayerSize7, 1, &gpu_env.queue, 0, NULL, NULL);

	// 7th layer to 6th layer
	gpu_env.status = clAmdBlasSgemm(gpu_env.order, clAmdBlasTrans, clAmdBlasNoTrans, nLayerSize5, nVectorPerBatch, nLayerSize6, 1.0, d_weight5, nLayerSize6, d_layer6err, nLayerSize6, 0.0, d_layer5err, nLayerSize5, 1, &gpu_env.queue, 0, NULL, NULL);

	gpu_deriv(gpu_env, deriv, d_layer5err, d_layer5act, nLayerSize5 * nVectorPerBatch, NULL);
	gpu_sumBatch(gpu_env, sumBatch, d_layer6err, d_delta_bias5, nLayerSize6, nVectorPerBatch, NULL);

	gpu_env.status = clAmdBlasSgemm(gpu_env.order, clAmdBlasNoTrans, clAmdBlasTrans, nLayerSize6, nLayerSize5, nVectorPerBatch, 1.0, d_layer6err, nLayerSize6, d_layer5act, nLayerSize5, 0.0, d_delta_weight5, nLayerSize6, 1, &gpu_env.queue, 0, NULL, NULL);

	// 6th layer to 5th layer
	gpu_env.status = clAmdBlasSgemm(gpu_env.order, clAmdBlasTrans, clAmdBlasNoTrans, nLayerSize4, nVectorPerBatch, nLayerSize5, 1.0, d_weight4, nLayerSize5, d_layer5err, nLayerSize5, 0.0, d_layer4err, nLayerSize4, 1, &gpu_env.queue, 0, NULL, NULL);

	gpu_deriv(gpu_env, deriv, d_layer4err, d_layer4act, nLayerSize4 * nVectorPerBatch, NULL);
	gpu_sumBatch(gpu_env, sumBatch, d_layer5err, d_delta_bias4, nLayerSize5, nVectorPerBatch, NULL);

	gpu_env.status = clAmdBlasSgemm(gpu_env.order, clAmdBlasNoTrans, clAmdBlasTrans, nLayerSize5, nLayerSize4, nVectorPerBatch, 1.0, d_layer5err, nLayerSize5, d_layer4act, nLayerSize4, 0.0, d_delta_weight4, nLayerSize5, 1, &gpu_env.queue, 0, NULL, NULL);

	// 5th layer to 4th layer
	gpu_env.status = clAmdBlasSgemm(gpu_env.order, clAmdBlasTrans, clAmdBlasNoTrans, nLayerSize3, nVectorPerBatch, nLayerSize4, 1.0, d_weight3, nLayerSize4, d_layer4err, nLayerSize4, 0.0, d_layer3err, nLayerSize3, 1, &gpu_env.queue, 0, NULL, NULL);

	gpu_deriv(gpu_env, deriv, d_layer3err, d_layer3act, nLayerSize3 * nVectorPerBatch, NULL);
	gpu_sumBatch(gpu_env, sumBatch, d_layer4err, d_delta_bias3, nLayerSize4, nVectorPerBatch, NULL);

	gpu_env.status = clAmdBlasSgemm(gpu_env.order, clAmdBlasNoTrans, clAmdBlasTrans, nLayerSize4, nLayerSize3, nVectorPerBatch, 1.0, d_layer4err, nLayerSize4, d_layer3act, nLayerSize3, 0.0, d_delta_weight3, nLayerSize4, 1, &gpu_env.queue, 0, NULL, NULL);

	// 4th layer to 3rd layer
	gpu_env.status = clAmdBlasSgemm(gpu_env.order, clAmdBlasTrans, clAmdBlasNoTrans, nLayerSize2, nVectorPerBatch, nLayerSize3, 1.0, d_weight2, nLayerSize3, d_layer3err, nLayerSize3, 0.0, d_layer2err, nLayerSize2, 1, &gpu_env.queue, 0, NULL, NULL);

	gpu_deriv(gpu_env, deriv, d_layer2err, d_layer2act, nLayerSize2 * nVectorPerBatch, NULL);
	gpu_sumBatch(gpu_env, sumBatch, d_layer3err, d_delta_bias2, nLayerSize3, nVectorPerBatch, NULL);

	gpu_env.status = clAmdBlasSgemm(gpu_env.order, clAmdBlasNoTrans, clAmdBlasTrans, nLayerSize3, nLayerSize2, nVectorPerBatch, 1.0, d_layer3err, nLayerSize3, d_layer2act, nLayerSize2, 0.0, d_delta_weight2, nLayerSize3, 1, &gpu_env.queue, 0, NULL, NULL);

	// 3rd layer to 2nd layer
	gpu_env.status = clAmdBlasSgemm(gpu_env.order, clAmdBlasTrans, clAmdBlasNoTrans, nLayerSize1, nVectorPerBatch, nLayerSize2, 1.0, d_weight1, nLayerSize2, d_layer2err, nLayerSize2, 0.0, d_layer1err, nLayerSize1, 1, &gpu_env.queue, 0, NULL, NULL);

	gpu_deriv(gpu_env, deriv, d_layer1err, d_layer1act, nLayerSize1 * nVectorPerBatch, NULL);
	gpu_sumBatch(gpu_env, sumBatch, d_layer2err, d_delta_bias1, nLayerSize2, nVectorPerBatch, NULL);

	gpu_env.status = clAmdBlasSgemm(gpu_env.order, clAmdBlasNoTrans, clAmdBlasTrans, nLayerSize2, nLayerSize1, nVectorPerBatch, 1.0, d_layer2err, nLayerSize2, d_layer1act, nLayerSize1, 0.0, d_delta_weight1, nLayerSize2, 1, &gpu_env.queue, 0, NULL, NULL);

	// 2nd layer to 1st layer
	gpu_env.status = clAmdBlasSgemm(gpu_env.order, clAmdBlasTrans, clAmdBlasNoTrans, nLayerSize0, nVectorPerBatch, nLayerSize1, 1.0, d_weight0, nLayerSize1, d_layer1err, nLayerSize1, 0.0, d_layer0err, nLayerSize0, 1, &gpu_env.queue, 0, NULL, NULL);

	gpu_deriv(gpu_env, deriv, d_layer0err, d_layer0act, nLayerSize0 * nVectorPerBatch, NULL);
	gpu_sumBatch(gpu_env, sumBatch, d_layer1err, d_delta_bias0, nLayerSize1, nVectorPerBatch, NULL);

	gpu_env.status = clAmdBlasSgemm(gpu_env.order, clAmdBlasNoTrans, clAmdBlasTrans, nLayerSize1, nLayerSize0, nVectorPerBatch, 1.0, d_layer1err, nLayerSize1, d_layer0act, nLayerSize0, 0.0, d_delta_weight0, nLayerSize1, 1, &gpu_env.queue, 0, NULL, NULL);

}

void autoencoder_GPU::update(){
	gpu_update(gpu_env, updateAE, d_weight7, d_delta_weight7, d_bias7, d_delta_bias7, eps_w, eps_b, nLayerSize7, nLayerSize8, NULL);

	gpu_update(gpu_env, updateAE, d_weight6, d_delta_weight6, d_bias6, d_delta_bias6, eps_w, eps_b, nLayerSize6, nLayerSize7, NULL);

	gpu_update(gpu_env, updateAE, d_weight5, d_delta_weight5, d_bias5, d_delta_bias5, eps_w, eps_b, nLayerSize5, nLayerSize6, NULL);

	gpu_update(gpu_env, updateAE, d_weight4, d_delta_weight4, d_bias4, d_delta_bias4, eps_w, eps_b, nLayerSize4, nLayerSize5, NULL);

	gpu_update(gpu_env, updateAE, d_weight3, d_delta_weight3, d_bias3, d_delta_bias3, eps_w, eps_b, nLayerSize3, nLayerSize4, NULL);

	gpu_update(gpu_env, updateAE, d_weight2, d_delta_weight2, d_bias2, d_delta_bias2, eps_w, eps_b, nLayerSize2, nLayerSize3, NULL);

	gpu_update(gpu_env, updateAE, d_weight1, d_delta_weight1, d_bias1, d_delta_bias1, eps_w, eps_b, nLayerSize1, nLayerSize2, NULL);

	gpu_update(gpu_env, updateAE, d_weight0, d_delta_weight0, d_bias0, d_delta_bias0, eps_w, eps_b, nLayerSize0, nLayerSize1, NULL);

}

void autoencoder_GPU::train(){
	for(int epoch = 0; epoch < nEpochNum; epoch++){
		dataprovider->reset();
		printf("Epoch %d\n", epoch + 1);
		gpu_reset(gpu_env, reset, d_error, nLayerSize0 * nVectorPerBatch, NULL);	

		for(int batch = 0; batch < nBatchNum; batch++){
			dataprovider->getNextDeviceBatch(d_layer0act);
			fprop();
			/*
			if(batch == 1){
				gpu_env.status = clEnqueueReadBuffer(gpu_env.queue, d_bias0, CL_TRUE, 0, nLayerSize1 * sizeof(floatType), (void*)bias0, 0, NULL, NULL);
				ofstream tempStream;
				tempStream.open("../log/bias.log", ios_base::trunc);
				for(unsigned i = 0; i < nLayerSize1; i++){
					tempStream << bias0[i] << ',';
					if((i + 1) % nLayerSize1 == 0){
						tempStream << endl;
					}
				}
				tempStream.close();
			}

			if(batch == 1){
				gpu_env.status = clEnqueueReadBuffer(gpu_env.queue, d_weight0, CL_TRUE, 0, nLayerSize0 * nLayerSize1 * sizeof(floatType), (void*)weight0, 0, NULL, NULL);
				ofstream tempStream;
				tempStream.open("../log/weight.log", ios_base::trunc);
				for(unsigned i = 0; i < nLayerSize0 * nLayerSize1; i++){
					tempStream << weight0[i] << ',';
					if((i + 1) % nLayerSize0 == 0){
						tempStream << endl;
					}
				}
				tempStream.close();
			}

			if(batch == 1){
				gpu_env.status = clEnqueueReadBuffer(gpu_env.queue, d_layer1act, CL_TRUE, 0, nVectorPerBatch * nLayerSize1 * sizeof(floatType), (void*)layer1act, 0, NULL, NULL);
				ofstream tempStream;
				tempStream.open("../log/activation.log", ios_base::trunc);
				for(unsigned i = 0; i < nVectorPerBatch * nLayerSize1; i++){
					tempStream << layer1act[i] << ',';
					if((i + 1) % nVectorPerBatch == 0){
						tempStream << endl;
					}
				}
				tempStream.close();
				// exit(0);
			}
			*/

			gpu_squareError(gpu_env, squareError, d_layer8act, d_layer0act, d_error, nLayerSize0 * nVectorPerBatch);
			bprop();
			update();
		
			/*
			if(!epoch){
				double errsum = 0.0;
				gpu_env.status = clEnqueueReadBuffer(gpu_env.queue, d_error, CL_TRUE, 0, nLayerSize0 * nVectorPerBatch * sizeof(floatType), (void*)error, 0, NULL, NULL);
				for(int i = 0; i < nLayerSize8 * nVectorPerBatch; i++){
					errsum += error[i];
				}
				printf("Epoch %d Batch %d Error %f\n", epoch + 1, batch + 1, errsum);
			}
			*/

		}

		double errsum = 0.0;
		gpu_env.status = clEnqueueReadBuffer(gpu_env.queue, d_error, CL_TRUE, 0, nLayerSize0 * nVectorPerBatch * sizeof(floatType), (void*)error, 0, NULL, NULL);
		for(int i = 0; i < nLayerSize8 * nVectorPerBatch; i++){
			errsum += error[i];
		}
		printf("Epoch %d Error %f\n", epoch + 1, errsum);

		ofstream fout;
		fout.open("../log/errorLog.txt", ios_base::app);
		struct timeval now;
		gettimeofday(&now, NULL);
		fout << now.tv_sec << ',' << errsum << endl;
		fout.close();
	}

	ofstream fout;

	gpu_env.status = clEnqueueReadBuffer(gpu_env.queue, d_weight0, CL_TRUE, 0, nLayerSize0 * nLayerSize1 * sizeof(floatType), (void*)weight0, 0, NULL, NULL);
	gpu_env.status = clEnqueueReadBuffer(gpu_env.queue, d_weight1, CL_TRUE, 0, nLayerSize1 * nLayerSize2 * sizeof(floatType), (void*)weight1, 0, NULL, NULL);
	gpu_env.status = clEnqueueReadBuffer(gpu_env.queue, d_weight2, CL_TRUE, 0, nLayerSize2 * nLayerSize3 * sizeof(floatType), (void*)weight2, 0, NULL, NULL);
	gpu_env.status = clEnqueueReadBuffer(gpu_env.queue, d_weight3, CL_TRUE, 0, nLayerSize3 * nLayerSize4 * sizeof(floatType), (void*)weight3, 0, NULL, NULL);
	gpu_env.status = clEnqueueReadBuffer(gpu_env.queue, d_weight4, CL_TRUE, 0, nLayerSize4 * nLayerSize5 * sizeof(floatType), (void*)weight4, 0, NULL, NULL);
	gpu_env.status = clEnqueueReadBuffer(gpu_env.queue, d_weight5, CL_TRUE, 0, nLayerSize5 * nLayerSize6 * sizeof(floatType), (void*)weight5, 0, NULL, NULL);
	gpu_env.status = clEnqueueReadBuffer(gpu_env.queue, d_weight6, CL_TRUE, 0, nLayerSize6 * nLayerSize7 * sizeof(floatType), (void*)weight6, 0, NULL, NULL);
	gpu_env.status = clEnqueueReadBuffer(gpu_env.queue, d_weight7, CL_TRUE, 0, nLayerSize7 * nLayerSize8 * sizeof(floatType), (void*)weight7, 0, NULL, NULL);

	gpu_env.status = clEnqueueReadBuffer(gpu_env.queue, d_bias0, CL_TRUE, 0, nLayerSize1 * sizeof(floatType), (void*)bias0, 0, NULL, NULL);
	gpu_env.status = clEnqueueReadBuffer(gpu_env.queue, d_bias1, CL_TRUE, 0, nLayerSize2 * sizeof(floatType), (void*)bias1, 0, NULL, NULL);
	gpu_env.status = clEnqueueReadBuffer(gpu_env.queue, d_bias2, CL_TRUE, 0, nLayerSize3 * sizeof(floatType), (void*)bias2, 0, NULL, NULL);
	gpu_env.status = clEnqueueReadBuffer(gpu_env.queue, d_bias3, CL_TRUE, 0, nLayerSize4 * sizeof(floatType), (void*)bias3, 0, NULL, NULL);
	gpu_env.status = clEnqueueReadBuffer(gpu_env.queue, d_bias4, CL_TRUE, 0, nLayerSize5 * sizeof(floatType), (void*)bias4, 0, NULL, NULL);
	gpu_env.status = clEnqueueReadBuffer(gpu_env.queue, d_bias5, CL_TRUE, 0, nLayerSize6 * sizeof(floatType), (void*)bias5, 0, NULL, NULL);
	gpu_env.status = clEnqueueReadBuffer(gpu_env.queue, d_bias6, CL_TRUE, 0, nLayerSize7 * sizeof(floatType), (void*)bias6, 0, NULL, NULL);
	gpu_env.status = clEnqueueReadBuffer(gpu_env.queue, d_bias7, CL_TRUE, 0, nLayerSize8 * sizeof(floatType), (void*)bias7, 0, NULL, NULL);

	fout.open("../data/autoencoderWeight.dat", ios_base::binary | ios_base::trunc);
	fout.write((char*)weight0, nLayerSize0 * nLayerSize1 * sizeof(floatType));
	fout.write((char*)weight1, nLayerSize1 * nLayerSize2 * sizeof(floatType));
	fout.write((char*)weight2, nLayerSize2 * nLayerSize3 * sizeof(floatType));
	fout.write((char*)weight3, nLayerSize3 * nLayerSize4 * sizeof(floatType));
	fout.write((char*)weight4, nLayerSize4 * nLayerSize5 * sizeof(floatType));
	fout.write((char*)weight5, nLayerSize5 * nLayerSize6 * sizeof(floatType));
	fout.write((char*)weight6, nLayerSize6 * nLayerSize7 * sizeof(floatType));
	fout.write((char*)weight7, nLayerSize7 * nLayerSize8 * sizeof(floatType));
	fout.close();
	fout.open("../data/autoencoderBias.dat", ios_base::binary | ios_base::trunc);
	fout.write((char*)bias0, nLayerSize1 * sizeof(floatType));
	fout.write((char*)bias1, nLayerSize2 * sizeof(floatType));
	fout.write((char*)bias2, nLayerSize3 * sizeof(floatType));
	fout.write((char*)bias3, nLayerSize4 * sizeof(floatType));
	fout.write((char*)bias4, nLayerSize5 * sizeof(floatType));
	fout.write((char*)bias5, nLayerSize6 * sizeof(floatType));
	fout.write((char*)bias6, nLayerSize7 * sizeof(floatType));
	fout.write((char*)bias7, nLayerSize8 * sizeof(floatType));
	fout.close();
		
}

