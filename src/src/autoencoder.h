#ifndef _AUTOENCODER_H_
#define _AUTOENCODER_H_

#include "utils.h"
#include "cifar10.h"

class autoencoder{
protected:
	// member variables for the hyper-parameters in the RBM implementation
	unsigned int nEpochNum; // total number of epoches
	unsigned int nBatchNum; // total number of mini-batches
	unsigned int nVectorPerBatch; // the number of input vectors in each min-batch

	// The size of each layer
	unsigned int nLayerSize0; 
	unsigned int nLayerSize1;
	unsigned int nLayerSize2;
	unsigned int nLayerSize3;
	unsigned int nLayerSize4;
	unsigned int nLayerSize5;
	unsigned int nLayerSize6;
	unsigned int nLayerSize7;
	unsigned int nLayerSize8;

	floatType eps_w; // learning rate for weights
	floatType eps_b; // learning rate for biases

	// network parameters
	floatType* weight0;
	floatType* weight1;
	floatType* weight2;
	floatType* weight3;
	floatType* weight4;
	floatType* weight5;
	floatType* weight6;
	floatType* weight7;

	floatType* bias0;
	floatType* bias1;
	floatType* bias2;
	floatType* bias3;
	floatType* bias4;
	floatType* bias5;
	floatType* bias6;
	floatType* bias7;

	// layers of the autoencoder network, both activations and errors
	floatType* layer0act;
	floatType* layer0err;
	floatType* layer1act;
	floatType* layer1err;
	floatType* layer2act;
	floatType* layer2err;
	floatType* layer3act;
	floatType* layer3err;
	floatType* layer4act;
	floatType* layer4err;
	floatType* layer4state;
	floatType* layer5act;
	floatType* layer5err;
	floatType* layer6act;
	floatType* layer6err;
	floatType* layer7act;
	floatType* layer7err;
	floatType* layer8act;
	floatType* layer8err;

	// used for updating parameters
	floatType* delta_weight0;
	floatType* delta_weight1;
	floatType* delta_weight2;
	floatType* delta_weight3;
	floatType* delta_weight4;
	floatType* delta_weight5;
	floatType* delta_weight6;
	floatType* delta_weight7;

	floatType* delta_bias0;
	floatType* delta_bias1;
	floatType* delta_bias2;
	floatType* delta_bias3;
	floatType* delta_bias4;
	floatType* delta_bias5;
	floatType* delta_bias6;
	floatType* delta_bias7;

	// error vector
	floatType* error;

public:
	// data object
	dataProvider* dataprovider;

	autoencoder();
	~autoencoder();
	
	// forward propagation
	virtual void fprop();
	// back propagation
	virtual void bprop();
	// update the parameters of the network
	virtual void update();

	// the training function
	virtual void train();
};

class autoencoder_GPU : public autoencoder{
protected:
	// network parameters
	cl_mem d_weight0;
	cl_mem d_weight1;
	cl_mem d_weight2;
	cl_mem d_weight3;
	cl_mem d_weight4;
	cl_mem d_weight5;
	cl_mem d_weight6;
	cl_mem d_weight7;

	cl_mem d_bias0;
	cl_mem d_bias1;
	cl_mem d_bias2;
	cl_mem d_bias3;
	cl_mem d_bias4;
	cl_mem d_bias5;
	cl_mem d_bias6;
	cl_mem d_bias7;

	// layers of the autoencoder network, both activations and errors
	cl_mem d_layer0act;
	cl_mem d_layer0err;
	cl_mem d_layer1act;
	cl_mem d_layer1err;
	cl_mem d_layer2act;
	cl_mem d_layer2err;
	cl_mem d_layer3act;
	cl_mem d_layer3err;
	cl_mem d_layer4act;
	cl_mem d_layer4err;
	cl_mem d_layer4state;
	cl_mem d_layer5act;
	cl_mem d_layer5err;
	cl_mem d_layer6act;
	cl_mem d_layer6err;
	cl_mem d_layer7act;
	cl_mem d_layer7err;
	cl_mem d_layer8act;
	cl_mem d_layer8err;

	// used for updating parameters
	cl_mem d_delta_weight0;
	cl_mem d_delta_weight1;
	cl_mem d_delta_weight2;
	cl_mem d_delta_weight3;
	cl_mem d_delta_weight4;
	cl_mem d_delta_weight5;
	cl_mem d_delta_weight6;
	cl_mem d_delta_weight7;

	cl_mem d_delta_bias0;
	cl_mem d_delta_bias1;
	cl_mem d_delta_bias2;
	cl_mem d_delta_bias3;
	cl_mem d_delta_bias4;
	cl_mem d_delta_bias5;
	cl_mem d_delta_bias6;
	cl_mem d_delta_bias7;

	// error vector
	cl_mem d_error;

	// OpenCL kernels
	cl_kernel squareError;
	cl_kernel sigmoid;
	cl_kernel addBias;
	cl_kernel addBias_noreset;
	cl_kernel sumBatch;
	cl_kernel add;
	cl_kernel getStates;
	cl_kernel updateWeights;
	cl_kernel updateBias;
	cl_kernel randNum;
	cl_kernel randn;
	cl_kernel reset;
	cl_kernel rounding;
	cl_kernel subtract;
	cl_kernel deriv;
	cl_kernel updateAE;

public:
	// OpenCL environment
	CL_ENV gpu_env;

	// data object
	dataProvider_GPU* dataprovider;

	autoencoder_GPU();
	~autoencoder_GPU();
	
	// forward propagation
	void fprop();
	// back propagation
	void bprop();
	// update the parameters of the network
	void update();

	// the training function
	void train();
};

#endif
