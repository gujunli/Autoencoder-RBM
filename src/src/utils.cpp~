#include "utils.h"
#include "kat.h"
#include<cstring>
#include<ctime>

/*
 * Clear the buffer a, which contains n float point entries.
 * Set all entries in a to zero.
*/
void reset(floatType* a, unsigned int n){
	memset(a, 0, n * sizeof(floatType));
	return;
}

void gpu_reset(CL_ENV gpu_env, cl_kernel ker_reset, cl_mem a, unsigned n, cl_event* event){
	clSetKernelArg(ker_reset, 0, sizeof(cl_mem), (void*)&a);
	clSetKernelArg(ker_reset, 1, sizeof(cl_uint), (void*)&n);
	size_t globalws[1] = {n};
	gpu_env.status = clEnqueueNDRangeKernel(gpu_env.queue, ker_reset, 1, NULL, globalws, NULL, 0, NULL, NULL);
}

/*
 * Set elements in the input vector with uniform random numbers in the range [inf sup]
 * The input argument a[] contains the results after calling this function. The number
 * of elements in the a[.] is denoted by n.
*/ 
void randomInit(floatType* a, unsigned int n, floatType inf, floatType sup){
	srand(time(NULL));

	for(int i = 0; i < n; i++)
		a[i] = (rand() / (floatType)RAND_MAX) * (sup - inf) + inf;
	return;
}

void gpu_randomInit(CL_ENV gpu_env, cl_kernel ker_rand, cl_mem a, unsigned int n, floatType inf, floatType sup, cl_event* event){

	gpu_random(gpu_env, ker_rand, a, n, inf, sup, NULL); // ker_rand looks better
}

void gpu_random(CL_ENV gpu_env, cl_kernel ker_rand, cl_mem a, unsigned int n, floatType inf, floatType sup, cl_event* event){
	static unsigned c = 0;
	unsigned nrounds = 20;
	array4x32	rndctr4;
	rndctr4.v[0] = rndctr4.v[1] = rndctr4.v[2] = rndctr4.v[3] = c++;
	cl_uint size = n / 4;

	clSetKernelArg(ker_rand, 0, sizeof(cl_mem), 	(void*)&a);
    	clSetKernelArg(ker_rand, 1, sizeof(array4x32),	(void*)&rndctr4);
	clSetKernelArg(ker_rand, 2, sizeof(floatType),	(void*)&inf);
	clSetKernelArg(ker_rand, 3, sizeof(floatType),	(void*)&sup);
    	clSetKernelArg(ker_rand, 4, sizeof(cl_uint),	(void*)&nrounds);
    	clSetKernelArg(ker_rand, 5, sizeof(cl_uint),	(void*)&size);
	size_t globalws[1] = {size};
	gpu_env.status = clEnqueueNDRangeKernel(gpu_env.queue, ker_rand, 1, NULL, globalws, NULL, 0, NULL, NULL);
}


/*
 * Set elements in the input vector with random numbers of a normal distribution
 * N ( miu=E, std=V ). The input argument a[] contains the results after calling 
 * this function. The number of elements in the a[.] is denoted by n.
*/ 
void gaussInit(floatType* a, unsigned int n, floatType E, floatType V){
	srand(time(NULL));
	
	for(int i = 0; i < n; i++)
		a[i] = gaussRand(E, V);
	return;
}


floatType gaussRand(floatType E, floatType V){
    static double V1, V2, S;
    static int phase = 0;
    floatType X;
     
    if ( phase == 0 ) {

        do {
            double U1 = (double)rand() / RAND_MAX;
            double U2 = (double)rand() / RAND_MAX;
             
            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while(S >= 1 || S == 0);
         
        X = V1 * sqrt(-2 * log(S) / S);
    } else
        X = V2 * sqrt(-2 * log(S) / S);
         
    phase = 1 - phase;
 
    return X * V + E;
}

void gpu_gaussInit(CL_ENV gpu_env, cl_kernel ker_randn, cl_mem a, unsigned int n, floatType E, floatType V, cl_event* event){

	static unsigned c = 0;
	unsigned nrounds = 20;
	array4x32	rndctr4;
	rndctr4.v[0] = rndctr4.v[1] = rndctr4.v[2] = rndctr4.v[3] = c++;
	cl_uint size = n / 4;

	clSetKernelArg(ker_randn, 0, sizeof(cl_mem), 	(void*)&a);
    	clSetKernelArg(ker_randn, 1, sizeof(array4x32),	(void*)&rndctr4);
	clSetKernelArg(ker_randn, 2, sizeof(floatType),	(void*)&E);
	clSetKernelArg(ker_randn, 3, sizeof(floatType),	(void*)&V);
    	clSetKernelArg(ker_randn, 4, sizeof(cl_uint),	(void*)&nrounds);
    	clSetKernelArg(ker_randn, 5, sizeof(cl_uint),	(void*)&size);
	size_t globalws[1] = {size};
	gpu_env.status = clEnqueueNDRangeKernel(gpu_env.queue, ker_randn, 1, NULL, globalws, NULL, 0, NULL, NULL);
	
}

/*
 * c[.] += (b[.]-a[.])*(b[.]-a[.]) from 0 to n-1
*/
void gpu_squareError(CL_ENV gpu_env, cl_kernel ker_quad, cl_mem a, cl_mem b, cl_mem c, unsigned int n)
{
	clSetKernelArg(ker_quad, 0, sizeof(cl_mem), (void*)&a);
	clSetKernelArg(ker_quad, 1, sizeof(cl_mem), (void*)&b);
	clSetKernelArg(ker_quad, 2, sizeof(cl_mem), (void*)&c);
	clSetKernelArg(ker_quad, 3, sizeof(cl_uint), (void*)&n);

	size_t globalws[1] = {n};

	gpu_env.status = clEnqueueNDRangeKernel(gpu_env.queue, ker_quad, 1, NULL, globalws, NULL, 0, NULL, NULL);
}



/*
 * Return the results of sigmoid function with the values in the input vector a[.].
 * The input argument a[.] contains the input values before the calculation and the 
 * results after calling this function. The argument n is the number of elements in
 * the input argument a[.].
*/
void sigmoid(floatType* a, unsigned int n){
	for(int i = 0; i < n; i++){
		a[i] = 1 / (1 + exp(-a[i]));
	}
	return;
}

void gpu_sigmoid(CL_ENV gpu_env, cl_kernel ker_sigmoid, cl_mem a, unsigned int n, cl_event* event){
	clSetKernelArg(ker_sigmoid, 0, sizeof(cl_mem), (void*)&a);
	clSetKernelArg(ker_sigmoid, 1, sizeof(unsigned int), (void*)&n);
	size_t globalws[1] = {n};
	gpu_env.status = clEnqueueNDRangeKernel(gpu_env.queue, ker_sigmoid, 1, NULL, globalws, NULL, 0, NULL, event);
}

/*
 * This function resets prob and adds bias to prob: bias is a layerSize-dim vector 
 * and prob is an nVectorPerBatch x layerSize matrix. The addition simply replicate
 * the bias nVectorPerBatch times to form the prob matrix.
*/
void addBias(floatType* prob, floatType* bias, unsigned int layerSize, unsigned int nVectorPerBatch){
	for(int i = 0; i < layerSize; i++){
		floatType t = bias[i];
		for(int j = 0; j < nVectorPerBatch; j++){
			prob[j * layerSize + i] = t;
		}
	}
	return;
}

void gpu_addBias(CL_ENV gpu_env, cl_kernel ker_bias, cl_mem prob, cl_mem bias, unsigned int layerSize, unsigned int nVectorPerBatch, cl_event* event){
	clSetKernelArg(ker_bias, 0, sizeof(cl_mem), (void*)&prob);
	clSetKernelArg(ker_bias, 1, sizeof(cl_mem), (void*)&bias);
	clSetKernelArg(ker_bias, 2, sizeof(unsigned int), (void*)&layerSize);
	clSetKernelArg(ker_bias, 3, sizeof(unsigned int), (void*)&nVectorPerBatch);
	size_t globalws[1] = {layerSize * nVectorPerBatch};
	gpu_env.status = clEnqueueNDRangeKernel(gpu_env.queue, ker_bias, 1, NULL, globalws, NULL, 0, NULL, event);
}

/*
 * prob is an nVectorPerBatch x layerSize matrix and sum is a layerSize-dim vector.
 * This function sums each row of prob and stores it in sum
*/

void sumBatch(floatType* prob, floatType* sum, unsigned int layerSize, unsigned nVectorPerBatch){
	for(int i = 0; i < layerSize; i++){
		floatType t = 0.0;
		for(int j = 0; j < nVectorPerBatch; j++){
			t += prob[j * layerSize + i];
		}
		sum[i] = t;
	}
	return;
}

void gpu_sumBatch(CL_ENV gpu_env, cl_kernel ker_sumbatch, cl_mem prob, cl_mem bias, unsigned int layerSize, unsigned int nVectorPerBatch, cl_event* event){
	clSetKernelArg(ker_sumbatch, 0, sizeof(cl_mem), (void*)&prob);
	clSetKernelArg(ker_sumbatch, 1, sizeof(cl_mem), (void*)&bias);
	clSetKernelArg(ker_sumbatch, 2, sizeof(unsigned int), (void*)&layerSize);
	clSetKernelArg(ker_sumbatch, 3, sizeof(unsigned int), (void*)&nVectorPerBatch);
	size_t globalws[1] = {layerSize};
	gpu_env.status = clEnqueueNDRangeKernel(gpu_env.queue, ker_sumbatch, 1, NULL, globalws, NULL, 0, NULL, event);
}

/*
 * vector a, b
 * do a += b
*/
void gpu_add(CL_ENV gpu_env, cl_kernel ker_add, cl_mem a, cl_mem b, unsigned int n, cl_event* event){
	clSetKernelArg(ker_add, 0, sizeof(cl_mem), (void*)&a);
	clSetKernelArg(ker_add, 1, sizeof(cl_mem), (void*)&b);
	clSetKernelArg(ker_add, 2, sizeof(unsigned int), (void*)&n);
	size_t globalws[1] = {n};
	gpu_env.status = clEnqueueNDRangeKernel(gpu_env.queue, ker_add, 1, NULL, globalws, NULL, 0, NULL, event);
}

/*
 * vector a, b
 * do a = b - c
*/

void gpu_subtract(CL_ENV gpu_env, cl_kernel ker_sub, cl_mem a, cl_mem b, cl_mem c, unsigned int n, cl_event* event){

	clSetKernelArg(ker_sub, 0, sizeof(cl_mem), (void*)&a);
	clSetKernelArg(ker_sub, 1, sizeof(cl_mem), (void*)&b);
	clSetKernelArg(ker_sub, 2, sizeof(cl_mem), (void*)&c);
	clSetKernelArg(ker_sub, 3, sizeof(unsigned int), (void*)&n);
	size_t globalws[1] = {n};
	gpu_env.status = clEnqueueNDRangeKernel(gpu_env.queue, ker_sub, 1, NULL, globalws, NULL, 0, NULL, event);
}

/*
 * Sampling from a Bernoulli ditribution
 * P(states[i] = 1) = prob[i]
 * P(states[i] = 0) = 1 - prob[i]
*/
void gpu_getStates(CL_ENV gpu_env, cl_kernel kern_getstate, cl_mem states, cl_mem prob, unsigned int n, cl_event* event){
	clSetKernelArg(kern_getstate, 0, sizeof(cl_mem), (void*)&states);
	clSetKernelArg(kern_getstate, 1, sizeof(cl_mem), (void*)&prob);
	clSetKernelArg(kern_getstate, 2, sizeof(unsigned int), (void*)&n);
	size_t globalws[1] = {n};
	gpu_env.status = clEnqueueNDRangeKernel(gpu_env.queue, kern_getstate, 1, NULL, globalws, NULL, 0, NULL, event);
}

/*
 * states[i] = (prob[i] > 0.5) ? 1.0 : 0.0
*/
void gpu_rounding(CL_ENV gpu_env, cl_kernel ker_round, cl_mem states, cl_mem prob, unsigned int n, cl_event* event){
	clSetKernelArg(ker_round, 0, sizeof(cl_mem), (void*)&states);
	clSetKernelArg(ker_round, 1, sizeof(cl_mem), (void*)&prob);
	clSetKernelArg(ker_round, 2, sizeof(unsigned int), (void*)&n);
	size_t globalws[1] = {n};
	gpu_env.status = clEnqueueNDRangeKernel(gpu_env.queue, ker_round, 1, NULL, globalws, NULL, 0, NULL, event);
}

/*
 * update weights in contrastive divergence training method
*/
void gpu_updateWeights(CL_ENV gpu_env, cl_kernel ker_upWeight, cl_mem weights, cl_mem delta_weights, cl_mem posProds, cl_mem negProds, floatType momentum, floatType eps_w, floatType weightCost, unsigned int nVisLayerSize, unsigned int nHidLayerSize, unsigned int nVectorPerBatch, cl_event* event){
	clSetKernelArg(ker_upWeight, 0, sizeof(cl_mem), (void*)&weights);
	clSetKernelArg(ker_upWeight, 1, sizeof(cl_mem), (void*)&delta_weights);
	clSetKernelArg(ker_upWeight, 2, sizeof(cl_mem), (void*)&posProds);
	clSetKernelArg(ker_upWeight, 3, sizeof(cl_mem), (void*)&negProds);
	clSetKernelArg(ker_upWeight, 4, sizeof(floatType), (void*)&momentum);
	clSetKernelArg(ker_upWeight, 5, sizeof(floatType), (void*)&eps_w);
	clSetKernelArg(ker_upWeight, 6, sizeof(floatType), (void*)&weightCost);
	clSetKernelArg(ker_upWeight, 7, sizeof(unsigned int), (void*)&nVectorPerBatch);
	size_t globalws[1] = {nVisLayerSize * nHidLayerSize};
	gpu_env.status = clEnqueueNDRangeKernel(gpu_env.queue, ker_upWeight, 1, NULL, globalws, NULL, 0, NULL, event);
}

/*
 * update biases in contrastive divergence training method
*/

void gpu_updateBias(CL_ENV gpu_env, cl_kernel ker_upBias, cl_mem bias, cl_mem delta_bias, cl_mem posAct, cl_mem negAct, floatType momentum, floatType eps_b, unsigned int nLayerSize, unsigned int nVectorPerBatch, cl_event* event){
	clSetKernelArg(ker_upBias, 0, sizeof(cl_mem), (void*)&bias);
	clSetKernelArg(ker_upBias, 1, sizeof(cl_mem), (void*)&delta_bias);
	clSetKernelArg(ker_upBias, 2, sizeof(cl_mem), (void*)&posAct);
	clSetKernelArg(ker_upBias, 3, sizeof(cl_mem), (void*)&negAct);
	clSetKernelArg(ker_upBias, 4, sizeof(floatType), (void*)&momentum);
	clSetKernelArg(ker_upBias, 5, sizeof(floatType), (void*)&eps_b);
	clSetKernelArg(ker_upBias, 6, sizeof(unsigned int), (void*)&nVectorPerBatch);
	size_t globalws[1] = {nLayerSize};
	gpu_env.status = clEnqueueNDRangeKernel(gpu_env.queue, ker_upBias, 1, NULL, globalws, NULL, 0, NULL, event);
}

/*
 * calculate the derivatives in the back propagation algorithm
*/
void gpu_deriv(CL_ENV gpu_env, cl_kernel ker_deriv, cl_mem err, cl_mem act, unsigned int n, cl_event* event){
	clSetKernelArg(ker_deriv, 0, sizeof(cl_mem), (void*)&err);
	clSetKernelArg(ker_deriv, 1, sizeof(cl_mem), (void*)&act);
	clSetKernelArg(ker_deriv, 2, sizeof(unsigned int), (void*)&n);
	size_t globalws[1] = {n};
	gpu_env.status = clEnqueueNDRangeKernel(gpu_env.queue, ker_deriv, 1, NULL, globalws, NULL, 0, NULL, event);
}

/*
 * update weights and biases in the back propagation algorithm
*/
void gpu_update(CL_ENV gpu_env, cl_kernel kern, cl_mem weight, cl_mem delta_weight, cl_mem bias, cl_mem delta_bias, floatType eps_w, floatType eps_b, unsigned int nBottomLayerSize, unsigned int nUpperLayerSize, cl_event* event){
	clSetKernelArg(kern, 0, sizeof(cl_mem), (void*)&weight);
	clSetKernelArg(kern, 1, sizeof(cl_mem), (void*)&delta_weight);
	clSetKernelArg(kern, 2, sizeof(cl_mem), (void*)&bias);
	clSetKernelArg(kern, 3, sizeof(cl_mem), (void*)&delta_bias);
	clSetKernelArg(kern, 4, sizeof(floatType), (void*)&eps_w);
	clSetKernelArg(kern, 5, sizeof(floatType), (void*)&eps_b);
	clSetKernelArg(kern, 6, sizeof(unsigned int), (void*)&nBottomLayerSize);
	clSetKernelArg(kern, 7, sizeof(unsigned int), (void*)&nUpperLayerSize);
	size_t globalws[1] = {nBottomLayerSize * nUpperLayerSize};
	gpu_env.status = clEnqueueNDRangeKernel(gpu_env.queue, kern, 1, NULL, globalws, NULL, 0, NULL, event);
}


/*
 * This function returns the Euclidean distance between vector/matrix a and b.
 * Both a and b have a length of n. 
*/

floatType EuDist(floatType* a, floatType* b, unsigned int n){
	floatType result = 0.0;
	for(int i = 0; i < n; i++){
		result += (a[i] - b[i]) * (a[i] - b[i]);
	}
	return result;
}


/*
 * log out data as a csv file
*/
void logData(string filename, floatType* data, unsigned int Length, unsigned int stride, unsigned int nImageNum){
	ofstream fout;
	fout.open(filename.c_str(), ios_base::trunc);
	for(int i = 0; i < Length; i++){
		fout << data[i * nImageNum];
		if((i + 1) % stride == 0)
			fout << endl;
		else
			fout << ',';
	}
	fout.close();
	return;
}

/*
 * log out data as binary file
*/

void logBinaryData(string filename, floatType* data, unsigned int Length, unsigned int stride, unsigned int nImageNum){
	ofstream fout;
	fout.open(filename.c_str(), ios_base::trunc | ios_base::binary);
	fout.write((char*)data, Length * nImageNum);
	fout.close();
	return;
}

/*
 * Initialize the OpenCL GPU environment contained by a CL_ENV object cl_env.
 * cl_env is a newly defined CL_ENV object without initialization and this function
 * allocates platforms and devices, then builds programs with kernel files.
*/

void gpu_init(CL_ENV& cl_env, unsigned int deviceIndex){
	// initialize the OpenCL environment
	cl_env.platform = 0;
	cl_env.device = 0;
	cl_env.props[0] = CL_CONTEXT_PLATFORM;
	cl_env.props[1] = cl_env.props[2] = 0;
	cl_env.ctx = 0;
	cl_env.queue = 0;
	cl_env.event = NULL;
	cl_env.order = clAmdBlasColumnMajor;

	// setup OpenCL environment
	cl_env.status = clGetPlatformIDs(1, &cl_env.platform, NULL);
	if (cl_env.status != CL_SUCCESS) {
		printf( "clGetPlatformIDs() failed with %d\n", cl_env.status );
		return;
	}

	cl_device_id device[10];
	cl_uint devices_n;

	cl_env.status = clGetDeviceIDs(cl_env.platform, CL_DEVICE_TYPE_GPU, 10, device, &devices_n);
	if (cl_env.status != CL_SUCCESS) {
		printf( "clGetDeviceIDs() failed with %d\n", cl_env.status );
		return;
	}

	// print device information
	for(int i = 0; i < devices_n; i++){
		char buffer[10240];
		cl_uint buf_uint;
		cl_ulong buf_ulong;
		printf("  --Device: %d --\n", i);
		clGetDeviceInfo(device[i], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
		printf("  DEVICE_NAME = %s\n", buffer);
		clGetDeviceInfo(device[i], CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL);
		printf("  DEVICE_VENDOR = %s\n", buffer);
		clGetDeviceInfo(device[i], CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL);
		printf("  DEVICE_VERSION = %s\n", buffer);
		clGetDeviceInfo(device[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buf_uint), &buf_uint, NULL);
		printf("  DEVICE_MAX_COMPUTE_UNITS = %u\n", (unsigned int)buf_uint);
		clGetDeviceInfo(device[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buf_uint), &buf_uint, NULL);
		printf("  DEVICE_MAX_CLOCK_FREQUENCY = %u\n", (unsigned int)buf_uint);
		clGetDeviceInfo(device[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
		printf("  DEVICE_GLOBAL_MEM_SIZE = %llu\n", (unsigned long long)buf_ulong);
	}

	// use the first device listed
	cl_env.device = device[deviceIndex];

	cl_env.props[1] = (cl_context_properties)cl_env.platform;
	cl_env.ctx = clCreateContext(cl_env.props, 1, &cl_env.device, NULL, NULL, &cl_env.status);
	if (cl_env.status != CL_SUCCESS) {
		printf( "clCreateContext() failed with %d\n", cl_env.status );
		return;
	}

	cl_env.queue = clCreateCommandQueue(cl_env.ctx, cl_env.device, CL_QUEUE_PROFILING_ENABLE, &cl_env.status);
	if (cl_env.status != CL_SUCCESS) {
		printf( "clCreateCommandQueue() failed with %d\n", cl_env.status );
		clReleaseContext(cl_env.ctx);
		return;
	}

	/* Setup clAmdBlas. */
	cl_env.status = clAmdBlasSetup();
	if (cl_env.status != CL_SUCCESS) {
		printf("clAmdBlasSetup() failed with %d\n", cl_env.status);
		clReleaseCommandQueue(cl_env.queue);
		clReleaseContext(cl_env.ctx);
		return;
	}

}

/*
 * load kernel source from an OpenCL source file for runtime compiling
*/
void loadKernelSource(string filename, char* source){
	ifstream fin;
	string srt;
	fin.open(filename.c_str(), ios_base::in);
	char t[500];
	while(fin.getline(t, 500)){
		srt.append(t);
		srt.push_back('\n');
	}
	for(int i = 0; i < srt.length(); i++)
		source[i] = srt[i];
	source[srt.length()] = 0;
	return;
}
