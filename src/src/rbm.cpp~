#include <sys/time.h>
#include "rbm.h"

RBM::RBM(){
	// initialize hyper parameters
	nEpochNum = 20; // the max epoch is 50 in Hinton's Science paper, as a default value
	nBatchNum = 600;
	nVectorPerBatch = 100;

	nVisLayerSize = 784;
	nHidLayerSize = 1024;
	linear = false;

	weightCost = 0.0002;
	initialMomentum = 0.5;
	finalMomentum = 0.9;

	// allocate memory for member variables
	try{

		weights = new floatType[nHidLayerSize * nVisLayerSize];
		hidBias = new floatType[nHidLayerSize];
		visBias = new floatType[nVisLayerSize];

		delta_weights = new floatType[nHidLayerSize * nVisLayerSize];
		delta_hidBias = new floatType[nHidLayerSize];
		delta_visBias = new floatType[nVisLayerSize];

		posHidProbs = new floatType[nHidLayerSize * nVectorPerBatch];
		negHidProbs = new floatType[nHidLayerSize * nVectorPerBatch];
		posProds = new floatType[nHidLayerSize * nVisLayerSize];
		negProds = new floatType[nHidLayerSize * nVisLayerSize];
		negData = new floatType[nVisLayerSize * nVectorPerBatch];
		posHidAct = new floatType[nHidLayerSize];
		negHidAct = new floatType[nHidLayerSize];
		posVisAct = new floatType[nVisLayerSize];
		negVisAct = new floatType[nVisLayerSize];
		posHidStates = new floatType[nHidLayerSize * nVectorPerBatch];
	}
	catch (bad_alloc& ba){
		cerr << "bad allocation caught: " << ba.what() << endl;
		system("pause");
		exit(-1);
	}

	// initialize member variables
	eps_w = 0.001;
	eps_vb = 0.001;
	eps_hb = 0.001;

	gaussInit(weights, nHidLayerSize * nVisLayerSize, 0, 0.01);
	reset(hidBias, nHidLayerSize);
	reset(visBias, nVisLayerSize);
	reset(posHidProbs, nHidLayerSize * nVectorPerBatch);
	reset(negHidProbs, nHidLayerSize * nVectorPerBatch);
	reset(posProds, nHidLayerSize * nVisLayerSize);
	reset(negProds, nHidLayerSize * nVisLayerSize);
	reset(delta_weights, nHidLayerSize * nVisLayerSize);
	reset(delta_hidBias, nHidLayerSize);
	reset(delta_visBias, nVisLayerSize);
}

RBM::RBM(unsigned int vis, unsigned int hid, bool linearity, unsigned numEpoch, unsigned numBatch, unsigned nVecPerBatch, floatType wCost, floatType initMom, floatType finalMom, string layertag){
	// initialize hyper parameters
	nEpochNum = numEpoch; // the max epoch is 50 in Hinton's Science paper, as a default value
	nBatchNum = numBatch;
	nVectorPerBatch = nVecPerBatch;

	nVisLayerSize = vis;
	nHidLayerSize = hid;
	linear = linearity;

	weightCost = wCost;
	initialMomentum = initMom;
	finalMomentum = finalMom;

	dataTag	= "../data/";
	logTag	= "../log/";
	dataTag.append(layertag);
	logTag.append(layertag);

	// allocate memory for member variables
	try{

		weights = new floatType[nHidLayerSize * nVisLayerSize];
		hidBias = new floatType[nHidLayerSize];
		visBias = new floatType[nVisLayerSize];

		delta_weights = new floatType[nHidLayerSize * nVisLayerSize];
		delta_hidBias = new floatType[nHidLayerSize];
		delta_visBias = new floatType[nVisLayerSize];

		posHidProbs = new floatType[nHidLayerSize * nVectorPerBatch];
		negHidProbs = new floatType[nHidLayerSize * nVectorPerBatch];
		posProds = new floatType[nHidLayerSize * nVisLayerSize];
		negProds = new floatType[nHidLayerSize * nVisLayerSize];
		negData = new floatType[nVisLayerSize * nVectorPerBatch];
		posHidAct = new floatType[nHidLayerSize];
		negHidAct = new floatType[nHidLayerSize];
		posVisAct = new floatType[nVisLayerSize];
		negVisAct = new floatType[nVisLayerSize];
		posHidStates = new floatType[nHidLayerSize * nVectorPerBatch];
		error = new floatType[nVisLayerSize * nVectorPerBatch];
	}
	catch (bad_alloc& ba){
		cerr << "bad allocation caught: " << ba.what() << endl;
		system("pause");
		exit(-1);
	}

	// initialize member variables
	if(linear){
		eps_w = 0.001;
		eps_vb = 0.001;
		eps_hb = 0.001;
	}
	else{
		eps_w = 0.01;
		eps_vb = 0.01;
		eps_hb = 0.01;
	}
	
	if(linear){
		gaussInit(weights, nHidLayerSize * nVisLayerSize, 0, 0.1);
	}
	else{
		gaussInit(weights, nHidLayerSize * nVisLayerSize, 0, 0.01);
	}
	reset(hidBias, nHidLayerSize);
	reset(visBias, nVisLayerSize);
	reset(posHidProbs, nHidLayerSize * nVectorPerBatch);
	reset(negHidProbs, nHidLayerSize * nVectorPerBatch);
	reset(posProds, nHidLayerSize * nVisLayerSize);
	reset(negProds, nHidLayerSize * nVisLayerSize);
	reset(delta_weights, nHidLayerSize * nVisLayerSize);
	reset(delta_hidBias, nHidLayerSize);
	reset(delta_visBias, nVisLayerSize);
	reset(error, nVisLayerSize * nVectorPerBatch);
}

void RBM::setInputData(vector<floatType*> trainData){
	batchData = trainData;
	return;
}

void RBM::posProp(){
	addBias(posHidProbs, hidBias, nHidLayerSize, nVectorPerBatch);
	sgemm('n', 'n', nHidLayerSize, nVectorPerBatch, nVisLayerSize, 1.0, weights, nHidLayerSize, posData, nVisLayerSize, 1.0, posHidProbs, nHidLayerSize);
	
	if(!linear){
		sigmoid(posHidProbs, nHidLayerSize * nVectorPerBatch);
	}

	sgemm('n', 't', nHidLayerSize, nVisLayerSize, nVectorPerBatch, 1.0, posHidProbs, nHidLayerSize, posData, nVisLayerSize, 0.0, posProds, nHidLayerSize);
	sumBatch(posHidProbs, posHidAct, nHidLayerSize, nVectorPerBatch);
	sumBatch(posData, posVisAct, nVisLayerSize, nVectorPerBatch);

	return;
}

void RBM::generateStates(){
	if(linear){
		gaussInit(posHidStates, nHidLayerSize * nVectorPerBatch, 0.0, 1.0);
		for(int i = 0; i < nHidLayerSize * nVectorPerBatch; i++){
			posHidStates[i] += posHidProbs[i];
		}
	}
	else{
		randomInit(posHidStates, nHidLayerSize * nVectorPerBatch, 0.0, 1.0);
		for(int i = 0; i < nHidLayerSize * nVectorPerBatch; i++){
			posHidStates[i] = (posHidProbs[i] > posHidStates[i]) ? 1.0 : 0.0;
		}
	}
	return;
}

void RBM::negProp(){
	addBias(negData, visBias, nVisLayerSize, nVectorPerBatch);
	sgemm('t', 'n', nVisLayerSize, nVectorPerBatch, nHidLayerSize, 1.0, weights, nHidLayerSize, posHidStates, nHidLayerSize, 1.0, negData, nVisLayerSize);
	
	sigmoid(negData, nVisLayerSize * nVectorPerBatch);

	addBias(negHidProbs, hidBias, nHidLayerSize, nVectorPerBatch);
	sgemm('n', 'n', nHidLayerSize, nVectorPerBatch, nVisLayerSize, 1.0, weights, nHidLayerSize, negData, nVisLayerSize, 1.0, negHidProbs, nHidLayerSize);
	if(!linear){
		sigmoid(negHidProbs, nHidLayerSize * nVectorPerBatch);
	}

	sgemm('n', 't', nHidLayerSize, nVisLayerSize, nVectorPerBatch, 1.0, negHidProbs, nHidLayerSize, negData, nVisLayerSize, 0.0, negProds, nHidLayerSize);
	sumBatch(negHidProbs, negHidAct, nHidLayerSize, nVectorPerBatch);
	sumBatch(negData, negVisAct, nVisLayerSize, nVectorPerBatch);

	return;
}

void RBM::update(){
	for(int i = 0; i < nVisLayerSize * nHidLayerSize; i++){
		delta_weights[i] = momentum * delta_weights[i] + eps_w * ((posProds[i] - negProds[i]) / nVectorPerBatch - weightCost * weights[i]);
		weights[i] += delta_weights[i];
	}
	for(int i = 0; i < nVisLayerSize; i++){
		delta_visBias[i] = momentum * delta_visBias[i] + (eps_vb / nVectorPerBatch) * (posVisAct[i] - negVisAct[i]);
		visBias[i] += delta_visBias[i];
	}
	for(int i = 0; i < nHidLayerSize; i++){
		delta_hidBias[i] = momentum * delta_hidBias[i] + (eps_hb / nVectorPerBatch) * (posHidAct[i] - negHidAct[i]);
		hidBias[i] += delta_hidBias[i];
	}
	
	return;
}

void RBM::train(){

	for(int epoch = 0; epoch < nEpochNum; epoch++){
		dataprovider->reset();
		double errsum = 0.0;
		printf("Epoch %d\n", epoch + 1);
		momentum = (epoch < 5) ? initialMomentum : finalMomentum;
		for(int batch = 0; batch < nBatchNum; batch++){
			printf("Epoch %d Batch %d\n", epoch + 1, batch + 1);
			posData = dataprovider->getNextBatch();
			posProp();
			generateStates();
			negProp();
			errsum += EuDist(posData, negData, nVisLayerSize * nVectorPerBatch);
			update();
		}
		printf("Epoch %d Error %f\n", epoch + 1, errsum);

		ofstream fout;
		fout.open("../log/errorLog.txt", ios_base::app);
		struct timeval now;
		gettimeofday(&now, NULL);
		fout << now.tv_sec << ',' << errsum << endl;
		fout.close();

		string logWeightFileName = logTag.append("Weight");
		generateFileName(&logWeightFileName, epoch, 3);
		logData(logWeightFileName, weights, nVisLayerSize * nHidLayerSize, nHidLayerSize, 1);
		string logHidBiasFileName = logTag.append("HidBias");
		generateFileName(&logHidBiasFileName, epoch, 3);
		logData(logHidBiasFileName, hidBias, nHidLayerSize, nHidLayerSize, 1);
		string logVisBiasFileName = logTag.append("VisBias");
		generateFileName(&logVisBiasFileName, epoch, 3);
		logData(logVisBiasFileName, visBias, nVisLayerSize, nVisLayerSize, 1);

	}

	return;
}

RBM::~RBM(){
	delete[] weights; 
	delete[] hidBias; 
	delete[] visBias; 
 		  
	delete[] delta_weights; 
	delete[] delta_hidBias; 
	delete[] delta_visBias; 
		  
	delete[] posData;
	delete[] posHidProbs; 
	delete[] negHidProbs; 
	delete[] posProds; 
	delete[] negProds;  
	delete[] negData; 
	delete[] posHidAct; 
	delete[] posVisAct; 
	delete[] negHidAct; 
	delete[] negVisAct; 
	delete[] posHidStates;
}		  
