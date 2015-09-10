#include <sys/time.h>
#include "autoencoder.h"

autoencoder::autoencoder(){

	nEpochNum = 5; // total number of epoches
	nBatchNum = 68000 * 30 * 81 / 128; // total number of mini-batches
	nVectorPerBatch = 128; // the number of input vectors in each min-batch

	// The size of each layer
	nLayerSize0 = 336; 
	nLayerSize1 = 1024;
	nLayerSize2 = 512;
	nLayerSize3 = 256;
	nLayerSize4 = 128;
	nLayerSize5 = 256;
	nLayerSize6 = 512;
	nLayerSize7 = 1024;
	nLayerSize8 = 336;

	eps_w = 0.000001; // learning rate for weights
	eps_b = 0.000001; // learning rate for biases

	weight0 = new floatType[nLayerSize0 * nLayerSize1];
	weight1 = new floatType[nLayerSize1 * nLayerSize2];
	weight2 = new floatType[nLayerSize2 * nLayerSize3];
	weight3 = new floatType[nLayerSize3 * nLayerSize4];
	weight4 = new floatType[nLayerSize4 * nLayerSize5];
	weight5 = new floatType[nLayerSize5 * nLayerSize6];
	weight6 = new floatType[nLayerSize6 * nLayerSize7];
	weight7 = new floatType[nLayerSize7 * nLayerSize8];

	bias0 = new floatType[nLayerSize1];
	bias1 = new floatType[nLayerSize2];
	bias2 = new floatType[nLayerSize3];
	bias3 = new floatType[nLayerSize4];
	bias4 = new floatType[nLayerSize5];
	bias5 = new floatType[nLayerSize6];
	bias6 = new floatType[nLayerSize7];
	bias7 = new floatType[nLayerSize8];

	layer0act = new floatType[nLayerSize0 * nVectorPerBatch];
	layer0err = new floatType[nLayerSize0 * nVectorPerBatch];
	layer1act = new floatType[nLayerSize1 * nVectorPerBatch];
	layer1err = new floatType[nLayerSize1 * nVectorPerBatch];
	layer2act = new floatType[nLayerSize2 * nVectorPerBatch];
	layer2err = new floatType[nLayerSize2 * nVectorPerBatch];
	layer3act = new floatType[nLayerSize3 * nVectorPerBatch];
	layer3err = new floatType[nLayerSize3 * nVectorPerBatch];
	layer4act = new floatType[nLayerSize4 * nVectorPerBatch];
	layer4err = new floatType[nLayerSize4 * nVectorPerBatch];
	layer4state = new floatType[nLayerSize4 * nVectorPerBatch];
	layer5act = new floatType[nLayerSize5 * nVectorPerBatch];
	layer5err = new floatType[nLayerSize5 * nVectorPerBatch];
	layer6act = new floatType[nLayerSize6 * nVectorPerBatch];
	layer6err = new floatType[nLayerSize6 * nVectorPerBatch];
	layer7act = new floatType[nLayerSize7 * nVectorPerBatch];
	layer7err = new floatType[nLayerSize7 * nVectorPerBatch];
	layer8act = new floatType[nLayerSize8 * nVectorPerBatch];
	layer8err = new floatType[nLayerSize8 * nVectorPerBatch];

	delta_weight0 = new floatType[nLayerSize0 * nLayerSize1];
	delta_weight1 = new floatType[nLayerSize1 * nLayerSize2];
	delta_weight2 = new floatType[nLayerSize2 * nLayerSize3];
	delta_weight3 = new floatType[nLayerSize3 * nLayerSize4];
	delta_weight4 = new floatType[nLayerSize4 * nLayerSize5];
	delta_weight5 = new floatType[nLayerSize5 * nLayerSize6];
	delta_weight6 = new floatType[nLayerSize6 * nLayerSize7];
	delta_weight7 = new floatType[nLayerSize7 * nLayerSize8];

	delta_bias0 = new floatType[nLayerSize1];
	delta_bias1 = new floatType[nLayerSize2];
	delta_bias2 = new floatType[nLayerSize3];
	delta_bias3 = new floatType[nLayerSize4];
	delta_bias4 = new floatType[nLayerSize5];
	delta_bias5 = new floatType[nLayerSize6];
	delta_bias6 = new floatType[nLayerSize7];
	delta_bias7 = new floatType[nLayerSize8];

	// error vector
	error = new floatType[nLayerSize0 * nVectorPerBatch];

	// initialize weights and biases with RBM results
	ifstream fin;
	// the first RBM
	fin.open("../data/firstWeight.dat", ios_base::binary);
	fin.read((char*)weight0, nLayerSize0 * nLayerSize1 * sizeof(floatType));
	fin.close();
	for(unsigned int i = 0; i < nLayerSize0; i++){
		for(unsigned int j = 0; j < nLayerSize1; j++){
			weight7[j * nLayerSize0 + i] = weight0[i * nLayerSize1 + j];
		}
	}
	fin.open("../data/firstHidBias.dat", ios_base::binary);
	fin.read((char*)bias0, nLayerSize1 * sizeof(floatType));
	fin.close();
	fin.open("../data/firstVisBias.dat", ios_base::binary);
	fin.read((char*)bias7, nLayerSize8 * sizeof(floatType));
	fin.close();

	// the second RBM
	fin.open("../data/secondWeight.dat", ios_base::binary);
	fin.read((char*)weight1, nLayerSize1 * nLayerSize2 * sizeof(floatType));
	fin.close();
	for(unsigned int i = 0; i < nLayerSize1; i++){
		for(unsigned int j = 0; j < nLayerSize2; j++){
			weight6[j * nLayerSize1 + i] = weight1[i * nLayerSize2 + j];
		}
	}
	fin.open("../data/secondHidBias.dat", ios_base::binary);
	fin.read((char*)bias1, nLayerSize2 * sizeof(floatType));
	fin.close();
	fin.open("../data/secondVisBias.dat", ios_base::binary);
	fin.read((char*)bias6, nLayerSize7 * sizeof(floatType));
	fin.close();

	// the third RBM
	fin.open("../data/thirdWeight.dat", ios_base::binary);
	fin.read((char*)weight2, nLayerSize2 * nLayerSize3 * sizeof(floatType));
	fin.close();
	for(unsigned int i = 0; i < nLayerSize2; i++){
		for(unsigned int j = 0; j < nLayerSize3; j++){
			weight5[j * nLayerSize2 + i] = weight2[i * nLayerSize3 + j];
		}
	}
	fin.open("../data/thirdHidBias.dat", ios_base::binary);
	fin.read((char*)bias2, nLayerSize3 * sizeof(floatType));
	fin.close();
	fin.open("../data/thirdVisBias.dat", ios_base::binary);
	fin.read((char*)bias5, nLayerSize6 * sizeof(floatType));
	fin.close();

	// the fourth RBM
	fin.open("../data/fourthWeight.dat", ios_base::binary);
	fin.read((char*)weight3, nLayerSize3 * nLayerSize4 * sizeof(floatType));
	fin.close();
	for(unsigned int i = 0; i < nLayerSize3; i++){
		for(unsigned int j = 0; j < nLayerSize4; j++){
			weight4[j * nLayerSize3 + i] = weight3[i * nLayerSize4 + j];
		}
	}
	fin.open("../data/fourthHidBias.dat", ios_base::binary);
	fin.read((char*)bias3, nLayerSize4 * sizeof(floatType));
	fin.close();
	fin.open("../data/fourthVisBias.dat", ios_base::binary);
	fin.read((char*)bias4, nLayerSize5 * sizeof(floatType));
	fin.close();

}

void autoencoder::fprop(){
	// 1st layer to 2nd layer
	addBias(layer1act, bias0, nLayerSize1, nVectorPerBatch);
	sgemm('n', 'n', nLayerSize1, nVectorPerBatch, nLayerSize0, 1.0, weight0, nLayerSize1, layer0act, nLayerSize0, 1.0, layer1act, nLayerSize1);
	sigmoid(layer1act, nLayerSize1 * nVectorPerBatch);

	// 2nd layer to 3rd layer
	addBias(layer2act, bias1, nLayerSize2, nVectorPerBatch);
	sgemm('n', 'n', nLayerSize2, nVectorPerBatch, nLayerSize1, 1.0, weight1, nLayerSize2, layer1act, nLayerSize1, 1.0, layer2act, nLayerSize2);
	sigmoid(layer2act, nLayerSize2 * nVectorPerBatch);

	// 3rd layer to 4th layer
	addBias(layer3act, bias2, nLayerSize3, nVectorPerBatch);
	sgemm('n', 'n', nLayerSize3, nVectorPerBatch, nLayerSize2, 1.0, weight2, nLayerSize3, layer2act, nLayerSize2, 1.0, layer3act, nLayerSize3);
	sigmoid(layer3act, nLayerSize3 * nVectorPerBatch);

	// 4th layer to 5th layer
	addBias(layer4act, bias3, nLayerSize4, nVectorPerBatch);
	sgemm('n', 'n', nLayerSize4, nVectorPerBatch, nLayerSize3, 1.0, weight3, nLayerSize4, layer3act, nLayerSize3, 1.0, layer4act, nLayerSize4);
	sigmoid(layer4act, nLayerSize4 * nVectorPerBatch);

	// rounding for Layer 4
	for(int i = 0; i < nLayerSize4 * nVectorPerBatch; i++){
		layer4state[i] = (layer4act[i] > 0.5) ? 1.0 : 0.0;
	}

	// 5th layer to 6th layer
	addBias(layer5act, bias4, nLayerSize5, nVectorPerBatch);
	sgemm('n', 'n', nLayerSize5, nVectorPerBatch, nLayerSize4, 1.0, weight4, nLayerSize5, layer4state, nLayerSize4, 1.0, layer5act, nLayerSize5);
	sigmoid(layer5act, nLayerSize5 * nVectorPerBatch);

	// 6th layer to 7th layer
	addBias(layer6act, bias5, nLayerSize6, nVectorPerBatch);
	sgemm('n', 'n', nLayerSize6, nVectorPerBatch, nLayerSize5, 1.0, weight5, nLayerSize6, layer5act, nLayerSize5, 1.0, layer6act, nLayerSize6);
	sigmoid(layer6act, nLayerSize6 * nVectorPerBatch);

	// 7th layer to 8th layer
	addBias(layer7act, bias6, nLayerSize7, nVectorPerBatch);
	sgemm('n', 'n', nLayerSize7, nVectorPerBatch, nLayerSize6, 1.0, weight6, nLayerSize7, layer6act, nLayerSize6, 1.0, layer7act, nLayerSize7);
	sigmoid(layer7act, nLayerSize7 * nVectorPerBatch);

	// 8th layer to 9th layer
	addBias(layer8act, bias7, nLayerSize8, nVectorPerBatch);
	sgemm('n', 'n', nLayerSize8, nVectorPerBatch, nLayerSize7, 1.0, weight7, nLayerSize8, layer7act, nLayerSize7, 1.0, layer8act, nLayerSize8);
	sigmoid(layer8act, nLayerSize8 * nVectorPerBatch);
	
	floatType cost = EuDist(layer8act, layer0act, nLayerSize0 * nVectorPerBatch);

	for(int i = 0; i < nLayerSize8 * nVectorPerBatch; i++){
		error[i] += (layer8act[i] - layer0act[i]) * (layer8act[i] - layer0act[i]);
	}
	
}

void autoencoder::bprop(){

	// compute the error vector - need normalization factor?
	for(int i = 0; i < nLayerSize0 * nVectorPerBatch; i++){
		layer8err[i] = layer8act[i] - layer0act[i];
	}
	// back propagation
	sgemm('t', 'n', nLayerSize7, nVectorPerBatch, nLayerSize8, 1.0, weight7, nLayerSize8, layer8err, nLayerSize8, 0.0, layer7err, nLayerSize7);

	// calculate derivatives
	for(int i = 0; i < nLayerSize7 * nVectorPerBatch; i++){
		layer7err[i] *= (1 - layer7act[i]) * layer7act[i];
	}

	// compute gradients for biases and weights
	sumBatch(layer7err, delta_bias7, nLayerSize7, nVectorPerBatch);
	sgemm('n', 't', nLayerSize8, nLayerSize7, nVectorPerBatch, 1.0, layer8err, nLayerSize8, layer7act, nLayerSize7, 0.0, delta_weight7, nLayerSize8);

	// 8th layer to 7th layer
	sgemm('t', 'n', nLayerSize6, nVectorPerBatch, nLayerSize7, 1.0, weight6, nLayerSize7, layer7err, nLayerSize7, 0.0, layer6err, nLayerSize6);
	for(int i = 0; i < nLayerSize6 * nVectorPerBatch; i++){
		layer6err[i] *= (1 - layer6act[i]) * layer6act[i];
	}
	sumBatch(layer6err, delta_bias6, nLayerSize6, nVectorPerBatch);
	sgemm('n', 't', nLayerSize7, nLayerSize6, nVectorPerBatch, 1.0, layer7err, nLayerSize7, layer6act, nLayerSize6, 0.0, delta_weight6, nLayerSize7);
	
	// 7th layer to 6th layer
	sgemm('t', 'n', nLayerSize5, nVectorPerBatch, nLayerSize6, 1.0, weight5, nLayerSize6, layer6err, nLayerSize6, 0.0, layer5err, nLayerSize5);
	for(int i = 0; i < nLayerSize5 * nVectorPerBatch; i++){
		layer5err[i] *= (1 - layer5act[i]) * layer5act[i];
	}
	sumBatch(layer5err, delta_bias5, nLayerSize5, nVectorPerBatch);
	sgemm('n', 't', nLayerSize6, nLayerSize5, nVectorPerBatch, 1.0, layer6err, nLayerSize6, layer5act, nLayerSize5, 0.0, delta_weight5, nLayerSize6);

	// 6th layer to 5th layer
	sgemm('t', 'n', nLayerSize4, nVectorPerBatch, nLayerSize5, 1.0, weight4, nLayerSize5, layer5err, nLayerSize5, 0.0, layer4err, nLayerSize4);
	for(int i = 0; i < nLayerSize4 * nVectorPerBatch; i++){
		layer4err[i] *= (1 - layer4act[i]) * layer4act[i];
	}
	sumBatch(layer4err, delta_bias4, nLayerSize4, nVectorPerBatch);
	sgemm('n', 't', nLayerSize5, nLayerSize4, nVectorPerBatch, 1.0, layer5err, nLayerSize5, layer4act, nLayerSize4, 0.0, delta_weight4, nLayerSize5);

	// 5th layer to 4th layer
	sgemm('t', 'n', nLayerSize3, nVectorPerBatch, nLayerSize4, 1.0, weight3, nLayerSize4, layer4err, nLayerSize4, 0.0, layer3err, nLayerSize3);
	for(int i = 0; i < nLayerSize3 * nVectorPerBatch; i++){
		layer3err[i] *= (1 - layer3act[i]) * layer3act[i];
	}
	sumBatch(layer3err, delta_bias3, nLayerSize3, nVectorPerBatch);
	sgemm('n', 't', nLayerSize4, nLayerSize3, nVectorPerBatch, 1.0, layer4err, nLayerSize4, layer3act, nLayerSize3, 0.0, delta_weight3, nLayerSize4);

	// 4th layer to 3rd layer
	sgemm('t', 'n', nLayerSize2, nVectorPerBatch, nLayerSize3, 1.0, weight2, nLayerSize3, layer3err, nLayerSize3, 0.0, layer2err, nLayerSize2);
	for(int i = 0; i < nLayerSize2 * nVectorPerBatch; i++){
		layer2err[i] *= (1 - layer2act[i]) * layer2act[i];
	}
	sumBatch(layer2err, delta_bias2, nLayerSize2, nVectorPerBatch);
	sgemm('n', 't', nLayerSize3, nLayerSize2, nVectorPerBatch, 1.0, layer3err, nLayerSize3, layer2act, nLayerSize2, 0.0, delta_weight2, nLayerSize3);

	// 3rd layer to 2nd layer
	sgemm('t', 'n', nLayerSize1, nVectorPerBatch, nLayerSize2, 1.0, weight1, nLayerSize2, layer2err, nLayerSize2, 0.0, layer1err, nLayerSize1);
	for(int i = 0; i < nLayerSize1 * nVectorPerBatch; i++){
		layer1err[i] *= (1 - layer1act[i]) * layer1act[i];
	}
	sumBatch(layer1err, delta_bias1, nLayerSize1, nVectorPerBatch);
	sgemm('n', 't', nLayerSize2, nLayerSize1, nVectorPerBatch, 1.0, layer2err, nLayerSize2, layer1act, nLayerSize1, 0.0, delta_weight1, nLayerSize2);

	// 2nd layer to 1st layer
	sgemm('t', 'n', nLayerSize0, nVectorPerBatch, nLayerSize1, 1.0, weight0, nLayerSize1, layer1err, nLayerSize1, 0.0, layer0err, nLayerSize0);
	for(int i = 0; i < nLayerSize0 * nVectorPerBatch; i++){
		layer0err[i] *= (1 - layer0act[i]) * layer0act[i];
	}
	sumBatch(layer0err, delta_bias0, nLayerSize0, nVectorPerBatch);
	sgemm('n', 't', nLayerSize1, nLayerSize0, nVectorPerBatch, 1.0, layer1err, nLayerSize1, layer0act, nLayerSize0, 0.0, delta_weight0, nLayerSize1);

}

void autoencoder::update(){
	for(int i = 0; i < nLayerSize7 * nLayerSize8; i++){
		weight7[i] -= eps_w * delta_weight7[i];
	}
	for(int i = 0; i < nLayerSize8; i++){
		bias7[i] -= eps_b * delta_bias7[i];
	}

	for(int i = 0; i < nLayerSize6 * nLayerSize7; i++){
		weight6[i] -= eps_w * delta_weight6[i];
	}
	for(int i = 0; i < nLayerSize7; i++){
		bias6[i] -= eps_b * delta_bias6[i];
	}

	for(int i = 0; i < nLayerSize5 * nLayerSize6; i++){
		weight5[i] -= eps_w * delta_weight5[i];
	}
	for(int i = 0; i < nLayerSize6; i++){
		bias5[i] -= eps_b * delta_bias5[i];
	}

	for(int i = 0; i < nLayerSize4 * nLayerSize5; i++){
		weight4[i] -= eps_w * delta_weight4[i];
	}
	for(int i = 0; i < nLayerSize5; i++){
		bias4[i] -= eps_b * delta_bias4[i];
	}

	for(int i = 0; i < nLayerSize3 * nLayerSize4; i++){
		weight3[i] -= eps_w * delta_weight3[i];
	}
	for(int i = 0; i < nLayerSize4; i++){
		bias3[i] -= eps_b * delta_bias3[i];
	}

	for(int i = 0; i < nLayerSize2 * nLayerSize3; i++){
		weight2[i] -= eps_w * delta_weight2[i];
	}
	for(int i = 0; i < nLayerSize3; i++){
		bias2[i] -= eps_b * delta_bias2[i];
	}

	for(int i = 0; i < nLayerSize1 * nLayerSize2; i++){
		weight1[i] -= eps_w * delta_weight1[i];
	}
	for(int i = 0; i < nLayerSize2; i++){
		bias1[i] -= eps_b * delta_bias1[i];
	}

	for(int i = 0; i < nLayerSize0 * nLayerSize1; i++){
		weight0[i] -= eps_w * delta_weight0[i];
	}
	for(int i = 0; i < nLayerSize1; i++){
		bias0[i] -= eps_b * delta_bias0[i];
	}
}

void autoencoder::train(){
	for(int epoch = 0; epoch < nEpochNum; epoch++){
		dataprovider->reset();
		double errsum = 0.0;
		printf("Epoch %d\n", epoch + 1);

		for(int batch = 0; batch < nBatchNum; batch++){
			layer0act = dataprovider->getNextBatch();
			fprop();
			bprop();
			update();
			delete[] layer0act;
		}

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

autoencoder::~autoencoder(){};








