#include "mnist.h"

bool cmp(Pair a, Pair b){
	return a.seed < b.seed;
}

MNIST::MNIST(string fdata, string ldata){
	nPixelPerImage = 0;
	nImageNum = 0;
	rawData = NULL;
	label = NULL;
	strDataFileName = fdata;
	strLabelFileName = ldata;
}


void MNIST::loadData(void){
	ifstream fin;
	fin.open(strDataFileName.c_str(), ios_base::binary);

	// default data size
	// later to be modified to read from file
	unsigned int magicNum = 0, row = 0, column = 0;
	nImageNum = 0;

	byte buf;
	for(int i = 0; i < 4; i++){
		fin.read((char*)&buf, 1);
		magicNum <<= 8;
		magicNum |= buf;
	}
	for(int i = 0; i < 4; i++){
		fin.read((char*)&buf, 1);
		nImageNum <<= 8;
		nImageNum |= buf;
	}
	for(int i = 0; i < 4; i++){
		fin.read((char*)&buf, 1);
		row <<= 8;
		row |= buf;
	}
	for(int i = 0; i < 4; i++){
		fin.read((char*)&buf, 1);
		column <<= 8;
		column |= buf;
	}

	nPixelPerImage = row * column;
	if(rawData != NULL){
		delete[] rawData;
	}
	rawData = new floatType[nPixelPerImage * nImageNum];

	// read data from file
	for(int index = 0; index < nImageNum; index++){
		for(int feature = 0; feature < nPixelPerImage; feature++){
			byte temp;
			fin.read((char*)&temp, 1);
			rawData[feature + index * nPixelPerImage] = ((floatType)temp) / 255.0;
		}
	}

	fin.close();

	fin.open(strLabelFileName.c_str(), ios_base::binary);

	// read labels from file
	if(label != NULL){
		delete[] label;
	}
	label = new byte[nImageNum];
	for(int i = 0; i < 4; i++){
		fin >> buf;
		magicNum <<= 8;
		magicNum |= buf;
	}
	for(int i = 0; i < 4; i++){
		fin >> buf;
		nImageNum <<= 8;
		nImageNum |= buf;
	}
	fin.read((char*)label, nImageNum);
	fin.close();

	return;
}

void MNIST::makeBatch(unsigned int numBatch){
	Pair* perm = new Pair[nImageNum];
	for(int i = 0; i < nImageNum; i++){
		perm[i].seed = rand();
		perm[i].index = i;
	}
	sort(perm, perm + nImageNum, cmp);
	
	unsigned int nVectorPerBatch = nImageNum / numBatch;
	for(int i = 0; i < numBatch; i++){
		floatType* batch = new floatType[nPixelPerImage * nVectorPerBatch];
		for(int index = 0; index < nVectorPerBatch; index++){
			for(int feature = 0; feature < nPixelPerImage; feature++){
				batch[feature + index * nPixelPerImage] = rawData[feature + nPixelPerImage * perm[i * nVectorPerBatch + index].index];
			}
		}
		batchData.push_back(batch);
	}
}

vector<floatType*> MNIST::getData(unsigned int numBatch){
	loadData();
	makeBatch(numBatch);
	return batchData;
}
