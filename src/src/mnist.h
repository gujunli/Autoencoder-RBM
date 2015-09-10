#ifndef _MNIST_H_
#define _MNIST_H_

#include "utils.h"
#include<algorithm>
#include<string>

typedef unsigned char byte;

typedef struct{
	floatType seed;
	unsigned int index;
}Pair;

class MNIST
{
protected:
	unsigned int nPixelPerImage;
	unsigned int nImageNum;
	vector<floatType*> batchData;
	floatType* rawData;
	byte* label;
	string strDataFileName;
	string strLabelFileName;
public:
	MNIST(string fdata, string ldata);
	~MNIST();
	void loadData(void);
	void makeBatch(unsigned int numBatch);
	vector<floatType*> getData(unsigned int numBatch);
};

#endif