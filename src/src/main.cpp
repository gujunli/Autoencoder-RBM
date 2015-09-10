#include "rbm.h"
#include "autoencoder.h"
#include "mnist.h"
#include "cifar10.h"
#include <ctime>


// generate fake data for test cases
// The first column of the data is the index of that row, while the followings are all zeros.

void test(int fileCount, int vecCount, int vecLength){
	ofstream		fout;
	string			fileNamePrefix 	=	"fakedata";
	unsigned char*	fakeDataBuffer	=	new	unsigned char[vecLength * vecCount];

	for(int i = 0; i < vecLength * vecCount; i++){
		fakeDataBuffer[i] = 0;
	}
	
	for(int fileId = 0; fileId < fileCount; fileId++){
		string	fileName = fileNamePrefix;
		generateFileName(&fileName, fileId, 3);
		fout.open(fileName.c_str(), ios_base::binary | ios_base::trunc);
		
		for(int i = 0; i < vecCount; i++){
			unsigned index 		= fileId * vecCount + i;
			unsigned char byte0	= (index >> 24) & 255;
			unsigned char byte1	= (index >> 16) & 255;
			unsigned char byte2	= (index >> 8)	& 255;
			unsigned char byte3	= index 		& 255;

			fakeDataBuffer[i * vecLength] = byte0;
			fakeDataBuffer[i * vecLength + 1] = byte1;
			fakeDataBuffer[i * vecLength + 2] = byte2;
			fakeDataBuffer[i * vecLength + 3] = byte3;
		}
		
		fout.write((char*)fakeDataBuffer, vecLength * vecCount * sizeof(unsigned char));
		fout.close();
	}	

}



int main(void)
{
	string rawfile, patchfile, shuffledfile;
	
	rawfile			= "../data/trainingimages.dat";
	patchfile		= "../data/patch";
	shuffledfile 	= "../data/shuffledPatch";

	string inputFile0, inputFile1, inputFile2, inputFile3;
	inputFile0 = shuffledfile;
	//inputFile0 = patchfile;
	inputFile1 = "../data/firstProb.dat";
	inputFile2 = "../data/secondProb.dat";
	inputFile3 = "../data/thirdProb.dat";

	const unsigned nBatchNum 	= 68000 * 30 * 81 / 128;
	const unsigned nVecCount	= 68000 * 30 * 81;

	// pre-processing to construct the training dataset in the retina format	
	//cifarPreProcessor* cifar = new cifarPreProcessor(rawfile, patchfile);
	//printf("Generating retina format patches...\n");
	//cifar->makePatchDataFiles();
	//printf("done!\n");
	//delete cifar;

	// permutate the input vectors in random order
	// datashuffler* dsl = new datashuffler(patchfile, nVecCount, 400, 336);
	// dsl->run();
	// delete dsl;

	//RBM_GPU* rbm0 = new RBM_GPU(0, 336, 1024, true, 80, nBatchNum, 128, 0.0002, 0.9, 0.9, "first");
	//rbm0->dataprovider = new dataProvider_GPU(rbm0->gpu_env, inputFile0, 336, 128, false);
	//rbm0->dataprovider->getExpectation();
	//rbm0->train();
	//rbm0->test();
	//delete rbm0;

	RBM_GPU* rbm1 = new RBM_GPU(0, 1024, 512, false, 80, nBatchNum, 128, 0.0002, 0.9, 0.9, "second");
	rbm1->dataprovider = new dataProvider_GPU(rbm1->gpu_env, inputFile1, 1024, 128, true);
	rbm1->train();
	rbm1->test();
	delete rbm1;

	RBM_GPU* rbm2 = new RBM_GPU(0, 512, 256, false, 80, nBatchNum, 128, 0.0002, 0.9, 0.9, "third");
	rbm2->dataprovider = new dataProvider_GPU(rbm2->gpu_env, inputFile2, 512, 128, true);
	rbm2->train();
	rbm2->test();
	delete rbm2;

	//RBM_GPU* rbm3 = new RBM_GPU(0, 256, 128, false, 80, nBatchNum, 128, 0.0002, 0.9, 0.9, "fourth");
	//rbm3->dataprovider = new dataProvider_GPU(rbm3->gpu_env, inputFile3, 256, 128, true);
	//rbm3->train();
	//delete rbm3;

	//autoencoder_GPU* ae = new autoencoder_GPU;
	//ae->dataprovider = new dataProvider_GPU(ae->gpu_env, inputFile0, 336, 128, false);
	//ae->train();

	return 0;
}
