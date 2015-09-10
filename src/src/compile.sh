#!/bin/bash

g++ -I /opt/acml5.3.1/ifort64_fma4_mp/include/ -I /opt/AMDAPP/include -I /opt/clAmdBlas-1.10.321/include/ -L /opt/acml5.3.1/ifort64_fma4_mp/lib/ -L /opt/AMDAPP/lib/x86_64 -L /opt/clAmdBlas-1.10.321/lib64/ main.cpp cifar10.cpp mnist.cpp rbm.cpp rbm_gpu.cpp autoencoder.cpp autoencoder_gpu.cpp utils.cpp -l OpenCL -l clAmdBlas -l acml_mp -l iomp5 -o ../bin/autoencoder


#g++ -Wall shuffledata.cpp -o ../bin/shuffledata
