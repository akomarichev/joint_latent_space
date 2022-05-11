#!/usr/bin/env bash
/usr/local/cuda-10.0/bin/nvcc tf_nndistance_g.cu -o tf_nndistance_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
#g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -I /usr/local/lib/python2.7/dist-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-8.0/lib64/ -L/usr/local/lib/python2.7/dist-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I /home/artem/anaconda3/lib/python3.7/site-packages/tensorflow/include  -I /usr/local/cuda-10.0/include -I /home/artem/anaconda3/lib/python3.7/site-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-10.0/lib64 -L/home/artem/anaconda3/lib/python3.7/site-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=1

# nvcc = /usr/local/cuda-10.0/bin/nvcc
# cudalib = /usr/local/cuda-10.0/lib64
# tensorflow = /home/artem/anaconda3/lib/python3.7/site-packages/tensorflow/include
# tf = /home/artem/anaconda3/lib/python3.7/site-packages/tensorflow