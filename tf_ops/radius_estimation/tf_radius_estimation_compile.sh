#/bin/bash
CUDA_ROOT=/usr/local/cuda-10.0
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

echo $CUDA_ROOT
echo $TF_INC
echo $TF_LIB

#g++ -std=c++11 tf_radius_estimation.cpp tf_radius_estimation_g.cu.o -o tf_radius_estimation_so.so -shared -fPIC -I /home/artem/anaconda3/lib/python3.6/site-packages/tensorflow/include -I /usr/local/cuda-10.1/include -I /home/artem/anaconda3/lib/python3.6/site-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-10.1/lib64/ -L /home/artem/anaconda3/lib/python3.6/site-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0

$CUDA_ROOT/bin/nvcc tf_radius_estimation_g.cu -o tf_radius_estimation_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF>=1.4.0
g++ -std=c++11 tf_radius_estimation.cpp tf_radius_estimation_g.cu.o -o tf_radius_estimation_so.so -shared -fPIC -I$TF_INC/ -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -I$CUDA_ROOT/include -lcudart -L$CUDA_ROOT/lib64/ -O2 #-D_GLIBCXX_USE_CXX11_ABI=0