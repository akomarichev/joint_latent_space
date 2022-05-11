nvcc = /usr/local/cuda-10.0/bin/nvcc
cudalib = /usr/local/cuda-10.0/lib64
tensorflow = /home/artem/anaconda3/lib/python3.7/site-packages/tensorflow/include
tf = /home/artem/anaconda3/lib/python3.7/site-packages/tensorflow

all: tf_nndistance_so.so tf_nndistance_g.cu.o

tf_nndistance_so.so: tf_nndistance_g.cu.o tf_nndistance.cpp
	g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I $(tensorflow) -I $(tensorflow)/external/nsync/public -lcudart -L $(cudalib) -L$(tf) -ltensorflow_framework -O2

tf_nndistance_g.cu.o: tf_nndistance_g.cu
	$(nvcc) tf_nndistance_g.cu -o tf_nndistance_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC


clean:
	rm tf_nndistance_so.so
	rm  *.cu.o 
