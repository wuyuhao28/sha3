build:all
libcutil.so:
	/usr/local/cuda/bin/nvcc --shared -o libcutil.so memorypool.cpp memorypoolmanager.cpp matrixcal.cu byte_order.c sha3.c interface.cpp RequestQueue.cpp --compiler-options '-fPIC' -lpthread -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -lcublas -O3 -std=c++11 -m64

all:libcutil.so testxx.cpp
	/usr/local/cuda/bin/nvcc testxx.cpp -L. -lcutil -lpthread -I./src -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -lcublas -O3 -std=c++11
  

clean:
	rm -fr *.so *.o a.out
