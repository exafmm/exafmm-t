.SUFFIXES: .cpp .cu
CXX = mpiicpc
CXXFLAGS = -lfftw3 -lfftw3f -Wfatal-errors -g -O3 -fabi-version=6 -std=c++11 -fopenmp -debug all -traceback -I./include
NVCC = nvcc
NVCCFLAGS = -use_fast_math -dc -arch=sm_60 -Xcompiler "-g -O3 -fabi-version=6 -std=c++11 -fopenmp -I./include"
LDFLAGS = -lcufft -lcublas -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -lcudart -lcudadevrt
OBJ = main.o src/geometry.o src/laplace.o src/laplace_cuda.o src/profile.o link.o
LIB_PATHS = -L/mnt/nfs/packages/x86_64/cuda/cuda-10.0/lib64
%.o: %.cpp
	time $(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@ -D${TYPE}
%.o: %.cu
	time $(NVCC) $(NVCCFLAGS)  --device-c src/laplace_cuda.cu -o $@ -D${TYPE}
	time $(NVCC) --gpu-architecture=sm_60 --device-link src/laplace_cuda.o --output-file link.o

all: $(OBJ)
	$(CXX) $(CXXFLAGS) $? $(LIB_PATHS) $(LDFLAGS)
clean:
	rm -f $(OBJ) *.out

p4:
	./a.out -T 8 -n 200000 -P 4 -c 64

p16:
	./a.out -T 8 -n 1000000 -P 16 -c 320

t4:
	./a.out -T 32 -n 1000000 -P 4 -c 64

t16:
	./a.out -T 32 -n 1000000 -P 16 -c 320

