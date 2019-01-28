.SUFFIXES: .cpp .cu
CXX = mpiicpc
CXXFLAGS = -lfftw3 -lfftw3f -Wfatal-errors -g -O3 -mavx -fabi-version=6 -std=c++11 -fopenmp -debug all -traceback -I./include

NVCC = nvcc
NVCCFLAGS = -use_fast_math -arch=sm_60 -Xcompiler "-g -O3 -mavx -fabi-version=6 -std=c++11 -fopenmp -I./include"
LDFLAGS = -lcufft -lcublas -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -lcudart

OBJ = main.o src/geometry.o src/laplace.o src/laplace_cuda.o src/profile.o

%.o: %.cpp
	time $(CXX) $(CXXFLAGS) -c $< -o $@ -D${TYPE}
%.o: %.cu
	time $(NVCC) $(NVCCFLAGS) -c $<  -o $@ -D${TYPE}

all: $(OBJ)
	$(CXX) $(CXXFLAGS) $? $(LDFLAGS)
clean:
	rm -f $(OBJ) *.out

p4:
	./a.out -T 8 -n 1000000 -P 4 -c 64

p16:
	./a.out -T 8 -n 1000000 -P 16 -c 320

t4:
	./a.out -T 32 -n 1000000 -P 4 -c 64

t16:
	./a.out -T 32 -n 1000000 -P 16 -c 320
