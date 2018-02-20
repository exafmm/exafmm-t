.SUFFIXES: .cpp

CXX = mpic++ -g -O3 -mavx -fabi-version=6 -std=gnu++11 -fopenmp -I./include
#CXX = g++-mp-5 -g -O3 -msse4 -std=c++11 -fopenmp -I./include
LDFLAGS = -lfftw3 -lfftw3f -llapack -lblas

%.o: %.cpp
	$(CXX) -c $< -o $@

pfloat: main.cpp
	time $(CXX) -c $< -o main.o -DFLOAT -DPOTENTIAL

pdouble: main.cpp
	time $(CXX) -c $< -o main.o -DPOTENTIAL

float: main.cpp
	time $(CXX) -c $< -o main.o -DFLOAT

double: main.cpp
	time $(CXX) -c $< -o main.o

link: main.o
	$(CXX) $? $(LDFLAGS)

clean:
	rm -f *.o *.out

cleandat:
	rm -f $(PVFMM_DIR)/*f.data

tags:
	find . -name "*.cpp" -o -name "*.hpp" | xargs etags -f TAGS

p4:
	./a.out -T 8 -n 1000000 -P 4 -c 64

p16:
	./a.out -T 8 -n 1000000 -P 16 -c 320

y4:
	./a.out -T 32 -n 1000000 -P 4 64

y16:
	./a.out -T 32 -n 1000000 -P 16 -c 320

