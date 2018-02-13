.SUFFIXES: .cpp

CXX = mpic++ -g -O3 -mavx -fabi-version=6 -std=gnu++11 -fopenmp -I./include
#CXX = g++-mp-5 -g -O3 -msse4 -std=c++11 -fopenmp -I./include
LDFLAGS = -lfftw3 -lfftw3f -llapack -lblas

%.o: %.cpp
	$(CXX) -c $< -o $@

potential: main.cpp
	time $(CXX) $(CXXFLAGS) -c $< -o main.o -DFLOAT -DPOTENTIAL

force: main.cpp
	time $(CXX) -c $< -o main.o -DFLOAT

double: main.cpp
	time $(CXX) -c $< -o main.o

nonuniform: main.cpp
	time $(CXX) -c $< -o main.o -DFLOAT -DNONUNIFORM

link: main.o
	$(CXX) $? $(LDFLAGS)

clean:
	rm -f *.o *.out

cleandat:
	rm -f $(PVFMM_DIR)/*f.data

tags:
	find . -name "*.cpp" -o -name "*.hpp" | xargs etags -f TAGS

r4:
	./a.out -T 4 -n 100000 -P 4 -c 64

r16:
	./a.out -T 4 -n 100000 -P 16 -c 320

y4:
	./a.out -T 32 -n 1000000 -P 4 -c 64

y16:
	./a.out -T 32 -n 1000000 -P 16 -c 320

e4:
	./a.out -T 72 -n 1000000 -P 4 -c 64

e16:
	./a.out -T 72 -n 1000000 -P 16 -c 320
