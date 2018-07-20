.SUFFIXES: .cpp

WFLAGS = -fmudflap -fno-strict-aliasing -fsanitize=address -fsanitize=leak -fstack-protector -ftrapv -Wall -Warray-bounds -Wcast-align -Wcast-qual -Wextra -Wfatal-errors -Wformat=2 -Wformat-nonliteral -Wformat-security -Winit-self -Wmissing-format-attribute -Wmissing-include-dirs -Wmissing-noreturn -Wno-missing-field-initializers -Wno-overloaded-virtual -Wno-unused-local-typedefs -Wno-unused-parameter -Wno-unused-variable -Wpointer-arith -Wredundant-decls -Wreturn-type -Wshadow -Wstrict-aliasing -Wstrict-overflow=5 -Wswitch-enum -Wuninitialized -Wunreachable-code -Wunused-but-set-variable -Wwrite-strings -Wno-error=missing-field-initializers -Wno-error=overloaded-virtual -Wno-error=unused-local-typedefs -Wno-error=unused-parameter -Wno-error=unused-variable
# -Wsign-compare -Werror

CXX = mpiicpc
CXXFLAGS = -g -O3 -mavx -fabi-version=6 -std=c++11 -fopenmp -debug all -traceback -I./include
LDFLAGS = -lfftw3 -lfftw3f -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm
#CXX = mpicxx
#CXXFLAGS = -g -O3 -mavx -fabi-version=6 -std=c++11 -fopenmp -I./include
#LDFLAGS = -lfftw3 -lfftw3f -lpthread -lblas -llapack -lm

%.o: %.cxx
	$(CXX) $(CXXFLAGS) -c $< -o $@

float: main.cxx
	time $(CXX) $(CXXFLAGS) -c $< -o main.o -DFLOAT

double: main.cxx
	time $(CXX) $(CXXFLAGS) -c $< -o main.o

link: main.o
	$(CXX) $(CXXFLAGS) $? $(LDFLAGS)

clean:
	rm -f *.o *.out

p4:
	./a.out -T 8 -n 1000000 -P 4 -c 64

p16:
	./a.out -T 8 -n 1000000 -P 16 -c 320

y4:
	./a.out -T 32 -n 1000000 -P 4 -c 64

y16:
	./a.out -T 32 -n 1000000 -P 16 -c 320
