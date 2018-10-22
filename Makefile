.SUFFIXES: .cpp .cu

WFLAGS = -fmudflap -fno-strict-aliasing -fsanitize=address -fsanitize=leak -fstack-protector -ftrapv -Wall -Warray-bounds -Wcast-align -Wcast-qual -Wextra -Wfatal-errors -Wformat=2 -Wformat-nonliteral -Wformat-security -Winit-self -Wmissing-format-attribute -Wmissing-include-dirs -Wmissing-noreturn -Wno-missing-field-initializers -Wno-overloaded-virtual -Wno-unused-local-typedefs -Wno-unused-parameter -Wno-unused-variable -Wpointer-arith -Wredundant-decls -Wreturn-type -Wshadow -Wstrict-aliasing -Wstrict-overflow=5 -Wswitch-enum -Wuninitialized -Wunreachable-code -Wunused-but-set-variable -Wwrite-strings -Wno-error=missing-field-initializers -Wno-error=overloaded-virtual -Wno-error=unused-local-typedefs -Wno-error=unused-parameter -Wno-error=unused-variable
# -Wsign-compare -Werror

CXX = mpiicpc
CXXFLAGS = -g -O3 -mavx -fabi-version=6 -std=c++11 -fopenmp -debug all -traceback -I./include
LDFLAGS = -lfftw3 -lfftw3f -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm
#CXX = mpicxx
#CXXFLAGS = -g -O3 -mavx -fabi-version=6 -std=c++11 -fopenmp -I./include
#LDFLAGS = -lfftw3 -lfftw3f -lpthread -lblas -llapack -lm

OBJF = main.fo src/geometry.fo src/laplace.fo src/kernel.fo
OBJD = main.do src/geometry.do src/laplace.do src/kernel.do
OBJC =  main.co src/geometry.co src/laplace_c.co src/kernel.co
OBJZ =  main.zo src/geometry.zo src/laplace_c.zo src/kernel.zo
OBJH =  main.ho src/geometry.ho src/laplace_c.ho src/kernel.ho
OBJHD =  main.hdo src/geometry.hdo src/laplace_c.hdo src/kernel.hdo

%.fo: %.cpp
	time $(CXX) $(CXXFLAGS) -c $< -o $@ -DFLOAT -DEXAFMM_LAPLACE

%.do: %.cpp
	time $(CXX) $(CXXFLAGS) -c $< -o $@ -DEXAFMM_LAPLACE

%.co: %.cpp
	time $(CXX) $(CXXFLAGS) -c $< -o $@ -DFLOAT -DCOMPLEX -DEXAFMM_LAPLACE

%.zo: %.cpp
	time $(CXX) $(CXXFLAGS) -c $< -o $@ -DCOMPLEX -DEXAFMM_LAPLACE

%.ho: %.cpp
	time $(CXX) $(CXXFLAGS) -c $< -o $@ -DFLOAT -DCOMPLEX -DEXAFMM_HELMHOLTZ

%.hdo: %.cpp
	time $(CXX) $(CXXFLAGS) -c $< -o $@ -DCOMPLEX -DEXAFMM_HELMHOLTZ

real8: $(OBJF)
	$(CXX) $(CXXFLAGS) $? $(LDFLAGS)

real16: $(OBJD)
	$(CXX) $(CXXFLAGS) $? $(LDFLAGS)

complex8: $(OBJC)
	$(CXX) $(CXXFLAGS) $? $(LDFLAGS)

complex16: $(OBJZ)
	$(CXX) $(CXXFLAGS) $? $(LDFLAGS)

helmholtz8: $(OBJH)
	$(CXX) $(CXXFLAGS) $? $(LDFLAGS)

helmholtz16: $(OBJHD)
	$(CXX) $(CXXFLAGS) $? $(LDFLAGS)

clean:
	rm -f $(OBJF) $(OBJD) $(OBJC) $(OBJZ) $(OBJH) $(OBJHD) *.out

p4:
	./a.out -T 8 -n 1000000 -P 4 -c 64

p16:
	./a.out -T 8 -n 1000000 -P 16 -c 320

t4:
	./a.out -T 32 -n 1000000 -P 4 -c 64

t16:
	./a.out -T 32 -n 1000000 -P 16 -c 320
