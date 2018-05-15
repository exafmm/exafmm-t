#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fftw3.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <omp.h>
#include <set>
#include <sstream>
#include <stack>
#include <stdint.h>
#include <string>
#include <sys/stat.h>
#include <vector>

#include <pvfmm.h>
#include <args.h>
#include <profile.hpp>
#include <matrix.hpp>
#include <precomp_mat.hpp>
#include <kernel.hpp>
#include <sort.hpp>
#include <interac_list.hpp>
#include <fmm_tree.hpp>

using namespace pvfmm;
using namespace exafmm;
std::vector<real_t> plummer(int);
std::vector<real_t> nonuniform(int);

int main(int argc, char **argv) {
  Args args(argc, argv);
  omp_set_num_threads(args.threads);
  size_t N = args.numBodies;
  NCRIT = args.ncrit;
  MULTIPOLE_ORDER = args.P;
  NSURF = 6*(MULTIPOLE_ORDER-1)*(MULTIPOLE_ORDER-1) + 2;
  Profile::Enable(true);
  Profile::Tic("Total", true);
  std::vector<real_t> src_coord, src_value;
  srand48(0);
#if 0
  //src_coord = plummer(N);
  src_coord = nonuniform(N);
#else
  for(size_t i=0; i<3*N; i++) src_coord.push_back(drand48());
#endif
  for(size_t i=0; i<N; i++) src_value.push_back(drand48()-0.5);

  Bodies bodies(args.numBodies);
  for(int i=0; i<bodies.size(); i++) {
    bodies[i].X[0] = src_coord[3*i+0];
    bodies[i].X[1] = src_coord[3*i+1];
    bodies[i].X[2] = src_coord[3*i+2];
    bodies[i].q = src_value[i];
  }

  FMM_Nodes cells = buildTree(bodies);
  // fill in pt_coord, pt_src, pt_cnt, correct coord for compatibility
  // remove this later
  for(int i=0; i<cells.size(); i++) {
    for(int d=0; d<3; d++) {
      cells[i].coord[d] = cells[i].X[d] - cells[i].R;
    }
    if(cells[i].IsLeaf()) {
      for(Body* B=cells[i].body; B<cells[i].body+cells[i].numBodies; B++) {
        cells[i].pt_coord.push_back(B->X[0]);
        cells[i].pt_coord.push_back(B->X[1]);
        cells[i].pt_coord.push_back(B->X[2]);
        cells[i].pt_src.push_back(B->q);
      }
    }
  }

  // initialize equiv surface coords for all levels
  upwd_check_surf.resize(MAX_DEPTH);
  upwd_equiv_surf.resize(MAX_DEPTH);
  dnwd_check_surf.resize(MAX_DEPTH);
  dnwd_equiv_surf.resize(MAX_DEPTH);
  for(size_t depth=0; depth<MAX_DEPTH; depth++) {
    real_t c[3] = {0.0};
    upwd_check_surf[depth].resize(NSURF*3);
    upwd_equiv_surf[depth].resize(NSURF*3);
    dnwd_check_surf[depth].resize(NSURF*3);
    dnwd_equiv_surf[depth].resize(NSURF*3);
    upwd_check_surf[depth] = u_check_surf(c, depth);
    upwd_equiv_surf[depth] = u_equiv_surf(c, depth);
    dnwd_check_surf[depth] = d_check_surf(c, depth);
    dnwd_equiv_surf[depth] = d_equiv_surf(c, depth);
  }
  InteracList interacList;
  interacList.Initialize();
  Kernel potn_ker = BuildKernel<potentialP2P>("laplace", std::pair<int, int>(1, 1));
  Kernel grad_ker = BuildKernel<gradientP2P >("laplace_grad", std::pair<int, int>(1, 3),
                    &potn_ker, &potn_ker, NULL, &potn_ker, &potn_ker, NULL, &potn_ker, NULL);
#if POTENTIAL
  Kernel * kernel = &potn_ker;
#else
  Kernel * kernel = &grad_ker;
#endif
  kernel->Initialize();
  Profile::Tic("Precomputation", true);
  PrecompMat pmat(&interacList, kernel);
  Profile::Toc();
  FMM_Tree tree(kernel, &interacList, &pmat);
  for(size_t it=0; it<1; it++) {
    Profile::Tic("TotalTime", true);
    tree.root_node = &cells[0];
    tree.SetupFMM();
    tree.RunFMM();
    Profile::Toc();
  }
  std::cout << std::setw(20) << std::left << "Leaf Nodes" << " : "<< leafs.size() <<'\n';
  std::cout << std::setw(20) << std::left << "Tree Depth" << " : "<< LEVEL <<'\n';
  tree.CheckFMMOutput("Output");
  Profile::Toc();
  Profile::print();
  return 0;
}

// generate plummer distribution in 0 to 1 cube
std::vector<real_t> plummer(int numBodies) {
  srand48(0);
  std::vector<real_t> coord;
  int i = 0;
  while (i < numBodies) {
    real_t X1 = drand48();
    real_t X2 = drand48();
    real_t X3 = drand48();
    real_t R = 1.0 / sqrt( (pow(X1, -2.0 / 3.0) - 1.0) );
    if (R < 100) {
      real_t Z = (1.0 - 2.0 * X2) * R;
      real_t X = sqrt(R * R - Z * Z) * std::cos(2.0 * M_PI * X3);
      real_t Y = sqrt(R * R - Z * Z) * std::sin(2.0 * M_PI * X3);
      coord.push_back(X);
      coord.push_back(Y);
      coord.push_back(Z);
      i++;
    }
  }
  real_t Xmax = *std::max_element(coord.begin(), coord.end());
  real_t Xmin = *std::min_element(coord.begin(), coord.end());
  real_t scale = 0.5 / (std::max(fabs(Xmax), fabs(Xmin)) + 1);
  for(int i=0; i<coord.size(); i++) coord[i] = coord[i]*scale + 0.5;
  return coord;
}

std::vector<real_t> nonuniform(int numBodies) {
  srand48(0);
  std::vector<real_t> coord;
  for(size_t i=0; i<3*numBodies; i++) {
    if (i/3 < 0.1*numBodies) coord.push_back(drand48()*0.5);
    else {
      if (i/3 < 0.2*numBodies) coord.push_back(0.5 + drand48()*0.5);
      else coord.push_back(drand48());
    }
  }
  return coord;
}
