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

#ifdef __SSE__
#include <xmmintrin.h>
#endif
#ifdef __SSE2__
#include <emmintrin.h>
#endif
#ifdef __SSE3__
#include <pmmintrin.h>
#endif
#ifdef __AVX__
#include <immintrin.h>
#endif

//#include <vec.h>
#include <pvfmm.h>
#include <args.h>
#include <profile.hpp>
#include <vector.hpp>
#include <matrix.hpp>
#include <precomp_mat.hpp>
#include <integration.hpp>
#include <kernel.hpp>
#include <mortonid.hpp>
#include <sort.hpp>
#include <fmm_node.hpp>
#include <interac_list.hpp>
#include <fmm_tree.hpp>

using namespace pvfmm;
using namespace exafmm;

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
  real_t Xmin = *std::max_element(coord.begin(), coord.end());
  real_t scale = 0.5 / (std::max(fabs(Xmax), fabs(Xmin)) + 1);
  for(int i=0; i<coord.size(); i++) coord[i] = coord[i]*scale + 0.5;
  return coord;
}

int main(int argc, char **argv){
  Args args(argc, argv);
  omp_set_num_threads(args.threads);
  size_t N = args.numBodies;
  size_t M = args.ncrit;
  int mult_order = args.PP;
  int depth = 15;
  Profile::Enable(true);
  Profile::Tic("FMM_Test",true);
  Kernel potn_ker=BuildKernel<potentialP2P>("laplace"    , std::pair<int,int>(1,1));
  Kernel grad_ker=BuildKernel<gradientP2P >("laplace_grad", std::pair<int,int>(1,3),
					     &potn_ker, &potn_ker, NULL, &potn_ker, &potn_ker, NULL, &potn_ker, NULL);

  InitData init_data;
  init_data.max_depth=depth;
  init_data.max_pts=M;
  std::vector<real_t> src_coord, src_value;
  srand48(0);
#if PLUMMER
  src_coord = plummer(N);
#else
  for(size_t i=0; i<3*N; i++) src_coord.push_back(drand48());
#endif
  for(size_t i=0; i<N; i++) src_value.push_back(drand48()-0.5);
  init_data.coord=src_coord;
  init_data.value=src_value;

  // initialize equiv surface coords for all levels
  size_t m=mult_order;
  upwd_check_surf.resize(MAX_DEPTH);
  upwd_equiv_surf.resize(MAX_DEPTH);
  dnwd_check_surf.resize(MAX_DEPTH);
  dnwd_equiv_surf.resize(MAX_DEPTH);
  for(size_t depth=0;depth<MAX_DEPTH;depth++){
    real_t c[3]={0.0,0.0,0.0};
    upwd_check_surf[depth].Resize((6*(m-1)*(m-1)+2)*3);
    upwd_equiv_surf[depth].Resize((6*(m-1)*(m-1)+2)*3);
    dnwd_check_surf[depth].Resize((6*(m-1)*(m-1)+2)*3);
    dnwd_equiv_surf[depth].Resize((6*(m-1)*(m-1)+2)*3);
    upwd_check_surf[depth] = u_check_surf(m,c,depth);
    upwd_equiv_surf[depth] = u_equiv_surf(m,c,depth);
    dnwd_check_surf[depth] = d_check_surf(m,c,depth);
    dnwd_equiv_surf[depth] = d_equiv_surf(m,c,depth);
  }

  FMM_Tree tree(mult_order);
#if POTENTIAL
  tree.Initialize(mult_order, &potn_ker);
#else
  tree.Initialize(mult_order, &grad_ker);
#endif
  for(size_t it=0;it<2;it++){
    Profile::Tic("TotalTime",true);
    tree.Initialize(&init_data);
    Profile::Tic("SetSrcTrg",true);
    std::vector<FMM_Node*>& nodes=tree.GetNodeList();
#pragma omp parallel for
    for(size_t i=0;i<nodes.size();i++){
      nodes[i]->trg_coord = nodes[i]->pt_coord;
      nodes[i]->src_coord = nodes[i]->pt_coord;
      nodes[i]->src_value = nodes[i]->pt_value;
      nodes[i]->trg_scatter = nodes[i]->pt_scatter;
      nodes[i]->src_scatter = nodes[i]->pt_scatter;
    }
    Profile::Toc();
    tree.SetupFMM();
    tree.RunFMM();
    Profile::Toc();
  }
  long nleaf=0, maxdepth=0;
  std::vector<size_t> all_nodes(MAX_DEPTH+1,0);
  std::vector<size_t> leaf_nodes(MAX_DEPTH+1,0);
  std::vector<FMM_Node*>& nodes=tree.GetNodeList();
  for(size_t i=0;i<nodes.size();i++){
    FMM_Node* n=nodes[i];
    all_nodes[n->depth]++;
    if(n->IsLeaf()){
      leaf_nodes[n->depth]++;
      if(maxdepth<n->depth) maxdepth=n->depth;
      nleaf++;
    }
  }
  std::cout<<"Leaf Nodes : "<<nleaf<<'\n';
  std::cout<<"Tree Depth : "<<maxdepth<<'\n';
  tree.CheckFMMOutput("Output");
  Profile::Toc();
  Profile::print();
  return 0;
}
