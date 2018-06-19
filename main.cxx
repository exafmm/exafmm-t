#include <omp.h>
#include "args.h"
#include "build_tree.h"
#include "fmm_tree.h"
#include "interaction_list.h"
#include "kernel.h"
#include "precompute.h"

using namespace exafmm_t;
using namespace exafmm;
RealVec plummer(int);
RealVec nonuniform(int);

int main(int argc, char **argv) {
  Args args(argc, argv);
  omp_set_num_threads(args.threads);
  size_t N = args.numBodies;
  NCRIT = args.ncrit;
  MULTIPOLE_ORDER = args.P;
  NSURF = 6*(MULTIPOLE_ORDER-1)*(MULTIPOLE_ORDER-1) + 2;
  N1 = 2 * MULTIPOLE_ORDER;
  N2 = N1 * N1;
  N3 = N1 * N2;
  N3_ = N2 * (N1/2+1);
  Profile::Enable(true);
  RealVec src_coord, src_value;
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

  std::vector<Node*> leafs, nonleafs;
  Nodes nodes = buildTree(bodies, leafs, nonleafs);

  // fill in pt_coord, pt_src, correct coord for compatibility
  // remove this later
  for(int i=0; i<nodes.size(); i++) {
    for(int d=0; d<3; d++) {
      nodes[i].coord[d] = nodes[i].X[d] - nodes[i].R;
    }
    if(nodes[i].IsLeaf()) {
      for(Body* B=nodes[i].body; B<nodes[i].body+nodes[i].numBodies; B++) {
        nodes[i].pt_coord.push_back(B->X[0]);
        nodes[i].pt_coord.push_back(B->X[1]);
        nodes[i].pt_coord.push_back(B->X[2]);
        nodes[i].pt_src.push_back(B->q);
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

  Profile::Tic("Total", true);
  initRelCoord();    // initialize relative coords
  Profile::Tic("Precomputation", true);
  Precompute();
  Profile::Toc();
  setColleagues(nodes);
  buildList(nodes);
  M2LSetup(M2Ldata, nonleafs);
  upwardPass(nodes, leafs);
  downwardPass(nodes, leafs);
  Profile::Toc();
  std::cout << std::setw(20) << std::left << "Leaf Nodes" << " : "<< leafs.size() <<'\n';
  std::cout << std::setw(20) << std::left << "Tree Depth" << " : "<< leafs.back()->depth <<'\n';
  CheckFMMOutput(leafs);
  Profile::print();
  return 0;
}

// generate plummer distribution in 0 to 1 cube
RealVec plummer(int numBodies) {
  srand48(0);
  RealVec coord;
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

RealVec nonuniform(int numBodies) {
  srand48(0);
  RealVec coord;
  for(size_t i=0; i<3*numBodies; i++) {
    if (i/3 < 0.1*numBodies) coord.push_back(drand48()*0.5);
    else {
      if (i/3 < 0.2*numBodies) coord.push_back(0.5 + drand48()*0.5);
      else coord.push_back(drand48());
    }
  }
  return coord;
}
