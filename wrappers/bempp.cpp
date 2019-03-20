#include <iostream>
#include <omp.h>
#include "build_tree.h"
#include "build_list.h"
#include "config.h"
#include "dataset.h"
#if HELMHOLTZ
#include "helmholtz.h"
#else
#include "laplace.h"
#endif
#include "traverse.h"

using namespace std;
namespace exafmm_t {
  // global variables
  Args args;
  int P;
  int NSURF;
  int MAXLEVEL;
  vec3 XMIN0;
  real_t R0;
#if HELMHOLTZ
  real_t MU;
#endif
  Nodes NODES;
  NodePtrs LEAFS;

  // Convert SoA to AoS
  Bodies array_to_bodies(int count, real_t* coord) {
    Bodies bodies(count);
    for (int i=0; i<count; ++i) {
      bodies[i].ibody = i;
      for (int d=0; d<3; ++d) {
        bodies[i].X[d] = coord[3*i+d];
      }
    }
    return bodies;
  }

  // Initialize args and set global constants
  extern "C" void init_FMM(int threads) {
    P = 6;
    NSURF = 6*(P-1)*(P-1) + 2;
#if HELMHOLTZ
    MU = 20;
#endif
    args.P = P;
    args.ncrit = 120;
    args.threads = threads;
#if HAVE_OPENMP
    omp_set_num_threads(args.threads);
#endif
  }

  // build 2:1 balanced tree, precompute invariant matrices, build interaction lists
  extern "C" void setup_FMM(int src_count, real_t* src_coord,
                            int trg_count, real_t* trg_coord) {
    Bodies sources = array_to_bodies(src_count, src_coord);
    Bodies targets = array_to_bodies(trg_count, trg_coord);

    start("Build Tree");
    get_bounds(sources, targets, XMIN0, R0);
    NodePtrs nonleafs;
    NODES = build_tree(sources, targets, XMIN0, R0, LEAFS, nonleafs, args);
    balance_tree(NODES, sources, targets, XMIN0, R0, LEAFS, nonleafs, args);
    stop("Build Tree");

    init_rel_coord();
    start("Precomputation");
    precompute();
    stop("Precomputation");
    start("Build Lists");
    set_colleagues(NODES);
    build_list(NODES);
    stop("Build Lists");
    M2L_setup(nonleafs);
  }

  extern "C" void run_FMM(real_t* src_value, real_t* trg_value) {
    // update sources' charge
    for(size_t i=0; i<LEAFS.size(); ++i) {
      Node* leaf = LEAFS[i];
      for(int j=0; j<leaf->nsrcs; ++j) {
        int isrc = leaf->isrcs[j];
        leaf->src_value[j] = src_value[isrc];
      }
    }

    start("Total");
    upward_pass(NODES, LEAFS);
    downward_pass(NODES, LEAFS);
    stop("Total");

    // update targets' potential and gradient
    for(size_t i=0; i<LEAFS.size(); ++i) {
      Node* leaf = LEAFS[i];
      for(int j=0; j<leaf->ntrgs; ++j) {
        int itrg = leaf->itrgs[j];
        for(int d=0; d<4; ++d) {
          trg_value[4*itrg+d] = leaf->trg_value[4*j+d];
        }
      }
    }
  }

  extern "C" void verify_FMM(int src_count, real_t* src_coord, real_t* src_value,
                             int trg_count, real_t* trg_coord, real_t* trg_value) {
    int ntrgs = 20;  // number of sampled targets
    int stride = trg_count / ntrgs;
    // use RealVec type in P2P kernel
    RealVec src_coord_(src_coord, src_coord+3*src_count);
    RealVec src_value_(src_value, src_value+src_count);
    RealVec trg_coord_(3*ntrgs);
    RealVec trg_value_(4*ntrgs, 0);
    RealVec trg_value_fmm(4*ntrgs, 0);
    for(int i=0; i<ntrgs; ++i) {
      int itrg = i*stride;
      for(int d=0; d<3; ++d) {
        trg_coord_[3*i+d] = trg_coord[3*itrg+d];
      }
      for(int d=0; d<4; ++d) {
        trg_value_fmm[4*i+d] = trg_value[4*itrg+d];   // store sampled targets' values
      }
    }
    gradient_P2P(src_coord_, src_value_, trg_coord_, trg_value_);
    for(int i=0; i<ntrgs; ++i) {
      int itrg = i*stride;
      for(int d=0; d<4; ++d) {
        cout << trg_value_[4*i+d] << " " << trg_value[4*itrg+d] << endl;
      }
    }
  }
}
