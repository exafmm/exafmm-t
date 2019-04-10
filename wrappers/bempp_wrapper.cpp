#include <complex>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include "build_non_adaptive_tree.h"
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
  real_t WAVEK;
#endif
  Nodes NODES;
  NodePtrs LEAFS;

#if COMPLEX
  typedef complex_t value_t;
#else
  typedef real_t value_t;
#endif

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
    P = 10;
    NSURF = 6*(P-1)*(P-1) + 2;
#if HELMHOLTZ
    WAVEK = 20;
#endif
    args.P = P;
    args.ncrit = 120;
    args.threads = threads;
#if HAVE_OPENMP
    omp_set_num_threads(args.threads);
#endif
    MAXLEVEL = args.maxlevel;
  }

  // build non-adaptive tree, precompute invariant matrices, build interaction lists
  extern "C" void setup_FMM(int src_count, real_t* src_coord,
                            int trg_count, real_t* trg_coord) {
    Bodies sources = array_to_bodies(src_count, src_coord);
    Bodies targets = array_to_bodies(trg_count, trg_coord);

    start("Build Tree");
    get_bounds(sources, targets, XMIN0, R0);
    NodePtrs nonleafs;
    NODES = build_tree(sources, targets, XMIN0, R0, LEAFS, nonleafs);
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

  extern "C" void run_FMM(value_t* src_value, value_t* trg_value) {
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

  extern "C" void verify_FMM(int src_count, real_t* src_coord, value_t* src_value,
                             int trg_count, real_t* trg_coord, value_t* trg_value) {
    int ntrgs = 30;  // number of sampled targets
    int stride = trg_count / ntrgs;
    // use RealVec type in P2P kernel
    RealVec src_coord_(src_coord, src_coord+3*src_count);
    vector<value_t> src_value_(src_value, src_value+src_count);
    RealVec trg_coord_(3*ntrgs);
    vector<value_t> trg_value_(4*ntrgs, 0);
    // prepare the coordinates of the sampled targets
    for(int i=0; i<ntrgs; ++i) {
      int itrg = i*stride;
      for(int d=0; d<3; ++d) {
        trg_coord_[3*i+d] = trg_coord[3*itrg+d];
      }
    }
    gradient_P2P(src_coord_, src_value_, trg_coord_, trg_value_);
    // compute relative error in L2 norm
    real_t p_norm = 0, p_diff = 0, g_norm = 0, g_diff = 0;
    for(int i=0; i<ntrgs; ++i) {
      int itrg = i*stride;
#if COMPLEX
      p_norm += norm(trg_value_[4*i]);
      p_diff += norm(trg_value_[4*i]-trg_value[4*itrg]);
      for(int d=1; d<4; ++d) {
        g_norm += norm(trg_value_[4*i+d]);
        g_diff += norm(trg_value_[4*i+d]-trg_value[4*itrg+d]);
      }
#else
      p_norm += trg_value_[4*i] * trg_value_[4*i];
      p_diff += (trg_value_[4*i]-trg_value[4*itrg]) * (trg_value_[4*i]-trg_value[4*itrg]);
      for(int d=1; d<4; ++d) {
        g_norm += trg_value_[4*i+d] * trg_value_[4*i+d];
        g_diff += (trg_value_[4*i+d]-trg_value[4*itrg+d]) * (trg_value_[4*i+d]-trg_value[4*itrg+d]);
      }
#endif
    }
    cout << setw(20) << left << "Potn Error" << " : " << scientific << sqrt(p_diff/p_norm) << endl;
    cout << setw(20) << left << "Grad Error" << " : " << scientific << sqrt(g_diff/g_norm) << endl;
  }
}
