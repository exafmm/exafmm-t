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

namespace exafmm_t {
  // global variables
  Args args;
  int P;
  int NSURF;
  int MAXLEVEL;
  vec3 X0;
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
  extern "C" void init_FMM(int p, int maxlevel, int threads, real_t wavek=20) {
    P = p;
    NSURF = 6*(P-1)*(P-1) + 2;
#if HELMHOLTZ
    WAVEK = wavek;
#endif
    args.P = P;
    args.threads = threads;
#if HAVE_OPENMP
    omp_set_num_threads(args.threads);
#endif
    MAXLEVEL = maxlevel;
  }

  // build non-adaptive tree, precompute invariant matrices, build interaction lists
  extern "C" void setup_FMM(int src_count, real_t* src_coord,
                            int trg_count, real_t* trg_coord) {
    Bodies sources = array_to_bodies(src_count, src_coord);
    Bodies targets = array_to_bodies(trg_count, trg_coord);

    start("Build Tree");
    get_bounds(sources, targets, X0, R0);
    NodePtrs nonleafs;
    NODES = build_tree(sources, targets, X0, R0, LEAFS, nonleafs);
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
    std::vector<value_t> src_value_(src_value, src_value+src_count);
    RealVec trg_coord_(3*ntrgs);
    std::vector<value_t> trg_value_(4*ntrgs, 0);
    // prepare the coordinates of the sampled targets
    for(int i=0; i<ntrgs; ++i) {
      int itrg = i*stride;
      for(int d=0; d<3; ++d) {
        trg_coord_[3*i+d] = trg_coord[3*itrg+d];
      }
    }
    gradient_P2P(src_coord_, src_value_, trg_coord_, trg_value_);
    // compute relative error in L2 norm
    real_t p_norm = 0, p_diff = 0, F_norm = 0, F_diff = 0;
    for(int i=0; i<ntrgs; ++i) {
      int itrg = i*stride;
      p_norm += std::norm(trg_value_[4*i]);
      p_diff += std::norm(trg_value_[4*i]-trg_value[4*itrg]);
      for(int d=1; d<4; ++d) {
        F_norm += std::norm(trg_value_[4*i+d]);
        F_diff += std::norm(trg_value_[4*i+d]-trg_value[4*itrg+d]);
      }
    }
    print("Potential Error", std::sqrt(p_diff/p_norm));
    print("Gradient Error", std::sqrt(F_diff/F_norm));
  }

  extern "C" void print_tree() {
    print_divider("Tree");
    print("Root Center x", X0[0]);
    print("Root Center y", X0[1]);
    print("Root Center z", X0[2]);
    print("Root Radius R", R0);
    print("Tree Depth", MAXLEVEL);
    print("Leaf Nodes", LEAFS.size());
  }
}
