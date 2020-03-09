#include <cstring>  // std::memset()
#include <fstream>  // std::ifstream
#include <set>      // std::set
#include "laplace.h"
#include "math_wrapper.h"

namespace exafmm_t {
















  RealVec LaplaceFMM::verify(NodePtrs_t& leafs) {
    int ntrgs = 10;
    int stride = leafs.size() / ntrgs;
    Nodes_t targets;
    for(int i=0; i<ntrgs; i++) {
      targets.push_back(*(leafs[i*stride]));
    }
    Nodes_t targets2 = targets;    // used for direct summation
#pragma omp parallel for
    for(size_t i=0; i<targets2.size(); i++) {
      Node_t* target = &targets2[i];
      std::fill(target->trg_value.begin(), target->trg_value.end(), 0.);
      for(size_t j=0; j<leafs.size(); j++) {
        gradient_P2P(leafs[j]->src_coord, leafs[j]->src_value, target->trg_coord, target->trg_value);
      }
    }
    real_t p_diff = 0, p_norm = 0, F_diff = 0, F_norm = 0;
    for(size_t i=0; i<targets.size(); i++) {
      if (targets2[i].ntrgs != 0) {  // if current leaf is not empty
        p_norm += std::norm(targets2[i].trg_value[0]);
        p_diff += std::norm(targets2[i].trg_value[0] - targets[i].trg_value[0]);
        for(int d=1; d<4; d++) {
          F_diff += std::norm(targets2[i].trg_value[d] - targets[i].trg_value[d]);
          F_norm += std::norm(targets2[i].trg_value[d]);
        }
      }
    }
    RealVec rel_error(2);
    rel_error[0] = sqrt(p_diff/p_norm);   // potential error
    rel_error[1] = sqrt(F_diff/F_norm);   // gradient error

    return rel_error;
  }
}  // end namespace exafmm_t
