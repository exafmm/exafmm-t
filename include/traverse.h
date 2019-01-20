#ifndef traverse_h
#define traverse_h
#include "exafmm_t.h"
#include "timer.h"

namespace exafmm_t {
  void upward_pass(Nodes& nodes, NodePtrs& leafs) {
    start("P2M");
    P2M(leafs);
    stop("P2M");
    start("M2M");
    #pragma omp parallel
    #pragma omp single nowait
    M2M(&nodes[0]);
    stop("M2M");
  }

  void downward_pass(Nodes& nodes, NodePtrs& leafs) {
    start("P2L");
    P2L(nodes);
    stop("P2L");
    start("M2P");
    M2P(leafs);
    stop("M2P");
    start("P2P");
    P2P(leafs);
    stop("P2P");
    start("M2L");
    M2L(nodes);
    stop("M2L");
    start("L2L");
    #pragma omp parallel
    #pragma omp single nowait
    L2L(&nodes[0]);
    stop("L2L");
    start("L2P");
    L2P(leafs);
    stop("L2P");
  }

  RealVec verify(NodePtrs& leafs) {
    int ntrgs = 10;
    int stride = leafs.size() / ntrgs;
    Nodes targets;
    for(size_t i=0; i<ntrgs; i++) {
      targets.push_back(*(leafs[i*stride]));
    }
    Nodes targets2 = targets;    // used for direct summation
#pragma omp parallel for
    for(size_t i=0; i<targets2.size(); i++) {
      Node *target = &targets2[i];
#if COMPLEX
      std::fill(target->trg_value.begin(), target->trg_value.end(), complex_t(0.,0.));
#else
      std::fill(target->trg_value.begin(), target->trg_value.end(), 0.);
#endif
      for(size_t j=0; j<leafs.size(); j++) {
        gradient_P2P(leafs[j]->src_coord, leafs[j]->src_value, target->trg_coord, target->trg_value);
        // potentialP2P(leafs[j]->src_coord, leafs[j]->src_value, target->trg_coord, target->trg_value);
      }
    }
    real_t p_diff = 0, p_norm = 0, g_diff = 0, g_norm = 0;
    for(size_t i=0; i<targets.size(); i++) {
      if (targets2[i].ntrgs != 0) {  // if current leaf is not empty
#if COMPLEX
        p_norm += std::norm(targets2[i].trg_value[0]);
        p_diff += std::norm(targets2[i].trg_value[0] - targets[i].trg_value[0]);
#else
        p_norm += targets2[i].trg_value[0] * targets2[i].trg_value[0];
        p_diff += (targets2[i].trg_value[0] - targets[i].trg_value[0]) * (targets2[i].trg_value[0] - targets[i].trg_value[0]);
#endif
        for(int d=1; d<4; d++) {
#if COMPLEX
          g_diff += std::norm(targets2[i].trg_value[d] - targets[i].trg_value[d]);
          g_norm += std::norm(targets2[i].trg_value[d]);
#else
          g_diff += (targets2[i].trg_value[d] - targets[i].trg_value[d]) * (targets2[i].trg_value[d] - targets[i].trg_value[d]);
          g_norm += targets2[i].trg_value[d] * targets2[i].trg_value[d];
#endif
        }
      }
    }
    RealVec l2_error(2);
    l2_error[0] = sqrt(p_diff/p_norm);
    l2_error[1] = sqrt(g_diff/g_norm);
    return l2_error;
  }
}//end namespace
#endif
