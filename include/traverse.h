#ifndef traverse_h
#define traverse_h
#include "exafmm_t.h"
#include "profile.h"

namespace exafmm_t {
  void upwardPass(Nodes& nodes, NodePtrs& leafs) {
    Profile::Tic("P2M", false, 5);
    P2M(leafs);
    Profile::Toc();
    Profile::Tic("M2M", false, 5);
    #pragma omp parallel
    #pragma omp single nowait
    M2M(&nodes[0]);
    Profile::Toc();
  }

  void downwardPass(Nodes& nodes, NodePtrs& leafs) {
    Profile::Tic("P2L", false, 5);
    P2L(nodes);
    Profile::Toc();
    Profile::Tic("M2P", false, 5);
    M2P(leafs);
    Profile::Toc();
    Profile::Tic("P2P", false, 5);
    P2P(leafs);
    Profile::Toc();
    Profile::Tic("M2L", false, 5);
    M2L(nodes);
    Profile::Toc();
#if 0
  // check level 2 node dnward check after M2L
  Node& node = nodes[9];   // lvl 2, octant=0, parent's octant = 0
  for(int i=0; i<node.dnward_equiv.size(); i++) {
    cout << i << " " << node.dnward_equiv[i] << endl;
  }
#endif
    Profile::Tic("L2L", false, 5);
    #pragma omp parallel
    #pragma omp single nowait
    L2L(&nodes[0]);
    Profile::Toc();
#if 0
  // check level 2 node dnward check after L2L
  Node& node = nodes[9];   // lvl 2, octant=0, parent's octant = 0
  for(int i=0; i<node.dnward_equiv.size(); i++) {
    cout << i << " " << node.dnward_equiv[i] << endl;
  }
#endif
#if 0
  // check target's potential before L2P (after P2P)
  Node& node = nodes[9];   // lvl 2, octant=0, parent's octant = 0
  cout << node.is_leaf << " " << node.numTargets << endl;
  cout << node.numTargets << " " << node.trg_value.size() << endl;
  for(int i=0; i<node.numTargets; i++) {
    cout << i << " " << node.trg_coord[3*i] << " " << node.trg_value[i] << endl;
  }
#endif
    Profile::Tic("L2P", false, 5);
    L2P(leafs);
    Profile::Toc();
#if 0
  // check target's potential after L2P
  Node& node = nodes[9];   // lvl 2, octant=0, parent's octant = 0
  cout << node.is_leaf << " " << node.numTargets << endl;
  cout << node.numTargets << " " << node.trg_value.size() << endl;
  for(int i=0; i<node.numTargets; i++) {
    cout << i << " " << node.trg_value[i] << endl;
  }
#endif
  }

  RealVec verify(NodePtrs& leafs) {
    int numTargets = 10;
    int stride = leafs.size() / numTargets;
    Nodes targets;
    for(size_t i=0; i<numTargets; i++) {
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
        gradientP2P(leafs[j]->src_coord, leafs[j]->src_value, target->trg_coord, target->trg_value);
        // potentialP2P(leafs[j]->src_coord, leafs[j]->src_value, target->trg_coord, target->trg_value);
      }
    }
    real_t p_diff = 0, p_norm = 0, g_diff = 0, g_norm = 0;
    for(size_t i=0; i<targets.size(); i++) {
      if (targets2[i].numTargets != 0) {  // if current leaf is not empty
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
