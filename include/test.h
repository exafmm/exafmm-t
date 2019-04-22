#ifndef test_h
#define test_h
#include "exafmm_t.h"

namespace exafmm_t {
  void P2M(NodePtrs& leafs) {
#pragma omp parallel for
    for(size_t i=0; i<leafs.size(); ++i) {
      Node* leaf = leafs[i];
      leaf->up_equiv[0] += leaf->nsrcs;
    }
  }

  void M2M(Node* node) {
    if(node->is_leaf) return;
    for(int octant=0; octant<8; ++octant) {  
      if(node->children[octant])
#pragma omp task untied
        M2M(node->children[octant]);
    }
#pragma omp taskwait
    for(int octant=0; octant<8; octant++) {
      if(node->children[octant]) {
        Node* child = node->children[octant];
        node->up_equiv[0] += child->up_equiv[0];
      }
    }
  }
}  // end namespace
#endif
