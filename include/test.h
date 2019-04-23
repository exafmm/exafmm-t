#ifndef test_h
#define test_h
#include <cassert>
#include "exafmm_t.h"

namespace exafmm_t {
  /**
   * @brief Given the octant of a node, return a triplet's relative postion to its parent
   *
   * @param octant Octant of a node, integer from 0 to 7
   * 
   * @return Triplet of node's relative position to its parent, each element is -1 or 1
   */
  ivec3 octant2coord(int octant) {
    ivec3 rel_coord;
    rel_coord[0] = octant & 1 ? 1 : -1;
    rel_coord[1] = octant & 2 ? 1 : -1;
    rel_coord[2] = octant & 4 ? 1 : -1;
    return rel_coord;
  } 

  /** 
   * @brief Dummy P2M kernel, by assuming unit charges for all sources, monopole of a leaf equals to number of sources in the leaf
   *
   * @param leafs Vector of pointers to leafs
   */
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

  void M2L(NodePtrs& nonleafs) {
#pragma omp parallel for schedule(dynamic)
    for(size_t i=0; i<nonleafs.size(); ++i) {
      Node* trg_parent = nonleafs[i];
      NodePtrs& M2L_list = trg_parent->M2L_list;
      for(size_t j=0; j<M2L_list.size(); ++j) {
        Node* src_parent = M2L_list[j];
        if(src_parent) {
          // find src_parent's relative position to trg_parent
          NodePtrs::iterator it = std::find(trg_parent->colleagues.begin(),
                                            trg_parent->colleagues.end(), src_parent);
          assert(it != trg_parent->colleagues.end());   // src_parent has to be trg_parent's colleague
          int colleague_idx = it - trg_parent->colleagues.begin();
          ivec3 parent_rel_coord;
          parent_rel_coord[0] = colleague_idx % 3 - 1;
          parent_rel_coord[1] = (colleague_idx/3) % 3 - 1;
          parent_rel_coord[2] = (colleague_idx/9) % 3 - 1;
          for(int src_octant=0; src_octant<8; ++src_octant) {
            Node* src_child = src_parent->children[src_octant];
            ivec3 src_child_coord = octant2coord(src_octant);
            for(int trg_octant=0; trg_octant<8; ++trg_octant) {
              Node* trg_child = trg_parent->children[trg_octant];
              ivec3 trg_child_coord = octant2coord(trg_octant);
              if(src_child && trg_child) {
                ivec3 rel_coord = parent_rel_coord*2 + (src_child_coord - trg_child_coord) / 2;  // calculate relative coords between children
                bool is_colleague = true;
                for(int d=0; d<3; ++d) {
                  if(rel_coord[d]>1 || rel_coord[d]<-1)
                    is_colleague = false;
                }
                if(!is_colleague)  // perform M2L if they are not neighbors
                  trg_child->dn_equiv[0] += src_child->up_equiv[0];
              }
            }
          }
        }
      }
    }
  }

  void P2L(Nodes& nodes) {
#pragma omp parallel for schedule(dynamic)
    for(size_t i=0; i<nodes.size(); ++i) {
      NodePtrs& P2L_list = nodes[i].P2L_list;
      for(size_t j=0; j<P2L_list.size(); ++j) {
        nodes[i].dn_equiv[0] += P2L_list[j]->nsrcs;
      }
    }
  }

  void M2P(NodePtrs& leafs) {
#pragma omp parallel for schedule(dynamic)
    for(size_t i=0; i<leafs.size(); ++i) {
      NodePtrs& M2P_list = leafs[i]->M2P_list;
      for(size_t j=0; j<M2P_list.size(); ++j) {
        leafs[i]->trg_value[0] += M2P_list[j]->up_equiv[0];
      }
    }
  }

  void L2L(Node* node) {
    if(node->is_leaf) return;
    for(int octant=0; octant<8; octant++) {
      if(node->children[octant]) {
        Node* child = node->children[octant];
        child->dn_equiv[0] += node->dn_equiv[0];
      }
    }
    for(int octant=0; octant<8; ++octant) {
      if(node->children[octant])
#pragma omp task untied
        L2L(node->children[octant]);
    }
#pragma omp taskwait
  }

  void L2P(NodePtrs& leafs) {
#pragma omp parallel for
    for(size_t i=0; i<leafs.size(); ++i) {
      Node* leaf = leafs[i];
      leaf->trg_value[0] += leaf->dn_equiv[0];
    }
  }

  void P2P(NodePtrs& leafs) {
#pragma omp parallel for schedule(dynamic)
    for(size_t i=0; i<leafs.size(); ++i) {
      Node* leaf = leafs[i];
      NodePtrs& P2P_list = leaf->P2P_list;
      for(size_t j=0; j<P2P_list.size(); ++j) {
        leaf->trg_value[0] += P2P_list[j]->nsrcs;
      }
    }
  }
}// end namespace
#endif
