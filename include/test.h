#ifndef test_h
#define test_h
#include <cassert>
#include "exafmm_t.h"

namespace exafmm_t {
  //! A derived FMM class, assuming that all sources have a unit charge. Kernel functions only compute monopoles. This is used for testing tree and list construction.
  class TestFMM : public FMM {
    using Body_t = Body<real_t>;
    using Bodies_t = Bodies<real_t>;
    using Node_t = Node<real_t>;
    using Nodes_t = Nodes<real_t>;
    using NodePtrs_t = NodePtrs<real_t>;
 
  public:
    TestFMM() {}
    TestFMM(int ncrit_) { p = 1; nsurf = 1; ncrit = ncrit_;}

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

    //! Dummy P2M operator.
    void P2M(NodePtrs_t& leafs) {
#pragma omp parallel for
      for(size_t i=0; i<leafs.size(); ++i) {
        Node_t* leaf = leafs[i];
        leaf->up_equiv[0] += leaf->nsrcs;
      }
    }

    //! Dummy M2M operator.
    void M2M(Node_t* node) {
      if(node->is_leaf) return;
      for(int octant=0; octant<8; ++octant) {  
        if(node->children[octant])
#pragma omp task untied
          M2M(node->children[octant]);
      }
#pragma omp taskwait
      for(int octant=0; octant<8; octant++) {
        if(node->children[octant]) {
          Node_t* child = node->children[octant];
          node->up_equiv[0] += child->up_equiv[0];
        }
      }
    }

    //! Dummy M2L operator.
    void M2L(NodePtrs_t& nonleafs) {
#pragma omp parallel for schedule(dynamic)
      for(size_t i=0; i<nonleafs.size(); ++i) {
        Node_t* trg_parent = nonleafs[i];
        NodePtrs_t& M2L_list = trg_parent->M2L_list;
        for(size_t j=0; j<M2L_list.size(); ++j) {
          Node_t* src_parent = M2L_list[j];
          if(src_parent) {
            // find src_parent's relative position to trg_parent
            NodePtrs_t::iterator it = std::find(trg_parent->colleagues.begin(),
                                              trg_parent->colleagues.end(), src_parent);
            assert(it != trg_parent->colleagues.end());   // src_parent has to be trg_parent's colleague
            int colleague_idx = it - trg_parent->colleagues.begin();
            ivec3 parent_rel_coord;
            parent_rel_coord[0] = colleague_idx % 3 - 1;
            parent_rel_coord[1] = (colleague_idx/3) % 3 - 1;
            parent_rel_coord[2] = (colleague_idx/9) % 3 - 1;
            for(int src_octant=0; src_octant<8; ++src_octant) {
              Node_t* src_child = src_parent->children[src_octant];
              ivec3 src_child_coord = octant2coord(src_octant);
              for(int trg_octant=0; trg_octant<8; ++trg_octant) {
                Node_t* trg_child = trg_parent->children[trg_octant];
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

    //! Dummy P2L operator.
    void P2L(Nodes_t& nodes) {
#pragma omp parallel for schedule(dynamic)
      for(size_t i=0; i<nodes.size(); ++i) {
        NodePtrs_t& P2L_list = nodes[i].P2L_list;
        for(size_t j=0; j<P2L_list.size(); ++j) {
          nodes[i].dn_equiv[0] += P2L_list[j]->nsrcs;
        }
      }
    }

    //! Dummy M2P operator.
    void M2P(NodePtrs_t& leafs) {
#pragma omp parallel for schedule(dynamic)
      for(size_t i=0; i<leafs.size(); ++i) {
        NodePtrs_t& M2P_list = leafs[i]->M2P_list;
        for(size_t j=0; j<M2P_list.size(); ++j) {
          leafs[i]->trg_value[0] += M2P_list[j]->up_equiv[0];
        }
      }
    }

    //! Dummy L2L operator.
    void L2L(Node_t* node) {
      if(node->is_leaf) return;
      for(int octant=0; octant<8; octant++) {
        if(node->children[octant]) {
          Node_t* child = node->children[octant];
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

    //! Dummy L2P operator.
    void L2P(NodePtrs_t& leafs) {
#pragma omp parallel for
      for(size_t i=0; i<leafs.size(); ++i) {
        Node_t* leaf = leafs[i];
        leaf->trg_value[0] += leaf->dn_equiv[0];
      }
    }

    //! Dummy P2P operator.
    void P2P(NodePtrs_t& leafs) {
#pragma omp parallel for schedule(dynamic)
      for(size_t i=0; i<leafs.size(); ++i) {
        Node_t* leaf = leafs[i];
        NodePtrs_t& P2P_list = leaf->P2P_list;
        for(size_t j=0; j<P2P_list.size(); ++j) {
          leaf->trg_value[0] += P2P_list[j]->nsrcs;
        }
      }
    }
  };
}// end namespace
#endif
