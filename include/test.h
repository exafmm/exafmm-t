#ifndef test_h
#define test_h
#include <cassert>
#include "build_list.h"
#include "exafmm_t.h"
#include "fmm_base.h"

namespace exafmm_t {
  //! A derived FMM class, assuming that all sources have a unit charge. Kernel functions only compute monopoles. This is used for testing tree and list construction.
  template <typename T>
  class DummyFmm : public FmmBase<T> {
  public:
    DummyFmm() {}
    DummyFmm(int ncrit_) { this->p = 1; this->nsurf = 1; this->ncrit = ncrit_;}

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
    void P2M(NodePtrs<T>& leafs) {
#pragma omp parallel for
      for(size_t i=0; i<leafs.size(); ++i) {
        Node<T>* leaf = leafs[i];
        leaf->up_equiv[0] += leaf->nsrcs;
      }
    }

    //! Dummy M2M operator.
    void M2M(Node<T>* node) {
      if(node->is_leaf) return;
      for(int octant=0; octant<8; ++octant) {  
        if(node->children[octant])
#pragma omp task untied
          M2M(node->children[octant]);
      }
#pragma omp taskwait
      for(int octant=0; octant<8; octant++) {
        if(node->children[octant]) {
          Node<T>* child = node->children[octant];
          node->up_equiv[0] += child->up_equiv[0];
        }
      }
    }

    //! Dummy M2L operator.
    void M2L(NodePtrs<T>& nonleafs) {
#pragma omp parallel for schedule(dynamic)
      for (size_t i=0; i<nonleafs.size(); ++i) {
        Node<T>* trg_parent = nonleafs[i];
        NodePtrs<T>& M2L_list = trg_parent->M2L_list;
        for (size_t j=0; j<M2L_list.size(); ++j) {
          Node<T>* src_parent = M2L_list[j];
          if (src_parent) {
            for (int src_octant=0; src_octant<8; ++src_octant) {
              Node<T>* src_child = src_parent->children[src_octant];
              for (int trg_octant=0; trg_octant<8; ++trg_octant) {
                Node<T>* trg_child = trg_parent->children[trg_octant];
                if (!is_adjacent(src_child->key, trg_child->key)) {
                  trg_child->dn_equiv[0] += src_child->up_equiv[0];
                }
              }
            }
          }
        }
      }
    }

    //! Dummy P2L operator.
    void P2L(Nodes<T>& nodes) {
#pragma omp parallel for schedule(dynamic)
      for(size_t i=0; i<nodes.size(); ++i) {
        NodePtrs<T>& P2L_list = nodes[i].P2L_list;
        for(size_t j=0; j<P2L_list.size(); ++j) {
          nodes[i].dn_equiv[0] += P2L_list[j]->nsrcs;
        }
      }
    }

    //! Dummy M2P operator.
    void M2P(NodePtrs<T>& leafs) {
#pragma omp parallel for schedule(dynamic)
      for(size_t i=0; i<leafs.size(); ++i) {
        if (leafs[i]->ntrgs == 0) continue;
        NodePtrs<T>& M2P_list = leafs[i]->M2P_list;
        for (size_t j=0; j<M2P_list.size(); ++j) {
          leafs[i]->trg_value[0] += M2P_list[j]->up_equiv[0];
        }
      }
    }

    //! Dummy L2L operator.
    void L2L(Node<T>* node) {
      if(node->is_leaf) return;
      for(int octant=0; octant<8; octant++) {
        if(node->children[octant]) {
          Node<T>* child = node->children[octant];
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
    void L2P(NodePtrs<T>& leafs) {
#pragma omp parallel for
      for(size_t i=0; i<leafs.size(); ++i) {
        Node<T>* leaf = leafs[i];
        if (leaf->ntrgs==0) continue;
        leaf->trg_value[0] += leaf->dn_equiv[0];
      }
    }

    //! Dummy P2P operator.
    void P2P(NodePtrs<T>& leafs) {
#pragma omp parallel for schedule(dynamic)
      for(size_t i=0; i<leafs.size(); ++i) {
        Node<T>* leaf = leafs[i];
        if (leaf->ntrgs==0) continue;
        NodePtrs<T>& P2P_list = leaf->P2P_list;
        for(size_t j=0; j<P2P_list.size(); ++j) {
          leaf->trg_value[0] += P2P_list[j]->nsrcs;
        }
      }
    }

    // below are the virtual methods define in FmmBase class
    void potential_P2P(RealVec& src_coord, std::vector<T>& src_value,
                       RealVec& trg_coord, std::vector<T>& trg_value) {}


    void gradient_P2P(RealVec& src_coord, std::vector<T>& src_value,
                      RealVec& trg_coord, std::vector<T>& trg_value) {}

    void M2L(Nodes<T>& nodes) {}
  };

  /**
   * @brief A helper function to build the tree needed in kernel test.
   *
   * @tparam T Real or complex type.
   * @param parent Pointer to parent node.
   * @param first_child Pointer to first child node.
   */
  template <typename T>
  void set_children(Node<T>* parent, Node<T>* first_child) {
    parent->is_leaf = false;
    for (int octant=0; octant<8; ++octant) {
      Node<T>* child = first_child + octant;
      child->octant = octant;
      child->parent = parent;
      child->level = parent->level + 1;
      child->x = parent->x;
      child->r = parent->r / 2;
      for (int d=0; d<3; d++) {
        child->x[d] += child->r * (((octant & 1 << d) >> d) * 2 - 1);
      }
      parent->children.push_back(child);
    }
  }
}// end namespace
#endif
