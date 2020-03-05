#ifndef build_list_h
#define build_list_h
#include "exafmm_t.h"
#include "geometry.h"
#include "fmm_base.h"

namespace exafmm_t {
  /**
   * @brief Given the pointer of a node n, loop over the colleagues of n's parent to
   * build n's P2L_list and P2P_list.
   * 
   * @tparam T Target's value type (real or complex).
   * @param n Node pointer.
   * @param fmm FMM instance.
   * @param skip_P2P Whether to skip P2P interactions.
   */
  template <typename T>
  void build_list_parent_level(Node<T>* n, const FmmBase<T>& fmm, bool skip_P2P=false) {
    if (!n->parent) return;
    ivec3 rel_coord;
    int octant = n->octant;
    bool isleaf = n->is_leaf;
    for (int i=0; i<27; i++) {
      Node<T> * pc = n->parent->colleagues[i];
      if (pc && pc->is_leaf) {
        rel_coord[0]=( i %3)*4-4-(octant & 1?2:0)+1;
        rel_coord[1]=((i/3)%3)*4-4-(octant & 2?2:0)+1;
        rel_coord[2]=((i/9)%3)*4-4-(octant & 4?2:0)+1;
        int c_hash = hash(rel_coord);
        if (isleaf) {
          int idx1 = HASH_LUT[P2P0_Type][c_hash];
          if (idx1>=0 && !skip_P2P)
            n->P2P_list.push_back(pc);
        }
        int idx2 = HASH_LUT[P2L_Type][c_hash];
        if (idx2>=0) {
          if (isleaf && n->ntrgs<=fmm.nsurf)
            n->P2P_list.push_back(pc);
          else
            n->P2L_list.push_back(pc);
        }
      }
    }
  }

  /**
   * @brief Given the pointer of a node n, loop over the colleagues of n to
   * build n's M2L_list and P2P_list.
   * 
   * @tparam T Target's value type (real or complex).
   * @param n Node pointer.
   * @param skip_P2P Whether to skip P2P interactions.
   */
  template <typename T>
  void build_list_current_level(Node<T>* n, bool skip_P2P=false) {
    ivec3 rel_coord;
    bool isleaf = n->is_leaf;
    for (int i=0; i<27; i++) {
      Node<T> * col = n->colleagues[i];
      if(col) {
        rel_coord[0]=( i %3)-1;
        rel_coord[1]=((i/3)%3)-1;
        rel_coord[2]=((i/9)%3)-1;
        int c_hash = hash(rel_coord);
        if (col->is_leaf && isleaf) {
          int idx1 = HASH_LUT[P2P1_Type][c_hash];
          if (idx1>=0 && !skip_P2P)
            n->P2P_list.push_back(col);
        } else if (!col->is_leaf && !isleaf) {
          int idx2 = HASH_LUT[M2L_Type][c_hash];
          if (idx2>=0)
            n->M2L_list[idx2] = col;
        }
      }
    }
  }
  
  /**
   * @brief Given the pointer of a node n, loop over the children of n's colleagues to
   * build n's M2P_list and P2P_list.
   * 
   * @tparam T Target's value type (real or complex).
   * @param n Node pointer.
   * @param fmm FMM instance.
   * @param skip_P2P Whether to skip P2P interactions.
   */
  template <typename T>
  void build_list_child_level(Node<T>* n, const FmmBase<T>& fmm, bool skip_P2P=false) {
    if (!n->is_leaf) return;
    ivec3 rel_coord;
    for(int i=0; i<27; i++) {
      Node<T>* col = n->colleagues[i];
      if(col && !col->is_leaf) {
        for(int j=0; j<NCHILD; j++) {
          Node<T>* cc = col->children[j];
          rel_coord[0]=( i %3)*4-4+(j & 1?2:0)-1;
          rel_coord[1]=((i/3)%3)*4-4+(j & 2?2:0)-1;
          rel_coord[2]=((i/9)%3)*4-4+(j & 4?2:0)-1;
          int c_hash = hash(rel_coord);
          int idx1 = HASH_LUT[P2P2_Type][c_hash];
          int idx2 = HASH_LUT[M2P_Type][c_hash];
          if (idx1>=0 && !skip_P2P) {
            assert(col->children[j]->is_leaf); //2:1 balanced
            n->P2P_list.push_back(cc);
          }
          // since we currently don't save bodies' information in nonleaf nodes
          // M2P can only be switched to P2P when source is leaf
          if (idx2>=0) {
            if (cc->is_leaf && cc->nsrcs<=fmm.nsurf)
              n->P2P_list.push_back(cc);
            else
              n->M2P_list.push_back(cc);
          }
        }
      }
    }
  }

  /**
   * @brief Build interaction lists of each node in a tree.
   * 
   * @tparam T Target's value type (real or complex).
   * @param nodes Vector of nodes that represents an octree.
   * @param fmm FMM instance.
   * @param skip_P2P Whether to skip P2P interactions.
   */
  template <typename T>
  void build_list(Nodes<T>& nodes, const FmmBase<T>& fmm, bool skip_P2P=false) {
    #pragma omp parallel for
    for(size_t i=0; i<nodes.size(); i++) {
      Node<T>* node = &nodes[i];
      node->M2L_list.resize(REL_COORD[M2L_Type].size(), nullptr);
      build_list_parent_level(node, fmm, skip_P2P);   // P2P0 & P2L
      build_list_current_level(node, skip_P2P);  // P2P1 & M2L
#if NON_ADAPTIVE
      if (node->ntrgs)
        build_list_child_level(node, fmm, skip_P2P);  // P2P2 & M2P
#else
      build_list_child_level(node, fmm, skip_P2P);    // P2P2 & M2P
#endif
    }
  }

  /**
   * @brief Set the colleagues of a node and its descendants recursively using a preorder traversal.
   * 
   * @tparam T Target's value type (real or complex)
   * @param node Node pointer.
   */
  template <typename T>
  void set_colleagues(Node<T>* node) {
    Node<T> *parent, *colleague, *child;
    node->colleagues.resize(27, nullptr);
    if (node->level==0) {     // root node
      node->colleagues[13] = node;
    } else {                  // non-root node
      parent = node->parent;
      int l = node->octant;
      for(int i=0; i<27; ++i) { // loop over parent's colleagues
        colleague = parent->colleagues[i]; 
        if(colleague && !colleague->is_leaf) {
          for(int j=0; j<8; ++j) {  // loop over parent's colleages child
            child = colleague->children[j];
            if(child) {
              bool flag = true;
              int a = 1;
              int b = 1;
              int new_idx = 0;
              for (int k=0; k<3; ++k) {
                int idx_diff = (((i/b)%3)-1)*2 + ((j/a)%2) - ((l/a)%2);
                if (-1>idx_diff || idx_diff>1) flag=false;
                new_idx += (idx_diff+1)*b;
                a *= 2;
                b *= 3;
              }
              if(flag)
                node->colleagues[new_idx] = child;
            }
          }
        }
      }
    }
    if (!node->is_leaf) {
      for (int c=0; c<8; ++c) {
        if (node->children[c]) {
          set_colleagues(node->children[c]);
        }
      }
    }
  }

  /**
   * @brief Set the colleagues of each node in a tree.
   * 
   * @tparam T Target's value type (real or complex)
   * @param nodes Vector of nodes that represents an octree.
   */
  template <typename T>
  void set_colleagues(Nodes<T>& nodes) {
    set_colleagues(&nodes[0]);
  }
}
#endif
