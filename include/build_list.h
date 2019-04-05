#ifndef build_list_h
#define build_list_h
#include "exafmm_t.h"
#include "geometry.h"

namespace exafmm_t {
  // Build interaction lists of P2P0_Type and P2L_Type
  void build_list_parent_level(Node* n) {
    if (!n->parent) return;
    ivec3 rel_coord;
    int octant = n->octant;
    bool isleaf = n->is_leaf;
    for (int i=0; i<27; i++) {
      Node* pc = n->parent->colleagues[i];
      if (pc && pc->is_leaf) {
        rel_coord[0]=( i %3)*4-4-(octant & 1?2:0)+1;
        rel_coord[1]=((i/3)%3)*4-4-(octant & 2?2:0)+1;
        rel_coord[2]=((i/9)%3)*4-4-(octant & 4?2:0)+1;
        int c_hash = hash(rel_coord);
        if (isleaf) {
          int idx1 = HASH_LUT[P2P0_Type][c_hash];
          if (idx1>=0)
            n->P2P_list.push_back(pc);
        }
        int idx2 = HASH_LUT[P2L_Type][c_hash];
        if (idx2>=0) {
          if (isleaf && n->ntrgs<=NSURF)
            n->P2P_list.push_back(pc);
          else
            n->P2L_list.push_back(pc);
        }
      }
    }
  }

  // Build interaction lists of P2P1_Type and M2L_Type
  void build_list_current_level(Node* n) {
    ivec3 rel_coord;
    bool isleaf = n->is_leaf;
    for (int i=0; i<27; i++) {
      Node* col = n->colleagues[i];
      if(col) {
        rel_coord[0]=( i %3)-1;
        rel_coord[1]=((i/3)%3)-1;
        rel_coord[2]=((i/9)%3)-1;
        int c_hash = hash(rel_coord);
        if (col->is_leaf && isleaf) {
          int idx1 = HASH_LUT[P2P1_Type][c_hash];
          if (idx1>=0) n->P2P_list.push_back(col);
        } else if (!col->is_leaf && !isleaf) {
          int idx2 = HASH_LUT[M2L_Type][c_hash];
          if (idx2>=0) n->M2L_list[idx2] = col;
        }
      }
    }
  }
  
  // Build interaction lists of P2P2_Type and M2P_Type
  void build_list_child_level(Node* n) {
    if (!n->is_leaf) return;
    ivec3 rel_coord;
    for(int i=0; i<27; i++) {
      Node* col = n->colleagues[i];
      if(col && !col->is_leaf) {
        for(int j=0; j<NCHILD; j++) {
          Node* cc = col->children[j];
          rel_coord[0]=( i %3)*4-4+(j & 1?2:0)-1;
          rel_coord[1]=((i/3)%3)*4-4+(j & 2?2:0)-1;
          rel_coord[2]=((i/9)%3)*4-4+(j & 4?2:0)-1;
          int c_hash = hash(rel_coord);
          int idx1 = HASH_LUT[P2P2_Type][c_hash];
          int idx2 = HASH_LUT[M2P_Type][c_hash];
          if (idx1>=0) {
            assert(col->children[j]->is_leaf); //2:1 balanced
            n->P2P_list.push_back(cc);
          }
          // since we currently don't save bodies' information in nonleaf nodes
          // M2P can only be switched to P2P when source is leaf
          if (idx2>=0) {
            if (cc->is_leaf && cc->nsrcs<=NSURF)
              n->P2P_list.push_back(cc);
            else
              n->M2P_list.push_back(cc);
          }
        }
      }
    }
  }

  // Build interaction lists for all nodes 
  void build_list(Nodes& nodes) {
    #pragma omp parallel for
    for(size_t i=0; i<nodes.size(); i++) {
      Node* node = &nodes[i];
      node->M2L_list.resize(REL_COORD[M2L_Type].size(), nullptr);
      build_list_parent_level(node);   // P2P0 & P2L
      build_list_current_level(node);  // P2P1 & M2L
#if FULL_TREE
      if (node->ntrgs)
        build_list_child_level(node);  // P2P2 & M2P
#else
      build_list_child_level(node);    // P2P2 & M2P
#endif
    }
  }
  
  void set_colleagues(Node* node) {
    Node *parent, *colleague, *child;
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
              bool flag=true;
              int a=1, b=1, new_indx=0;
              for(int k=0; k<3; ++k) {
                int indx_diff=(((i/b)%3)-1)*2+((j/a)%2)-((l/a)%2);
                if(-1>indx_diff || indx_diff>1) flag=false;
                new_indx+=(indx_diff+1)*b;
                a*=2;
                b*=3;
              }
              if(flag)
                node->colleagues[new_indx] = child;
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

  void set_colleagues(Nodes& nodes) {
    set_colleagues(&nodes[0]);
  }
}
#endif
