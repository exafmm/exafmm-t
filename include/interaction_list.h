#ifndef interaction_list
#define interaction_list
#include "exafmm_t.h"

namespace exafmm_t {
  std::vector<std::vector<ivec3>> REL_COORD;
  std::vector<std::vector<int>> hash_lut;     // coord_hash -> index in rel_coord

  //! return x + 10y + 100z + 555
  int hash(ivec3& coord) {
    const int n = 5;
    return ((coord[2]+n) * (2*n) + (coord[1]+n)) *(2*n) + (coord[0]+n);
  }

  void init_rel_coord(int max_r, int min_r, int step, Mat_Type t) {
    const int max_hash = 2000;
    int n1 = (max_r*2)/step+1;
    int n2 = (min_r*2)/step-1;
    int count = n1*n1*n1 - (min_r>0?n2*n2*n2:0);
    hash_lut[t].resize(max_hash, -1);
    for(int k=-max_r; k<=max_r; k+=step) {
      for(int j=-max_r; j<=max_r; j+=step) {
        for(int i=-max_r; i<=max_r; i+=step) {
          if(abs(i)>=min_r || abs(j)>=min_r || abs(k)>=min_r) {
            ivec3 coord;
            coord[0] = i;
            coord[1] = j;
            coord[2] = k;
            REL_COORD[t].push_back(coord);
            hash_lut[t][hash(coord)] = REL_COORD[t].size() - 1;
          }
        }
      }
    }
  }

  void init_rel_coord() {
    REL_COORD.resize(Type_Count);
    hash_lut.resize(Type_Count);
    init_rel_coord(1, 1, 2, M2M_Type);
    init_rel_coord(1, 1, 2, L2L_Type);
    init_rel_coord(3, 3, 2, P2P0_Type);
    init_rel_coord(1, 0, 1, P2P1_Type);
    init_rel_coord(3, 3, 2, P2P2_Type);
    init_rel_coord(3, 2, 1, M2L_Helper_Type);
    init_rel_coord(1, 1, 1, M2L_Type);
    init_rel_coord(5, 5, 2, M2P_Type);
    init_rel_coord(5, 5, 2, P2L_Type);
  }

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
          int idx1 = hash_lut[P2P0_Type][c_hash];
          if (idx1>=0)
            n->P2P_list.push_back(pc);
        }
        int idx2 = hash_lut[P2L_Type][c_hash];
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
          int idx1 = hash_lut[P2P1_Type][c_hash];
          if (idx1>=0) n->P2P_list.push_back(col);
        } else if (!col->is_leaf && !isleaf) {
          int idx2 = hash_lut[M2L_Type][c_hash];
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
          int idx1 = hash_lut[P2P2_Type][c_hash];
          int idx2 = hash_lut[M2P_Type][c_hash];
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
      node->M2L_list.resize(REL_COORD[M2L_Type].size(), 0);
      build_list_parent_level(node);
      build_list_current_level(node);
      build_list_child_level(node);
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
