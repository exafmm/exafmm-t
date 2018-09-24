#ifndef interaction_list
#define interaction_list
#include "exafmm_t.h"

namespace exafmm_t {
  std::vector<std::vector<ivec3>> rel_coord;
  std::vector<std::vector<int>> hash_lut;     // coord_hash -> index in rel_coord

  //! return x + 10y + 100z + 555
  int hash(ivec3& coord) {
    const int n = 5;
    return ((coord[2]+n) * (2*n) + (coord[1]+n)) *(2*n) + (coord[0]+n);
  }

  void initRelCoord(int max_r, int min_r, int step, Mat_Type t) {
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
            rel_coord[t].push_back(coord);
            hash_lut[t][hash(coord)] = rel_coord[t].size() - 1;
          }
        }
      }
    }
  }

  void initRelCoord() {
    rel_coord.resize(Type_Count);
    hash_lut.resize(Type_Count);
    initRelCoord(1, 1, 2, M2M_Type);
    initRelCoord(1, 1, 2, L2L_Type);
    initRelCoord(3, 3, 2, P2P0_Type);
    initRelCoord(1, 0, 1, P2P1_Type);
    initRelCoord(3, 3, 2, P2P2_Type);
    initRelCoord(3, 2, 1, M2L_Helper_Type);
    initRelCoord(1, 1, 1, M2L_Type);
    initRelCoord(5, 5, 2, M2P_Type);
    initRelCoord(5, 5, 2, P2L_Type);
  }

  // Build interaction list for node n
  void buildList(Node* n, Mat_Type t) {
    const int n_child=8, n_collg=27;
    int c_hash, idx;
    ivec3 rel_coord;
    int p2n = n->octant;       // octant
    Node* p = n->parent; // parent node
    std::vector<Node*>& interac_list = n->interac_list[t];
    switch (t) {
    case P2P0_Type:
      if(p == NULL || !n->IsLeaf()) return;
      for(int i=0; i<n_collg; i++) {
        Node* pc = p->colleague[i];
        if(pc!=NULL && pc->IsLeaf()) {
          rel_coord[0]=( i %3)*4-4-(p2n & 1?2:0)+1;
          rel_coord[1]=((i/3)%3)*4-4-(p2n & 2?2:0)+1;
          rel_coord[2]=((i/9)%3)*4-4-(p2n & 4?2:0)+1;
          c_hash = hash(rel_coord);
          idx = hash_lut[t][c_hash];
          if(idx>=0)
            n->P2Plist.push_back(pc);
        }
      }
      break;
    case P2P1_Type:
      if(!n->IsLeaf()) return;
      for(int i=0; i<n_collg; i++) {
        Node* col=(Node*)n->colleague[i];
        if(col!=NULL && col->IsLeaf()) {
          rel_coord[0]=( i %3)-1;
          rel_coord[1]=((i/3)%3)-1;
          rel_coord[2]=((i/9)%3)-1;
          c_hash = hash(rel_coord);
          idx = hash_lut[t][c_hash];
          if(idx>=0)
            n->P2Plist.push_back(col);
        }
      }
      break;
    case P2P2_Type:
      if(!n->IsLeaf()) return;
      for(int i=0; i<n_collg; i++) {
        Node* col=(Node*)n->colleague[i];
        if(col!=NULL && !col->IsLeaf()) {
          for(int j=0; j<n_child; j++) {
            rel_coord[0]=( i %3)*4-4+(j & 1?2:0)-1;
            rel_coord[1]=((i/3)%3)*4-4+(j & 2?2:0)-1;
            rel_coord[2]=((i/9)%3)*4-4+(j & 4?2:0)-1;
            c_hash = hash(rel_coord);
            idx = hash_lut[t][c_hash];
            if(idx>=0) {
              assert(col->Child(j)->IsLeaf()); //2:1 balanced
              n->P2Plist.push_back(col->Child(j));
            }
          }
        }
      }
      break;
    case M2L_Type:
      if(n->IsLeaf()) return;
      for(int i=0; i<n_collg; i++) {
        Node* col=(Node*)n->colleague[i];
        if(col!=NULL && !col->IsLeaf()) {
          rel_coord[0]=( i %3)-1;
          rel_coord[1]=((i/3)%3)-1;
          rel_coord[2]=((i/9)%3)-1;
          c_hash = hash(rel_coord);
          idx=hash_lut[t][c_hash];
          if(idx>=0) interac_list[idx]=col;
        }
      }
      break;
    case M2P_Type:
      if(!n->IsLeaf()) return;
      for(int i=0; i<n_collg; i++) {
        Node* col=(Node*)n->colleague[i];
        if(col!=NULL && !col->IsLeaf()) {
          for(int j=0; j<n_child; j++) {
            rel_coord[0]=( i %3)*4-4+(j & 1?2:0)-1;
            rel_coord[1]=((i/3)%3)*4-4+(j & 2?2:0)-1;
            rel_coord[2]=((i/9)%3)*4-4+(j & 4?2:0)-1;
            c_hash = hash(rel_coord);
            idx=hash_lut[t][c_hash];
            if(idx>=0) 
              n->M2Plist.push_back(col->Child(j));
          }
        }
      }
      break;
    case P2L_Type:
      if(p == NULL) return;
      for(int i=0; i<n_collg; i++) {
        Node* pc=(Node*)p->colleague[i];
        if(pc!=NULL && pc->IsLeaf()) {
          rel_coord[0]=( i %3)*4-4-(p2n & 1?2:0)+1;
          rel_coord[1]=((i/3)%3)*4-4-(p2n & 2?2:0)+1;
          rel_coord[2]=((i/9)%3)*4-4-(p2n & 4?2:0)+1;
          c_hash = hash(rel_coord);
          idx=hash_lut[t][c_hash];
          if(idx>=0)
            n->P2Llist.push_back(pc);
        }
      }
      break;
    default:
      abort();
    }
  }

  // Fill in interac_list of all nodes, assume sources == target for simplicity
  void buildList(Nodes& nodes) {
    std::vector<Mat_Type> interactionTypes = {P2P0_Type, P2P1_Type, P2P2_Type,
                                              M2P_Type, P2L_Type, M2L_Type};
    for(int j=0; j<interactionTypes.size(); j++) {
      Mat_Type type = interactionTypes[j];
      int numRelCoord = rel_coord[type].size();
      #pragma omp parallel for
      for(size_t i=0; i<nodes.size(); i++) {
        Node* node = &nodes[i];
        node->interac_list[type].resize(numRelCoord, 0);
        buildList(node, type);
      }
    }
  }
  
  void setColleagues(Node* node) {
    Node *parent, *colleague, *child;
    for (int i=0; i<27; ++i) node->colleague[i] = NULL;
    if (node->depth==0) {     // root node
      node->colleague[13] = node;
    } else {                  // non-root node
      parent = node->parent;
      int l = node->octant;
      for(int i=0; i<27; ++i) { // loop over parent's colleagues
        colleague = parent->colleague[i]; 
        if(colleague!=NULL && !colleague->IsLeaf()) {
          for(int j=0; j<8; ++j) {  // loop over parent's colleages child
            child = colleague->Child(j);
            if(child!=NULL) {
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
                node->colleague[new_indx] = child;
            }
          }
        }
      }
    }
    if (!node->IsLeaf()) {
      for (int c=0; c<8; ++c) {
        if (node->child[c] != NULL) {
          setColleagues(node->child[c]);
        }
      }
    }
  }

  void setColleagues(Nodes& nodes) {
    setColleagues(&nodes[0]);
  }
}
#endif
