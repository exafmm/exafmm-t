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

  // Build interaction lists of P2P0_Type and P2L_Type
  void buildListParentLevel(Node* n) {
    if (n->parent == NULL) return;
    ivec3 rel_coord;
    int octant = n->octant;
    bool isleaf = n->IsLeaf();
    for (int i=0; i<27; i++) {
      Node* pc = n->parent->colleague[i];
      if (pc!=NULL && pc->IsLeaf()) {
        rel_coord[0]=( i %3)*4-4-(octant & 1?2:0)+1;
        rel_coord[1]=((i/3)%3)*4-4-(octant & 2?2:0)+1;
        rel_coord[2]=((i/9)%3)*4-4-(octant & 4?2:0)+1;
        int c_hash = hash(rel_coord);
        if (isleaf) {
          int idx1 = hash_lut[P2P0_Type][c_hash];
          if (idx1>=0)
            n->P2Plist_idx.push_back(pc->idx);
        }
        int idx2 = hash_lut[P2L_Type][c_hash];
        if (idx2>=0) {
          if (isleaf && n->numBodies<=NSURF)
            n->P2Plist_idx.push_back(pc->idx);
          else
            n->P2Llist_idx.push_back(pc->idx);
        }
      }
    }
  }

  // Build interaction lists of P2P1_Type and M2L_Type
  void buildListCurrentLevel(Node* n) {
    ivec3 rel_coord;
    bool isleaf = n->IsLeaf();
    for (int i=0; i<27; i++) {
      Node* col = n->colleague[i];
      if(col!=NULL) {
        rel_coord[0]=( i %3)-1;
        rel_coord[1]=((i/3)%3)-1;
        rel_coord[2]=((i/9)%3)-1;
        int c_hash = hash(rel_coord);
        if (col->IsLeaf() && isleaf) {
          int idx1 = hash_lut[P2P1_Type][c_hash];
          if (idx1>=0) n->P2Plist_idx.push_back(col->idx);
        }
      }
    }
  }
  
  // Build interaction lists of P2P2_Type and M2P_Type
  void buildListChildLevel(Node* n) {
    if (!n->IsLeaf()) return;
    ivec3 rel_coord;
    for(int i=0; i<27; i++) {
      Node* col = n->colleague[i];
      if(col!=NULL && !col->IsLeaf()) {
        for(int j=0; j<NCHILD; j++) {
          Node* cc = col->child[j];
          rel_coord[0]=( i %3)*4-4+(j & 1?2:0)-1;
          rel_coord[1]=((i/3)%3)*4-4+(j & 2?2:0)-1;
          rel_coord[2]=((i/9)%3)*4-4+(j & 4?2:0)-1;
          int c_hash = hash(rel_coord);
          int idx1 = hash_lut[P2P2_Type][c_hash];
          int idx2 = hash_lut[M2P_Type][c_hash];
          if (idx1>=0) {
            assert(col->child[j]->IsLeaf()); //2:1 balanced
            n->P2Plist_idx.push_back(cc->idx);
          }
          // since we currently don't save bodies' information in nonleaf nodes
          // M2P can only be switched to P2P when source is leaf
          if (idx2>=0) {
            if (cc->IsLeaf() && cc->numBodies<=NSURF)
              n->P2Plist_idx.push_back(cc->idx);
            else
              n->M2Plist_idx.push_back(cc->idx);
          }
        }
      }
    }
  }
  
  // Build M2L list
  void buildListM2L(Node* n, std::set<int> &sources_idx, std::set<int> &targets_idx)
  {
    if (!n->parent) return;
    Node* p = n->parent;
    int octant = n->octant;
    ivec3 rel_coord;
    for(int i=0; i<27; i++) {
      Node* pc = p->colleague[i];
      if (pc!=NULL && !pc->IsLeaf() && pc!=p) {
        for (int j=0; j<NCHILD; j++) {
          Node* pcc = pc->child[j];
          if (pcc != NULL) {
            rel_coord[0] = (    i%3 - 1)*2 + (j & 1 ? 0.5 : -0.5) - (octant & 1 ? 0.5 : -0.5);
            rel_coord[1] = ((i/3)%3 - 1)*2 + (j & 2 ? 0.5 : -0.5) - (octant & 2 ? 0.5 : -0.5);
            rel_coord[2] = ((i/9)%3 - 1)*2 + (j & 4 ? 0.5 : -0.5) - (octant & 4 ? 0.5 : -0.5);
            int c_hash = hash(rel_coord);
            int idx = hash_lut[M2L_Helper_Type][c_hash];
            if (idx>=0) {
              n->M2Llist_idx.push_back(pcc->idx);
              n->M2LRelPos.push_back(idx);
              sources_idx.insert(pcc->idx);
            }
          }
        }
      }
    }
    if (n->M2Llist_idx.size()>0)
      targets_idx.insert(n->idx);
  }

  // Build interaction lists for all nodes 
  void buildList(Nodes& nodes, std::vector<int> &M2Lsources_idx, std::vector<int> &M2Ltargets_idx) {
    std::set<int> sources_idx;
    std::set<int> targets_idx;
    #pragma omp parallel
    {
      std::set<int> sources_idx_;  // thread private
      std::set<int> targets_idx_;
      #pragma omp for nowait 
      for(size_t i=0; i<nodes.size(); i++) {
        Node* node = &nodes[i];
        buildListParentLevel(node);
        buildListCurrentLevel(node);
        buildListChildLevel(node);
        buildListM2L(node, sources_idx_, targets_idx_);
      }
      #pragma omp critical
      {
        sources_idx.insert(sources_idx_.begin(), sources_idx_.end());
        targets_idx.insert(targets_idx_.begin(), targets_idx_.end());
      }
    }
    M2Lsources_idx.assign(sources_idx.begin(), sources_idx.end());
    M2Ltargets_idx.assign(targets_idx.begin(), targets_idx.end());
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
