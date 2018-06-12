#ifndef _PVFMM_INTERAC_LIST_HPP_
#define _PVFMM_INTERAC_LIST_HPP_
#include "pvfmm.h"

namespace pvfmm {
  //! return x + 10y + 100z + 555
  int coord_hash(int* c) {
    const int n=5;
    return ( (c[2]+n) * (2*n) + (c[1]+n) ) *(2*n) + (c[0]+n);
  }

  //! swap x,y,z so that |z|>|y|>|x|, return hash of new coord
  int class_hash(int* c_) {
    int c[3]= {abs(c_[0]), abs(c_[1]), abs(c_[2])};
    if(c[1]>c[0] && c[1]>c[2]) {
      int tmp=c[0];
      c[0]=c[1];
      c[1]=tmp;
    }
    if(c[0]>c[2]) {
      int tmp=c[0];
      c[0]=c[2];
      c[2]=tmp;
    }
    if(c[0]>c[1]) {
      int tmp=c[0];
      c[0]=c[1];
      c[1]=tmp;
    }
    assert(c[0]<=c[1] && c[1]<=c[2]);
    return coord_hash(&c[0]);
  }

  void InitList(int max_r, int min_r, int step, Mat_Type t) {
    const int max_hash = 2000;
    int n1 = (max_r*2)/step+1;
    int n2 = (min_r*2)/step-1;
    int count=n1*n1*n1-(min_r>0?n2*n2*n2:0);
    std::vector<ivec3>& M=rel_coord[t];
    M.resize(count);
    hash_lut[t].assign(max_hash, -1);
    std::vector<int> class_size_hash(max_hash, 0);
    for(int k=-max_r; k<=max_r; k+=step)
      for(int j=-max_r; j<=max_r; j+=step)
        for(int i=-max_r; i<=max_r; i+=step)
          if(abs(i)>=min_r || abs(j)>=min_r || abs(k) >= min_r) {
            int c[3]= {i, j, k};
            // count the number of coords of the same class
            // ex. (-1,-1,2) is in the same class as (-2,-1,1)
            class_size_hash[class_hash(c)]++;
          }
    // class count -> class count displacement
    std::vector<int> class_disp_hash(max_hash, 0);
    for(int i=1; i<max_hash; i++) {
      class_disp_hash[i] = class_disp_hash[i-1] + class_size_hash[i-1];
    }

    int count_=0;
    for(int k=-max_r; k<=max_r; k+=step)
      for(int j=-max_r; j<=max_r; j+=step)
        for(int i=-max_r; i<=max_r; i+=step)
          if(abs(i)>=min_r || abs(j)>=min_r || abs(k) >= min_r) {
            int c[3]= {i, j, k};
            int& idx=class_disp_hash[class_hash(c)]; // idx is the displ of current class
            for(int l=0; l<3; l++) M[idx][l]=c[l]; // store the sorted coords
            hash_lut[t][coord_hash(c)]=idx;          // store mapping: hash -> index in rel_coord
            count_++;
            idx++;
          }
    assert(count_==count);
    interac_class[t].resize(count);
    perm_list[t].resize(count);
    for(int j=0; j<count; j++) {      // j is now the index of sorted rel_coord
      if(M[j][0]<0) perm_list[t][j].push_back(ReflecX);
      if(M[j][1]<0) perm_list[t][j].push_back(ReflecY);
      if(M[j][2]<0) perm_list[t][j].push_back(ReflecZ);
      int coord[3];
      coord[0]=abs(M[j][0]);
      coord[1]=abs(M[j][1]);
      coord[2]=abs(M[j][2]);
      if(coord[1]>coord[0] && coord[1]>coord[2]) {
        perm_list[t][j].push_back(SwapXY);
        int tmp=coord[0];
        coord[0]=coord[1];
        coord[1]=tmp;
      }
      if(coord[0]>coord[2]) {
        perm_list[t][j].push_back(SwapXZ);
        int tmp=coord[0];
        coord[0]=coord[2];
        coord[2]=tmp;
      }
      if(coord[0]>coord[1]) {
        perm_list[t][j].push_back(SwapXY);
        int tmp=coord[0];
        coord[0]=coord[1];
        coord[1]=tmp;
      }
      assert(coord[0]<=coord[1] && coord[1]<=coord[2]);
      int c_hash = coord_hash(&coord[0]);
      interac_class[t][j]=hash_lut[t][c_hash];  // j-th rel_coord -> abs coord of the same class
    }
  }

  void InitAll() {
    interac_class.resize(Type_Count);
    perm_list.resize(Type_Count);
    rel_coord.resize(Type_Count);
    hash_lut.resize(Type_Count);
    InitList(0, 0, 1, M2M_V_Type);
    InitList(0, 0, 1, M2M_U_Type);
    InitList(0, 0, 1, L2L_V_Type);
    InitList(0, 0, 1, L2L_U_Type);
    InitList(1, 1, 2, M2M_Type); // count = 8, (+1 or -1)
    InitList(1, 1, 2, L2L_Type);
    InitList(3, 3, 2, P2P0_Type);  // count = 4^3-2^3 = 56
    InitList(1, 0, 1, P2P1_Type);
    InitList(3, 3, 2, P2P2_Type);
    InitList(3, 2, 1, M2L_Helper_Type);
    InitList(1, 1, 1, M2L_Type);
    InitList(5, 5, 2, M2P_Type);
    InitList(5, 5, 2, P2L_Type);
  }

  // Build t-type interaction list for node n
  void BuildList(FMM_Node* n, Mat_Type t) {
    const int n_child=8, n_collg=27;
    int c_hash, idx, rel_coord[3];
    int p2n = n->octant;       // octant
    FMM_Node* p = n->parent; // parent node
    std::vector<FMM_Node*>& interac_list = n->interac_list[t];
    switch (t) {
    case P2P0_Type:
      if(p == NULL || !n->IsLeaf()) return;
      for(int i=0; i<n_collg; i++) {
        FMM_Node* pc = p->colleague[i];
        if(pc!=NULL && pc->IsLeaf()) {
          rel_coord[0]=( i %3)*4-4-(p2n & 1?2:0)+1;
          rel_coord[1]=((i/3)%3)*4-4-(p2n & 2?2:0)+1;
          rel_coord[2]=((i/9)%3)*4-4-(p2n & 4?2:0)+1;
          c_hash = coord_hash(rel_coord);
          idx = hash_lut[t][c_hash];
          if(idx>=0) interac_list[idx] = pc;
        }
      }
      break;
    case P2P1_Type:
      if(!n->IsLeaf()) return;
      for(int i=0; i<n_collg; i++) {
        FMM_Node* col=(FMM_Node*)n->colleague[i];
        if(col!=NULL && col->IsLeaf()) {
          rel_coord[0]=( i %3)-1;
          rel_coord[1]=((i/3)%3)-1;
          rel_coord[2]=((i/9)%3)-1;
          c_hash = coord_hash(rel_coord);
          idx = hash_lut[t][c_hash];
          if(idx>=0) interac_list[idx] = col;
        }
      }
      break;
    case P2P2_Type:
      if(!n->IsLeaf()) return;
      for(int i=0; i<n_collg; i++) {
        FMM_Node* col=(FMM_Node*)n->colleague[i];
        if(col!=NULL && !col->IsLeaf()) {
          for(int j=0; j<n_child; j++) {
            rel_coord[0]=( i %3)*4-4+(j & 1?2:0)-1;
            rel_coord[1]=((i/3)%3)*4-4+(j & 2?2:0)-1;
            rel_coord[2]=((i/9)%3)*4-4+(j & 4?2:0)-1;
            c_hash = coord_hash(rel_coord);
            idx = hash_lut[t][c_hash];
            if(idx>=0) {
              assert(col->Child(j)->IsLeaf()); //2:1 balanced
              interac_list[idx] = (FMM_Node*)col->Child(j);
            }
          }
        }
      }
      break;
    case M2L_Type:
      if(n->IsLeaf()) return;
      for(int i=0; i<n_collg; i++) {
        FMM_Node* col=(FMM_Node*)n->colleague[i];
        if(col!=NULL && !col->IsLeaf()) {
          rel_coord[0]=( i %3)-1;
          rel_coord[1]=((i/3)%3)-1;
          rel_coord[2]=((i/9)%3)-1;
          c_hash = coord_hash(rel_coord);
          idx=hash_lut[t][c_hash];
          if(idx>=0) interac_list[idx]=col;
        }
      }
      break;
    case M2P_Type:
      if(!n->IsLeaf()) return;
      for(int i=0; i<n_collg; i++) {
        FMM_Node* col=(FMM_Node*)n->colleague[i];
        if(col!=NULL && !col->IsLeaf()) {
          for(int j=0; j<n_child; j++) {
            rel_coord[0]=( i %3)*4-4+(j & 1?2:0)-1;
            rel_coord[1]=((i/3)%3)*4-4+(j & 2?2:0)-1;
            rel_coord[2]=((i/9)%3)*4-4+(j & 4?2:0)-1;
            c_hash = coord_hash(rel_coord);
            idx=hash_lut[t][c_hash];
            if(idx>=0) interac_list[idx]=(FMM_Node*)col->Child(j);
          }
        }
      }
      break;
    case P2L_Type:
      if(p == NULL) return;
      for(int i=0; i<n_collg; i++) {
        FMM_Node* pc=(FMM_Node*)p->colleague[i];
        if(pc!=NULL && pc->IsLeaf()) {
          rel_coord[0]=( i %3)*4-4-(p2n & 1?2:0)+1;
          rel_coord[1]=((i/3)%3)*4-4-(p2n & 2?2:0)+1;
          rel_coord[2]=((i/9)%3)*4-4-(p2n & 4?2:0)+1;
          c_hash = coord_hash(rel_coord);
          idx=hash_lut[t][c_hash];
          if(idx>=0) interac_list[idx]=pc;
        }
      }
      break;
    default:
      abort();
    }
  }

  // Fill in interac_list of all nodes, assume sources == target for simplicity
  void BuildInteracLists(FMM_Nodes& cells) {
    std::vector<Mat_Type> interactionTypes = {P2P0_Type, P2P1_Type, P2P2_Type,
                                              M2P_Type, P2L_Type, M2L_Type};
    for(int j=0; j<interactionTypes.size(); j++) {
      Mat_Type type = interactionTypes[j];
      int numRelCoord = rel_coord[type].size();  // num of possible relative positions
      #pragma omp parallel for
      for(size_t i=0; i<cells.size(); i++) {
        FMM_Node* node = &cells[i];
        node->interac_list[type].resize(numRelCoord, 0);
        BuildList(node, type);
      }
    }
  }

  void SetColleagues(FMM_Node* node=NULL) {
    FMM_Node* parent_node;
    FMM_Node* tmp_node1;
    FMM_Node* tmp_node2;
    for(int i=0; i<27; i++) node->colleague[i] = NULL;
    parent_node = node->parent;
    if(parent_node==NULL) return;
    int l=node->octant;         // l is octant
    for(int i=0; i<27; i++) {
      tmp_node1 = parent_node->colleague[i];  // loop over parent's colleagues
      if(tmp_node1!=NULL && !tmp_node1->IsLeaf()) {
        for(int j=0; j<8; j++) {
          tmp_node2=tmp_node1->Child(j);    // loop over parent's colleages child
          if(tmp_node2!=NULL) {
            bool flag=true;
            int a=1, b=1, new_indx=0;
            for(int k=0; k<3; k++) {
              int indx_diff=(((i/b)%3)-1)*2+((j/a)%2)-((l/a)%2);
              if(-1>indx_diff || indx_diff>1) flag=false;
              new_indx+=(indx_diff+1)*b;
              a*=2;
              b*=3;
            }
            if(flag)
              node->colleague[new_indx] = tmp_node2;
          }
        }
      }
    }
  }

  void SetColleagues(FMM_Nodes& nodes) {
    nodes[0].colleague[13] = &nodes[0];
    for(int i=1; i<nodes.size(); i++) {
      SetColleagues(&nodes[i]);
    }
  }
}
#endif
