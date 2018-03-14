#ifndef _PVFMM_INTERAC_LIST_HPP_
#define _PVFMM_INTERAC_LIST_HPP_
#include "sort.hpp"

namespace pvfmm{
class InteracList{
public:
  std::vector<std::vector<ivec3> > rel_coord;
  std::vector<std::vector<int> > hash_lut;     // coord_hash -> index in rel_coord
  std::vector<std::vector<size_t> > interac_class;  // index -> index of abs_coord of the same class
  std::vector<std::vector<std::vector<Perm_Type> > > perm_list; // index -> list of permutations needed in order to change from abs_coord to rel_coord

  InteracList(){}

  void Initialize(){
    interac_class.resize(Type_Count);
    perm_list.resize(Type_Count);
    rel_coord.resize(Type_Count);
    hash_lut.resize(Type_Count);

    InitList(0,0,1,M2M_V_Type);
    InitList(0,0,1,M2M_U_Type);
    InitList(0,0,1,L2L_V_Type);
    InitList(0,0,1,L2L_U_Type);

    InitList(0,0,1,P2M_Type);
    InitList(1,1,2,M2M_Type);    // count = 8, (+1 or -1)
    InitList(1,1,2,L2L_Type);
    InitList(0,0,1,L2P_Type);

    InitList(3,3,2,U0_Type);     // count = 4^3-2^3 = 56
    InitList(1,0,1,U1_Type);
    InitList(3,3,2,U2_Type);

    InitList(3,2,1,V_Type);
    InitList(1,1,1,V1_Type);
    InitList(5,5,2,W_Type);
    InitList(5,5,2,X_Type);
  }

  void InitList(int max_r, int min_r, int step, Mat_Type t){
    const int max_hash = 2000;
    int n1 = (max_r*2)/step+1;
    int n2 = (min_r*2)/step-1;
    size_t count=n1*n1*n1-(min_r>0?n2*n2*n2:0);
    std::vector<ivec3>& M=rel_coord[t];
    M.resize(count);
    hash_lut[t].assign(max_hash, -1);
    std::vector<int> class_size_hash(max_hash, 0);
    std::vector<int> class_disp_hash(max_hash, 0);
    for(int k=-max_r;k<=max_r;k+=step)
      for(int j=-max_r;j<=max_r;j+=step)
	for(int i=-max_r;i<=max_r;i+=step)
	  if(abs(i)>=min_r || abs(j)>=min_r || abs(k) >= min_r){
	    int c[3]={i,j,k};
            // count the number of coords of the same class
            // ex. (-1,-1,2) is in the same class as (-2,-1,1)
	    class_size_hash[class_hash(c)]++;
	  }
    // class count -> class count displacement
    scan(&class_size_hash[0], &class_disp_hash[0], max_hash);
    size_t count_=0;
    for(int k=-max_r;k<=max_r;k+=step)
      for(int j=-max_r;j<=max_r;j+=step)
	for(int i=-max_r;i<=max_r;i+=step)
	  if(abs(i)>=min_r || abs(j)>=min_r || abs(k) >= min_r){
	    int c[3]={i,j,k};
	    int& idx=class_disp_hash[class_hash(c)]; // idx is the displ of current class
	    for(size_t l=0;l<3;l++) M[idx][l]=c[l];  // store the sorted coords
	    hash_lut[t][coord_hash(c)]=idx;          // store mapping: hash -> index in rel_coord
	    count_++;
	    idx++;
	  }
    assert(count_==count);
    interac_class[t].resize(count);
    perm_list[t].resize(count);
    for(size_t j=0;j<count;j++){         // j is now the index of sorted rel_coord
      if(M[j][0]<0) perm_list[t][j].push_back(ReflecX);
      if(M[j][1]<0) perm_list[t][j].push_back(ReflecY);
      if(M[j][2]<0) perm_list[t][j].push_back(ReflecZ);
      int coord[3];
      coord[0]=abs(M[j][0]);
      coord[1]=abs(M[j][1]);
      coord[2]=abs(M[j][2]);
      if(coord[1]>coord[0] && coord[1]>coord[2]){
	perm_list[t][j].push_back(SwapXY);
	int tmp=coord[0]; coord[0]=coord[1]; coord[1]=tmp;
      }
      if(coord[0]>coord[2]){
	perm_list[t][j].push_back(SwapXZ);
	int tmp=coord[0]; coord[0]=coord[2]; coord[2]=tmp;
      }
      if(coord[0]>coord[1]){
	perm_list[t][j].push_back(SwapXY);
	int tmp=coord[0]; coord[0]=coord[1]; coord[1]=tmp;
      }
      assert(coord[0]<=coord[1] && coord[1]<=coord[2]);
      int c_hash = coord_hash(&coord[0]);
      interac_class[t][j]=hash_lut[t][c_hash];  // j-th rel_coord -> abs coord of the same class
    }
  }

  //! return x + 10y + 100z + 555
  int coord_hash(int* c){
    const int n=5;
    return ( (c[2]+n) * (2*n) + (c[1]+n) ) *(2*n) + (c[0]+n);
  }

  //! swap x,y,z so that |z|>|y|>|x|, return hash of new coord
  int class_hash(int* c_){
    int c[3]={abs(c_[0]), abs(c_[1]), abs(c_[2])};
    if(c[1]>c[0] && c[1]>c[2])
      {int tmp=c[0]; c[0]=c[1]; c[1]=tmp;}
    if(c[0]>c[2])
      {int tmp=c[0]; c[0]=c[2]; c[2]=tmp;}
    if(c[0]>c[1])
      {int tmp=c[0]; c[0]=c[1]; c[1]=tmp;}
    assert(c[0]<=c[1] && c[1]<=c[2]);
    return coord_hash(&c[0]);
  }
};

}//end namespace

#endif //_PVFMM_INTERAC_LIST_HPP_
