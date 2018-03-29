#ifndef _PVFMM_FMM_TREE_HPP_
#define _PVFMM_FMM_TREE_HPP_
#include "intrinsics.h"
#include "pvfmm.h"
#include <queue>
#include "geometry.h"
#include "precomp_mat.hpp"

namespace pvfmm{
  struct PackedData{
    size_t len;
    Matrix<real_t>* ptr;
    Vector<size_t> cnt;
    Vector<size_t> dsp;

    PackedData() {}
    PackedData(Matrix<real_t>* mat, std::vector<FMM_Node*> nodes, int type) {
      ptr = mat;
      len = mat->Dim(0) * mat->Dim(1);
      cnt.Resize(nodes.size());
      dsp.Resize(nodes.size());
      for(int i=0; i<nodes.size(); i++) {
        FMM_Node* node = nodes[i];
        Vector<real_t>* vec;
        switch (type) {
          case 1:  vec = &(node->src_coord); break;  // src_coord
          case 2:  vec = &(node->src_value); break;  // src_value
          case 3:  vec = &(node->trg_coord); break;  // trg_coord
          case 4:  vec = &(node->trg_value); break;  // trg_value
          case 5:  vec = &(upwd_equiv_surf[node->depth]); break;  // upward equivalent surface coords
          case 6:  vec = &(upwd_check_surf[node->depth]); break;  // upward check      surface coords
          case 7:  vec = &(node->upward_equiv); break;  // upward equivalent charges
          case 8:  vec = &(dnwd_equiv_surf[node->depth]); break;  // downward equivalent surface coords
          case 9:  vec = &(dnwd_check_surf[node->depth]); break;  // downward check      surface coords
          case 10: vec = &(node->dnward_equiv); break;  // downward equivalent charges
          default: assert(0 && "PackedData type has to be an integer from 1 to 10");
        }
        dsp[i] = vec->data_ptr - mat[0][0];
        cnt[i] = vec->Dim();
      }
    }
  };

  struct ptSetupData{
    const Kernel* kernel;
    PackedData src_coord;
    PackedData src_value;
    PackedData trg_coord;
    PackedData trg_value;
    InteracData pt_interac_data;
  };

  struct SetupBase {
    const Kernel* kernel;
    std::vector<FMM_Node*> nodes_in ;
    std::vector<FMM_Node*> nodes_out;
    Matrix<real_t>* input_data;
    Matrix<real_t>* output_data;
  };

  // U, X, W lists & P2M & L2P Setup
  struct BodiesSetup : SetupBase {
    Matrix<real_t>* coord_data;
    ptSetupData pt_setup_data;
  };

  // M2M & L2L Setup
  struct CellsSetup : SetupBase {
    int level;
    size_t M_dim0;
    size_t M_dim1;
    Mat_Type interac_type;
    std::vector<Vector<real_t>*> input_vector;
    std::vector<Vector<real_t>*> output_vector;
    std::vector<size_t> interac_cnt;
    std::vector<size_t> interac_mat;
    std::vector<size_t>  input_perm;
    std::vector<size_t> output_perm;
    std::vector<char>* precomp_data;
  };
  
  // M2L Setup
  struct M2LSetup : SetupBase {
    Mat_Type interac_type;
    std::vector<Vector<real_t>*> input_vector;
    std::vector<Vector<real_t>*> output_vector;
    VListData vlist_data;
  };

class FMM_Tree {
public:
  int multipole_order;
  FMM_Node* root_node;
  const Kernel* kernel;
  InteracList* interacList;
  PrecompMat* mat;
  std::vector<FMM_Node*> node_lst;
  BodiesSetup U_data, W_data, X_data;
  BodiesSetup P2M_data, L2P_data;
  std::vector<CellsSetup> M2M_data;
  std::vector<CellsSetup> L2L_data;
  M2LSetup M2L_data;

public:
  FMM_Tree(int multi_order, const Kernel* kernel_, InteracList* interacList_, PrecompMat* mat_): 
    multipole_order(multi_order), kernel(kernel_), interacList(interacList_), mat(mat_),
    root_node(NULL) {
    vprecomp_fft_flag = false;
    vlist_fft_flag = false;
    vlist_ifft_flag = false; 
  }

  ~FMM_Tree(){
    if(root_node!=NULL) delete root_node;
    if(vprecomp_fft_flag) fft_destroy_plan(vprecomp_fftplan);
    if(vlist_fft_flag) fft_destroy_plan(vlist_fftplan);
    if(vlist_ifft_flag) fft_destroy_plan(vlist_ifftplan);
    vlist_fft_flag = false;
    vlist_ifft_flag = false;
  }

/* 1st Part: Tree Construction
 * Interface: Initialize(init_data) */
private:
  inline int p2oLocal(std::vector<MortonId> & nodes, std::vector<MortonId>& leaves,
		      unsigned int maxNumPts, unsigned int maxDepth, bool complete) {
    assert(maxDepth<=MAX_DEPTH);
    std::vector<MortonId> leaves_lst;
    unsigned int init_size=leaves.size();
    unsigned int num_pts=nodes.size();
    MortonId curr_node=leaves[0];
    MortonId last_node=leaves[init_size-1].NextId();
    MortonId next_node;
    unsigned int curr_pt=0;
    unsigned int next_pt=curr_pt+maxNumPts;
    while(next_pt <= num_pts){
      next_node = curr_node.NextId();
      while( next_pt < num_pts && next_node > nodes[next_pt] && curr_node.GetDepth() < maxDepth-1 ){
	curr_node = curr_node.getDFD(curr_node.GetDepth()+1);
	next_node = curr_node.NextId();
      }
      leaves_lst.push_back(curr_node);
      curr_node = next_node;
      unsigned int inc=maxNumPts;
      while(next_pt < num_pts && curr_node > nodes[next_pt]){
	inc=inc<<1;
	next_pt+=inc;
	if(next_pt > num_pts){
	  next_pt = num_pts;
	  break;
	}
      }
      curr_pt = std::lower_bound(&nodes[0]+curr_pt,&nodes[0]+next_pt,curr_node,std::less<MortonId>())-&nodes[0];
      if(curr_pt >= num_pts) break;
      next_pt = curr_pt + maxNumPts;
      if(next_pt > num_pts) next_pt = num_pts;
    }
    if(complete) {
      while(curr_node<last_node){
	while( curr_node.NextId() > last_node && curr_node.GetDepth() < maxDepth-1 )
	  curr_node = curr_node.getDFD(curr_node.GetDepth()+1);
	leaves_lst.push_back(curr_node);
	curr_node = curr_node.NextId();
      }
    }
    leaves=leaves_lst;
    return 0;
  }

  inline int points2Octree(const std::vector<MortonId>& pt_mid, std::vector<MortonId>& nodes,
			   unsigned int maxDepth, unsigned int maxNumPts) {
    int myrank=0, np=1;
    Profile::Tic("SortMortonId", true, 10);
    std::vector<MortonId> pt_sorted;
    HyperQuickSort(pt_mid, pt_sorted);
    size_t pt_cnt=pt_sorted.size();
    Profile::Toc();

    Profile::Tic("p2o_local", false, 10);
    nodes.resize(1);
    nodes[0]=MortonId();
    p2oLocal(pt_sorted, nodes, maxNumPts, maxDepth, myrank==np-1);
    Profile::Toc();
    return 0;
  }

  FMM_Node* FindNode(MortonId& key, bool subdiv, FMM_Node* start=NULL) {
    int num_child=1UL<<3;
    FMM_Node* n=start;
    if(n==NULL) n=root_node;
    while(n->GetMortonId()<key && (!n->IsLeaf()||subdiv)){
      if(n->IsLeaf()) n->Subdivide();
      if(n->IsLeaf()) break;
      for(int j=0;j<num_child;j++){
	if(n->Child(j)->GetMortonId().NextId()>key){
	  n=n->Child(j);
	  break;
	}
      }
    }
    assert(!subdiv || n->GetMortonId()==key);
    return n;
  }

  FMM_Node* PreorderNxt(FMM_Node* curr_node) {
    assert(curr_node!=NULL);
    int n=(1UL<<3);
    if(!curr_node->IsLeaf())
      for(int i=0;i<n;i++)
	if(curr_node->Child(i)!=NULL)
	  return curr_node->Child(i);
    FMM_Node* node=curr_node;
    while(true){
      int i=node->octant+1;
      node=node->Parent();
      if(node==NULL) return NULL;
      for(;i<n;i++)
	if(node->Child(i)!=NULL)
	  return node->Child(i);
    }
  }

  FMM_Node* PostorderNxt(FMM_Node* curr_node) {
    assert(curr_node!=NULL);
    FMM_Node* node=curr_node;
    int j=node->octant+1;
    node=node->Parent();
    if(node==NULL) return NULL;
    int n=(1UL<<3);
    for(;j<n;j++){
      if(node->Child(j)!=NULL){
	node=node->Child(j);
	while(true){
	  if(node->IsLeaf()) return node;
	  for(int i=0;i<n;i++) {
	    if(node->Child(i)!=NULL){
	      node=node->Child(i);
	      break;
	    }
	  }
	}
      }
    }
    return node;
  }

  FMM_Node* PostorderFirst() {
    FMM_Node* node=root_node;
    int n=(1UL<<3);
    while(true){
      if(node->IsLeaf()) return node;
      for(int i=0;i<n;i++) {
	if(node->Child(i)!=NULL){
	  node=node->Child(i);
	  break;
	}
      }
    }
  }

public:
  std::vector<FMM_Node*>& GetNodeList() {
    node_lst.clear();
    FMM_Node* n=root_node;
    while(n!=NULL){
      node_lst.push_back(n);
      n=PreorderNxt(n);
    }
    return node_lst;
  }

  void Initialize(InitData* init_data) {
    Profile::Tic("InitTree",true);{
      Profile::Tic("InitRoot",false,5);
      int max_depth=init_data->max_depth;
      if(max_depth>MAX_DEPTH) max_depth=MAX_DEPTH;
      if(root_node) delete root_node;
      root_node=new FMM_Node();
      root_node->Initialize(NULL,0,init_data);
      Profile::Toc();

      Profile::Tic("Points2Octee",true,5);
      std::vector<MortonId> lin_oct;
      std::vector<MortonId> pt_mid;
      Vector<real_t>& pt_c=root_node->pt_coord;
      size_t pt_cnt=pt_c.Dim()/3;
      pt_mid.resize(pt_cnt);
#pragma omp parallel for
      for(size_t i=0;i<pt_cnt;i++){
        pt_mid[i]=MortonId(pt_c[i*3+0],pt_c[i*3+1],pt_c[i*3+2],max_depth);
      }
      points2Octree(pt_mid,lin_oct,max_depth,init_data->max_pts);
      Profile::Toc();

      Profile::Tic("SortPoints",true,5);
      std::vector<Vector<real_t>*> coord_lst;
      std::vector<Vector<real_t>*> value_lst;
      root_node->NodeDataVec(coord_lst, value_lst);
      assert(coord_lst.size()==value_lst.size());

      std::vector<size_t> index;
      for(size_t i=0;i<coord_lst.size();i++){
        if(!coord_lst[i]) continue;
        Vector<real_t>& pt_c=*coord_lst[i];
        size_t pt_cnt=pt_c.Dim()/3;
        pt_mid.resize(pt_cnt);
#pragma omp parallel for
        for(size_t i=0;i<pt_cnt;i++){
    	  pt_mid[i]=MortonId(pt_c[i*3+0],pt_c[i*3+1],pt_c[i*3+2],max_depth);
        }
        SortIndex(pt_mid, index);
        Forward  (pt_c, index);
        if(value_lst[i]!=NULL){
          Vector<real_t>& pt_v=*value_lst[i];
          Forward(pt_v, index);
        }
      }
      Profile::Toc();

      Profile::Tic("PointerTree",false,5);
      int omp_p=1;
      for(int i=0;i<1;i++){
        size_t size=lin_oct.size();
        size_t idx=0;
        FMM_Node* n=root_node;
        while(n!=NULL && idx<size){
          MortonId mortonId=n->GetMortonId();
          if(mortonId.isAncestor(lin_oct[idx])){
            if(n->IsLeaf()) n->Subdivide();
          }else if(mortonId==lin_oct[idx]){
            if(!n->IsLeaf()) n->Truncate();
            assert(n->IsLeaf());
            idx++;
          }else{
            n->Truncate();
          }
          n=PreorderNxt(n);
        }
      }
      Profile::Toc();
    }
    Profile::Toc();
  }
/* End of 1nd Part: Tree Construction */

/* 2nd Part: Setup FMM */
private:
  void SetColleagues(FMM_Node* node=NULL) {
    int n1=27;
    int n2=8;
    if(node==NULL){         // for root node
      FMM_Node* curr_node=root_node;
      if(curr_node!=NULL){
	curr_node->SetColleague(curr_node,(n1-1)/2);
        curr_node=PreorderNxt(curr_node);
      }
      std::vector<std::vector<FMM_Node*> > nodes(MAX_DEPTH);
      // traverse all nodes, store nodes at each level in a vector
      while(curr_node!=NULL){
        nodes[curr_node->depth].push_back(curr_node);
        curr_node=PreorderNxt(curr_node);
      }

      for(size_t i=0;i<MAX_DEPTH;i++){    // loop over each level
        size_t j0=nodes[i].size();        // j0 is num of nodes at level i
        FMM_Node** nodes_=&nodes[i][0];
#pragma omp parallel for
        for(size_t j=0;j<j0;j++){
          SetColleagues(nodes_[j]);
        }
      }
    } else {
      FMM_Node* parent_node;
      FMM_Node* tmp_node1;
      FMM_Node* tmp_node2;
      for(int i=0;i<n1;i++)node->SetColleague(NULL,i);
      parent_node=node->Parent();
      if(parent_node==NULL) return;
      int l=node->octant;         // l is octant
      for(int i=0;i<n1;i++){
        tmp_node1=parent_node->Colleague(i);  // loop over parent's colleagues
        if(tmp_node1!=NULL)
        if(!tmp_node1->IsLeaf()){
          for(int j=0;j<n2;j++){
            tmp_node2=tmp_node1->Child(j);    // loop over parent's colleages child
            if(tmp_node2!=NULL){
              bool flag=true;
              int a=1,b=1,new_indx=0;
              for(int k=0;k<3;k++){
                int indx_diff=(((i/b)%3)-1)*2+((j/a)%2)-((l/a)%2);
                if(-1>indx_diff || indx_diff>1) flag=false;
                new_indx+=(indx_diff+1)*b;
                a*=2;b*=3;
              }
              if(flag){
                node->SetColleague(tmp_node2,new_indx);
              }
            }
          }
        }
      }
    }
  }

  // generate node_list for different operators and node_data_buff (src trg information)
  void CollectNodeData(std::vector<FMM_Node*>& nodes, std::vector<Matrix<real_t> >& node_data_buff, std::vector<std::vector<FMM_Node*> >& n_list) {
    n_list.resize(7);
    std::vector<std::vector<Vector<real_t>* > > vec_list(7);      // vec of vecs of pointers
    // post-order traversal
    std::vector<FMM_Node*> leafs, nonleafs;                       // input "nodes" is post-order
    for (int i=0; i<nodes.size(); i++) {
      if (nodes[i]->IsLeaf()) {
        leafs.push_back(nodes[i]);
        nodes[i]->pt_cnt[0] += nodes[i]->src_coord.Dim()  / 3;
        nodes[i]->pt_cnt[1] += nodes[i]->trg_coord.Dim()  / 3;
      } else {
        nonleafs.push_back(nodes[i]);
        nodes[i]->src_coord.Resize(0);
        nodes[i]->trg_coord.Resize(0);
        nodes[i]->src_value.Resize(0);
        nodes[i]->trg_value.Resize(0);
        for (int j=0; j<8; j++) {
          FMM_Node* child = nodes[i]->Child(j);
          nodes[i]->pt_cnt[0] += child->pt_cnt[0];
          nodes[i]->pt_cnt[1] += child->pt_cnt[1];
        }
      }
    }
    // level order traversal
    std::vector<FMM_Node*> nodesLevelOrder;            // level order traversal with leafs
    std::vector<FMM_Node*> nonleafsLevelOrder;         // level order traversal without leafs
    std::queue<FMM_Node*> nodesQueue;
    nodesQueue.push(root_node);
    while (!nodesQueue.empty()) {
      FMM_Node* curr = nodesQueue.front();
      nodesQueue.pop();
      if (curr != root_node)  nodesLevelOrder.push_back(curr);
      if (!curr->IsLeaf()) nonleafsLevelOrder.push_back(curr);
      for (int i=0; i<8; i++) {
        FMM_Node* child = curr->Child(i);
        if (child!=NULL) nodesQueue.push(child);
      }
    }
    nodesLevelOrder.push_back(root_node);   // level 0 root is the last one instead of the first elem in pvfmm's level order TT
    // fill in node_lists
    n_list[0] = nodesLevelOrder;
    n_list[1] = nodesLevelOrder;
    n_list[2] = nonleafsLevelOrder;
    n_list[3] = nonleafsLevelOrder;
    n_list[4] = leafs;
    n_list[5] = leafs;
    n_list[6] = leafs;
    // fill in vec_list
    int n_ue = mat->ClassMat(M2M_U_Type, 0).Dim(1);
    int n_de = mat->ClassMat(L2L_V_Type, 0).Dim(0);
    for(int i=0; i<nodesLevelOrder.size(); i++) {
      FMM_Node* node = nodesLevelOrder[i];
      node->upward_equiv.Resize(n_ue);
      node->dnward_equiv.Resize(n_de);
      vec_list[0].push_back( &(node->upward_equiv) );
      vec_list[1].push_back( &(node->dnward_equiv) );
    }
    int trg_dof = kernel->ker_dim[1];
    for(int i=0; i<leafs.size(); i++) {
      FMM_Node* leaf = leafs[i];
      int n_trg_val = (leaf->trg_coord.Dim()/3) * trg_dof;
      leaf->trg_value.Resize(n_trg_val);
      vec_list[4].push_back( &(leaf->src_value) );        // src_value should be initialized before CollectNodeData
      vec_list[5].push_back( &(leaf->trg_value) );
      vec_list[6].push_back( &(leaf->src_coord) );
      vec_list[6].push_back( &(leaf->trg_coord) );
    }
    for(int depth=0; depth<MAX_DEPTH; depth++){
      vec_list[6].push_back( &upwd_check_surf[depth] );   // these surf coords are generated in FMM_Tree constructor
      vec_list[6].push_back( &upwd_equiv_surf[depth] );
      vec_list[6].push_back( &dnwd_check_surf[depth] );
      vec_list[6].push_back( &dnwd_equiv_surf[depth] );
    }

    node_data_buff.resize(vec_list.size()+1);             // FMM_Tree member node_data_buff: vec of 8 Matrices
    for(int idx=0; idx<vec_list.size(); idx++){           // loop over vec_list
      Matrix<real_t>& buff = node_data_buff[idx];         // reference to idx-th buff chunk
      std::vector<Vector<real_t>*>& vecs = vec_list[idx]; // reference to idx-th chunk of Vector's pointers
      int nvecs = vecs.size();                            // num of Vector's pointers in current chunk
      if (!nvecs) continue;
      std::vector<int> size(nvecs);                       // size of each Vector that ptr points to
      std::vector<int> disp(nvecs, 0);                    // displacement of sizes
#pragma omp parallel for
      for (int i=0; i<nvecs; i++) size[i] = vecs[i]->Dim();  // calculate Vector size
      scan(&size[0], &disp[0], nvecs);                    // calculate Vector size's displ
      size_t buff_size = size[nvecs-1] + disp[nvecs-1];   // total buff size needed
//if(idx==0) std::cout << "buff_size 0 " << buff_size << std::endl;
      if (!buff_size) continue;
      if (buff.Dim(0)*buff.Dim(1) < buff_size) {
        buff.ReInit(1,buff_size*1.05);                    // buff is a huge 1-row matrix
      }
#pragma omp parallel for
      for (int i=0; i<nvecs; i++) {
        if (size[i]>0) {
          memcpy(&buff[0][0]+disp[i], vecs[i]->data_ptr, size[i]*sizeof(real_t));
        }
        vecs[i]->ReInit3(size[i], &buff[0][0]+disp[i], false);  // keep data_ptr in nodes but release the ownership of the data to node_data_buff
      }
    }
  }

  // Build t-type interaction list for node n
  void BuildList(FMM_Node* n, Mat_Type t) {
    const int n_child=8, n_collg=27;
    int c_hash, idx, rel_coord[3];
    int p2n = n->octant;       // octant
    FMM_Node* p = n->Parent(); // parent node
    std::vector<FMM_Node*>& interac_list = n->interac_list[t];
    switch (t) {
      case P2M_Type:
        if(n->IsLeaf()) interac_list[0] = n;
        break;
      case L2P_Type:
	if(n->IsLeaf()) interac_list[0] = n;
	break;
      case M2M_Type:
	if(n->IsLeaf()) return;
	for(int j=0;j<n_child;j++) {
	  rel_coord[0] = -1+(j & 1?2:0);
	  rel_coord[1] = -1+(j & 2?2:0);
	  rel_coord[2] = -1+(j & 4?2:0);
	  c_hash = interacList->coord_hash(rel_coord);
	  idx = interacList->hash_lut[t][c_hash];
	  FMM_Node* chld=(FMM_Node*)n->Child(j);
	  if(idx>=0) interac_list[idx]=chld;
	}
	break;
      case L2L_Type:
	if(p == NULL) return;
        rel_coord[0] = -1+(p2n & 1?2:0);
        rel_coord[1] = -1+(p2n & 2?2:0);
        rel_coord[2] = -1+(p2n & 4?2:0);
        c_hash = interacList->coord_hash(rel_coord);
        idx = interacList->hash_lut[t][c_hash];
        if(idx>=0) interac_list[idx]=p;
	break;
      case U0_Type:
	if(p == NULL || !n->IsLeaf()) return;
	for(int i=0;i<n_collg;i++){
	  FMM_Node* pc = p->Colleague(i);
	  if(pc!=NULL && pc->IsLeaf()){
	    rel_coord[0]=( i %3)*4-4-(p2n & 1?2:0)+1;
	    rel_coord[1]=((i/3)%3)*4-4-(p2n & 2?2:0)+1;
	    rel_coord[2]=((i/9)%3)*4-4-(p2n & 4?2:0)+1;
	    c_hash = interacList->coord_hash(rel_coord);
	    idx = interacList->hash_lut[t][c_hash];
	    if(idx>=0) interac_list[idx] = pc;
	  }
	}
	break;
      case U1_Type:
	if(!n->IsLeaf()) return;
	for(int i=0;i<n_collg;i++){
	  FMM_Node* col=(FMM_Node*)n->Colleague(i);
	  if(col!=NULL && col->IsLeaf()){
            rel_coord[0]=( i %3)-1;
            rel_coord[1]=((i/3)%3)-1;
            rel_coord[2]=((i/9)%3)-1;
            c_hash = interacList->coord_hash(rel_coord);
            idx = interacList->hash_lut[t][c_hash];
            if(idx>=0) interac_list[idx] = col;
	  }
	}
	break;
      case U2_Type:
	if(!n->IsLeaf()) return;
	for(int i=0;i<n_collg;i++){
	  FMM_Node* col=(FMM_Node*)n->Colleague(i);
	  if(col!=NULL && !col->IsLeaf()){
	    for(int j=0;j<n_child;j++){
	      rel_coord[0]=( i %3)*4-4+(j & 1?2:0)-1;
	      rel_coord[1]=((i/3)%3)*4-4+(j & 2?2:0)-1;
	      rel_coord[2]=((i/9)%3)*4-4+(j & 4?2:0)-1;
	      c_hash = interacList->coord_hash(rel_coord);
	      idx = interacList->hash_lut[t][c_hash];
	      if(idx>=0){
		assert(col->Child(j)->IsLeaf()); //2:1 balanced
		interac_list[idx] = (FMM_Node*)col->Child(j);
	      }
	    }
	  }
	}
	break;
      case V_Type:
	if(p == NULL) return;
	for(int i=0;i<n_collg;i++){
	  FMM_Node* pc=(FMM_Node*)p->Colleague(i);
	  if(pc!=NULL?!pc->IsLeaf():0){
	    for(int j=0;j<n_child;j++){
	      rel_coord[0]=( i   %3)*2-2+(j & 1?1:0)-(p2n & 1?1:0);
	      rel_coord[1]=((i/3)%3)*2-2+(j & 2?1:0)-(p2n & 2?1:0);
	      rel_coord[2]=((i/9)%3)*2-2+(j & 4?1:0)-(p2n & 4?1:0);
	      c_hash = interacList->coord_hash(rel_coord);
	      idx=interacList->hash_lut[t][c_hash];
	      if(idx>=0) interac_list[idx]=(FMM_Node*)pc->Child(j);
	    }
	  }
	}
	break;
      case V1_Type:
	if(n->IsLeaf()) return;
	for(int i=0;i<n_collg;i++){
	  FMM_Node* col=(FMM_Node*)n->Colleague(i);
	  if(col!=NULL && !col->IsLeaf()){
            rel_coord[0]=( i %3)-1;
            rel_coord[1]=((i/3)%3)-1;
            rel_coord[2]=((i/9)%3)-1;
            c_hash = interacList->coord_hash(rel_coord);
            idx=interacList->hash_lut[t][c_hash];
            if(idx>=0) interac_list[idx]=col;
	  }
	}
	break;
      case W_Type:
	if(!n->IsLeaf()) return;
	for(int i=0;i<n_collg;i++){
	  FMM_Node* col=(FMM_Node*)n->Colleague(i);
	  if(col!=NULL && !col->IsLeaf()){
	    for(int j=0;j<n_child;j++){
	      rel_coord[0]=( i %3)*4-4+(j & 1?2:0)-1;
	      rel_coord[1]=((i/3)%3)*4-4+(j & 2?2:0)-1;
	      rel_coord[2]=((i/9)%3)*4-4+(j & 4?2:0)-1;
	      c_hash = interacList->coord_hash(rel_coord);
	      idx=interacList->hash_lut[t][c_hash];
	      if(idx>=0) interac_list[idx]=(FMM_Node*)col->Child(j);
	    }
	  }
	}
	break;
      case X_Type:
	if(p == NULL) return;
	for(int i=0;i<n_collg;i++){
	  FMM_Node* pc=(FMM_Node*)p->Colleague(i);
	  if(pc!=NULL && pc->IsLeaf()){
	    rel_coord[0]=( i %3)*4-4-(p2n & 1?2:0)+1;
	    rel_coord[1]=((i/3)%3)*4-4-(p2n & 2?2:0)+1;
	    rel_coord[2]=((i/9)%3)*4-4-(p2n & 4?2:0)+1;
	    c_hash = interacList->coord_hash(rel_coord);
	    idx=interacList->hash_lut[t][c_hash];
	    if(idx>=0) interac_list[idx]=pc;
	  }
	}
	break;
      default:
        abort();
    }
  }

  // Fill in interac_list of all nodes, assume sources == target for simplicity
  void BuildInteracLists() {
    std::vector<FMM_Node*>& nodes = GetNodeList();
    std::vector<Mat_Type> interactionTypes = {P2M_Type, M2M_Type, L2L_Type, L2P_Type, U0_Type,
                                              U1_Type, U2_Type, W_Type, X_Type, V1_Type};
    for(Mat_Type& type : interactionTypes) {
      int numRelCoord = interacList->rel_coord[type].size();  // num of possible relative positions
#pragma omp parallel for
      for(size_t i=0; i<nodes.size(); i++) {
        FMM_Node* node = nodes[i];
        node->interac_list[type].resize(numRelCoord, 0);
        BuildList(node, type);
      }
    }
  }

  // Initialize ptSetupData::pt_interac_data.interac_cst, then save ptSetupData in setup_data
  void PtSetup(BodiesSetup& setup_data, ptSetupData* data_){
    ptSetupData& data = *data_;
    if(data.pt_interac_data.interac_cnt.Dim()){
      InteracData& intdata=data.pt_interac_data;
      Vector<size_t>  cnt;
      Vector<size_t>& dsp=intdata.interac_cst;
      cnt.Resize(intdata.interac_cnt.Dim());
      dsp.Resize(intdata.interac_dsp.Dim());
#pragma omp parallel for
      for(size_t trg=0;trg<cnt.Dim();trg++){
        size_t trg_cnt=data.trg_coord.cnt[trg];
        cnt[trg]=0;
        for(size_t i=0;i<intdata.interac_cnt[trg];i++){
          size_t int_id=intdata.interac_dsp[trg]+i;
          size_t src=intdata.in_node[int_id];
          size_t src_cnt=data.src_coord.cnt[src];
          cnt[trg]+=(src_cnt)*trg_cnt;
        }
      }
      dsp[0]=cnt[0];
      scan(&cnt[0],&dsp[0],dsp.Dim());
    }
    setup_data.pt_setup_data = data;
    {
      size_t n=setup_data.output_data->Dim(0)*setup_data.output_data->Dim(1)*sizeof(real_t);
      if(dev_buffer.Dim()<n) dev_buffer.Resize(n);
    }
  }

  void SetupPrecomp(CellsSetup& setup_data){
    assert(setup_data.precomp_data && setup_data.level < MAX_DEPTH);
    Profile::Tic("SetupPrecomp",true,25);
    mat->CompactData(setup_data.level, setup_data.interac_type, *setup_data.precomp_data);
    Profile::Toc();
  }

  void SetupInterac(CellsSetup& setup_data){
    int level=setup_data.level;
    Mat_Type& interac_type = setup_data.interac_type;
    std::vector<FMM_Node*>& nodes_in =setup_data.nodes_in ;
    std::vector<FMM_Node*>& nodes_out=setup_data.nodes_out;
    Matrix<real_t>&  input_data=*setup_data. input_data;
    Matrix<real_t>& output_data=*setup_data.output_data;
    std::vector<Vector<real_t>*>&  input_vector=setup_data. input_vector;
    std::vector<Vector<real_t>*>& output_vector=setup_data.output_vector;
    size_t n_in =nodes_in .size();
    size_t n_out=nodes_out.size();
    if(setup_data.precomp_data->size()==0) SetupPrecomp(setup_data);

    Matrix<real_t>& M0 = mat->ClassMat(interac_type, 0);
    int M_dim0 = M0.Dim(0);
    int M_dim1 = M0.Dim(1);
    size_t mat_cnt=interacList->rel_coord[interac_type].size();

    Profile::Tic("Interac-Data",true,25);
    {
      std::vector<size_t> interac_mat;
      std::vector<size_t> interac_cnt;
      std::vector<size_t>  input_perm;
      std::vector<size_t> output_perm;
      size_t buff_size = 1024l*1024l*1024l;      // 1G buffer
      if(n_out && n_in) {
        std::vector<std::vector<size_t> > precomp_data_offset;
        {
          std::vector<char>& precomp_data=*setup_data.precomp_data;
          char* indx_ptr=&precomp_data[0];
          const int l1_l0 = MAX_DEPTH;
          int size = 1 + (2+2)*l1_l0;    // matches the definition in CompactData
          for(int i=0; i<mat_cnt; i++) {
            std::vector<size_t> temp_data((size_t*)indx_ptr + i*size, (size_t*)indx_ptr + (i+1)*size);
            precomp_data_offset.push_back(temp_data);
          }
        }
        int i = 0;
        for(FMM_Node* node_in : setup_data.nodes_in) node_in->node_id = i++;
        std::vector<std::vector<FMM_Node*> > trg_interac_list(setup_data.nodes_out.size());  // n_out*mat_cnt FMM_Node pointers
        std::vector<std::vector<FMM_Node*> > src_interac_list(setup_data.nodes_in.size());
        for(auto& nodes : src_interac_list) nodes.resize(mat_cnt, NULL);
        
#pragma omp parallel for
        for(int i=0; i<n_out; i++) {           // build index mapping bwt nodes_in & nodes_out
          std::vector<FMM_Node*>& interact_nodes = nodes_out[i]->interac_list[interac_type];
          assert(interact_nodes.size() == mat_cnt);
          trg_interac_list[i] = interact_nodes;
          for(int j=0; j<mat_cnt; j++) {
            if(trg_interac_list[i][j] != NULL) {
              src_interac_list[trg_interac_list[i][j]->node_id][j] = nodes_out[i];
            }
          }
        }
        
        size_t interac_dsp_=0;
        std::vector<std::vector<size_t>> interac_dsp(n_out, std::vector<size_t>(mat_cnt));
        for(size_t j=0;j<mat_cnt;j++){
          for(size_t i=0;i<n_out;i++){
            interac_dsp[i][j]=interac_dsp_;
            if(trg_interac_list[i][j]!=NULL) interac_dsp_++;
          }
          interac_mat.push_back(precomp_data_offset[interacList->interac_class[interac_type][j]][0]);
          interac_cnt.push_back(interac_dsp_-interac_dsp[0][j]);
        }
        assert((M_dim0+M_dim1)*sizeof(real_t) * interac_dsp_ < buff_size);    // assert the buffer size we need is smaller than 1G

        for(size_t i=0; i<n_out; i++) nodes_out[i]->node_id = i;
        for(size_t i=0; i<n_in ; i++) {
          for(size_t j=0; j<mat_cnt; j++) {
            FMM_Node* trg_node=src_interac_list[i][j];
            if(trg_node != NULL) {
              size_t depth=trg_node->depth;
              input_perm .push_back(precomp_data_offset[j][1+4*depth+0]);
              input_perm .push_back(precomp_data_offset[j][1+4*depth+1]);
              input_perm .push_back(interac_dsp[trg_node->node_id][j]*M_dim0*sizeof(real_t));
              input_perm .push_back((size_t)(& input_vector[i][0][0]- input_data[0]));
              assert(input_vector[i]->Dim() == M_dim0);
            }
          }
        }

        for(size_t i=0; i<n_out; i++) {
          for(size_t j=0; j<mat_cnt; j++) {
            if(trg_interac_list[i][j] != NULL) {
              size_t depth=nodes_out[i]->depth;
              output_perm.push_back(precomp_data_offset[j][1+4*depth+2]);
              output_perm.push_back(precomp_data_offset[j][1+4*depth+3]);
              output_perm.push_back(interac_dsp[i][j]*M_dim1*sizeof(real_t));
              output_perm.push_back((size_t)(&output_vector[i][0][0]-output_data[0]));
            }
          }
        }
      }
      if(dev_buffer.Dim()<buff_size) dev_buffer.Resize(buff_size);
      setup_data.M_dim0 = M_dim0;
      setup_data.M_dim1 = M_dim1;
      setup_data.interac_cnt = interac_cnt;
      setup_data.interac_mat = interac_mat;
      setup_data.input_perm = input_perm;
      setup_data.output_perm = output_perm;
    }
    Profile::Toc();
  }

  void ClearFMMData() {
    Profile::Tic("ClearFMMData",true);
    int omp_p=omp_get_max_threads();
#pragma omp parallel for
    for(int j=0;j<omp_p;j++){
      Matrix<real_t>* mat;
      mat=W_data. input_data;
      if(mat && mat->Dim(0)*mat->Dim(1)){
        size_t a=(mat->Dim(0)*mat->Dim(1)*(j+0))/omp_p;
        size_t b=(mat->Dim(0)*mat->Dim(1)*(j+1))/omp_p;
        memset(&(*mat)[0][a],0,(b-a)*sizeof(real_t));
      }
      mat=X_data.output_data;
      if(mat && mat->Dim(0)*mat->Dim(1)){
        size_t a=(mat->Dim(0)*mat->Dim(1)*(j+0))/omp_p;
        size_t b=(mat->Dim(0)*mat->Dim(1)*(j+1))/omp_p;
        memset(&(*mat)[0][a],0,(b-a)*sizeof(real_t));
      }
      mat=U_data.output_data;
      if(mat && mat->Dim(0)*mat->Dim(1)){
        size_t a=(mat->Dim(0)*mat->Dim(1)*(j+0))/omp_p;
        size_t b=(mat->Dim(0)*mat->Dim(1)*(j+1))/omp_p;
        memset(&(*mat)[0][a],0,(b-a)*sizeof(real_t));
      }
    }
    Profile::Toc();
  }

  template<typename ElemType>
  void CopyVec(std::vector<std::vector<ElemType> >& vec_, pvfmm::Vector<ElemType>& vec) {
    int omp_p=omp_get_max_threads();
    std::vector<size_t> vec_dsp(omp_p+1,0);
    for(size_t tid=0;tid<omp_p;tid++){
      vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
    }
    vec.Resize(vec_dsp[omp_p]);
#pragma omp parallel for
    for(size_t tid=0;tid<omp_p;tid++){
      memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
    }
  }

  void U_ListSetup(BodiesSetup& setup_data, std::vector<Matrix<real_t> >& buff, std::vector<std::vector<FMM_Node*> >& n_list){
    // initialize Setup_data
    setup_data.kernel = kernel->k_s2t;
    setup_data. input_data = &buff[4];        // src_value
    setup_data.output_data = &buff[5];        // trg_value
    setup_data. coord_data = &buff[6];        // src & trg coords, upward/dnward check/equiv surface
    setup_data.nodes_in .clear();
    setup_data.nodes_out.clear();
    setup_data.nodes_in = n_list[4];
    setup_data.nodes_out = n_list[5];
    // initialize ptSetupData
    ptSetupData data;
    data.kernel = setup_data.kernel;
    data.src_coord = PackedData(setup_data.coord_data, setup_data.nodes_in, SrcCoord);
    data.src_value = PackedData(setup_data.input_data, setup_data.nodes_in, SrcValue);
    data.trg_coord = PackedData(setup_data.coord_data, setup_data.nodes_out, TrgCoord);
    data.trg_value = PackedData(setup_data.output_data, setup_data.nodes_out, TrgValue);

    // initialize leaf's node_id, can put it in other functions
    int i = 0;
    for(FMM_Node* leaf : setup_data.nodes_in) leaf->node_id = i++;

    std::vector<FMM_Node*>& nodes_in =setup_data.nodes_in ;
    std::vector<FMM_Node*>& nodes_out=setup_data.nodes_out;
    {
      int omp_p=omp_get_max_threads();
      std::vector<std::vector<size_t> > in_node_(omp_p);
      std::vector<std::vector<size_t> > interac_cnt_(omp_p);
      size_t m=multipole_order;
      size_t Nsrf=(6*(m-1)*(m-1)+2);
#pragma omp parallel for
      for(size_t tid=0;tid<omp_p;tid++){
        std::vector<size_t>& in_node    =in_node_[tid]    ;
        std::vector<size_t>& interac_cnt=interac_cnt_[tid]        ;
        size_t a=(nodes_out.size()*(tid+0))/omp_p;
        size_t b=(nodes_out.size()*(tid+1))/omp_p;
        for(size_t i=a;i<b;i++){
          FMM_Node* tnode=nodes_out[i];
          real_t s=powf(0.5,tnode->depth);
          size_t interac_cnt_=0;
          {
            Mat_Type type=U0_Type;
            std::vector<FMM_Node*>& intlst=tnode->interac_list[type];
            for(size_t j=0;j<intlst.size();j++) if(intlst[j]){
              FMM_Node* snode=intlst[j];
              size_t snode_id=snode->node_id;
              if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
              in_node.push_back(snode_id);
              interac_cnt_++;
            }
          }
          {
            Mat_Type type=U1_Type;
            std::vector<FMM_Node*>& intlst=tnode->interac_list[type];
            for(size_t j=0;j<intlst.size();j++) if(intlst[j]){
              FMM_Node* snode=intlst[j];
              size_t snode_id=snode->node_id;
              if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
              in_node.push_back(snode_id);
              interac_cnt_++;
            }
          }
          {
            Mat_Type type=U2_Type;
            std::vector<FMM_Node*>& intlst=tnode->interac_list[type];
            for(size_t j=0;j<intlst.size();j++) if(intlst[j]){
              FMM_Node* snode=intlst[j];
              size_t snode_id=snode->node_id;
              if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
              in_node.push_back(snode_id);
              interac_cnt_++;
            }
          }
          {
            Mat_Type type=X_Type;
            std::vector<FMM_Node*>& intlst=tnode->interac_list[type];
            if(tnode->pt_cnt[1]<=Nsrf)
            for(size_t j=0;j<intlst.size();j++) if(intlst[j]){
              FMM_Node* snode=intlst[j];
              size_t snode_id=snode->node_id;
              if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
              in_node.push_back(snode_id);
              interac_cnt_++;
            }
          }
          {
            Mat_Type type=W_Type;
            std::vector<FMM_Node*>& intlst=tnode->interac_list[type];
            for(size_t j=0;j<intlst.size();j++) if(intlst[j]){
              FMM_Node* snode=intlst[j];
              size_t snode_id=snode->node_id;
              if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
              if(snode->pt_cnt[0]> Nsrf) continue;
              in_node.push_back(snode_id);
              interac_cnt_++;
            }
          }
          interac_cnt.push_back(interac_cnt_);
        }
      }
      {
        InteracData& pt_interac_data=data.pt_interac_data;
	CopyVec(in_node_,pt_interac_data.in_node);
	CopyVec(interac_cnt_,pt_interac_data.interac_cnt);
        {
          pvfmm::Vector<size_t>& cnt=pt_interac_data.interac_cnt;
          pvfmm::Vector<size_t>& dsp=pt_interac_data.interac_dsp;
          dsp.Resize(cnt.Dim()); if(dsp.Dim()) dsp[0]=0;
          scan(&cnt[0],&dsp[0],dsp.Dim());
        }
      }
    }
    PtSetup(setup_data, &data);
  }

  void W_ListSetup(BodiesSetup&  setup_data, std::vector<Matrix<real_t> >& buff, std::vector<std::vector<FMM_Node*> >& n_list){
    {
      setup_data.kernel = kernel->k_m2t;
      setup_data. input_data = &buff[0];              // upward_equiv
      setup_data.output_data = &buff[5];              // trg_value
      setup_data. coord_data = &buff[6];              // src & trg coords, upward/dnward check/equiv surface
      setup_data.nodes_in .clear();
      setup_data.nodes_out.clear();
      std::vector<FMM_Node*>& nodes_in =n_list[0];    // nodesLevelOrder
      std::vector<FMM_Node*>& nodes_out=n_list[5];    // leafs
      for(FMM_Node* node : nodes_in)
        if (node->pt_cnt[0]) setup_data.nodes_in.push_back(node);
      for(FMM_Node* node : nodes_out)
        if (node->trg_coord.Dim() && node->IsLeaf()) setup_data.nodes_out.push_back(node);
    }
    // initialize ptSetupData
    ptSetupData data;
    data.kernel=setup_data.kernel;
    data.src_coord = PackedData(setup_data.coord_data, setup_data.nodes_in, UpwardEquivCoord);
    data.src_value = PackedData(setup_data.input_data, setup_data.nodes_in, UpwardEquivValue);
    data.trg_coord = PackedData(setup_data.coord_data, setup_data.nodes_out, TrgCoord);
    data.trg_value = PackedData(setup_data.output_data, setup_data.nodes_out, TrgValue);

    int i = 0;
    for(FMM_Node* node : setup_data.nodes_in) node->node_id = i++;
    std::vector<FMM_Node*>& nodes_in =setup_data.nodes_in ;
    std::vector<FMM_Node*>& nodes_out=setup_data.nodes_out;
    {
      int omp_p=omp_get_max_threads();
      std::vector<std::vector<size_t> > in_node_(omp_p);
      std::vector<std::vector<size_t> > scal_idx_(omp_p);
      std::vector<std::vector<real_t> > coord_shift_(omp_p);
      std::vector<std::vector<size_t> > interac_cnt_(omp_p);
      size_t m=multipole_order;
      size_t Nsrf=(6*(m-1)*(m-1)+2);
#pragma omp parallel for
      for(size_t tid=0;tid<omp_p;tid++){
        std::vector<size_t>& in_node    =in_node_[tid]    ;
        std::vector<size_t>& scal_idx   =scal_idx_[tid]   ;
        std::vector<real_t>& coord_shift=coord_shift_[tid];
        std::vector<size_t>& interac_cnt=interac_cnt_[tid]        ;
        size_t a=(nodes_out.size()*(tid+0))/omp_p;
        size_t b=(nodes_out.size()*(tid+1))/omp_p;
        for(size_t i=a;i<b;i++){
          FMM_Node* tnode=nodes_out[i];
          real_t s=powf(0.5,tnode->depth);
          size_t interac_cnt_=0;
          {
            Mat_Type type=W_Type;
            std::vector<FMM_Node*>& intlst=tnode->interac_list[type];
            for(size_t j=0;j<intlst.size();j++) if(intlst[j]){
              FMM_Node* snode=intlst[j];
              size_t snode_id=snode->node_id;
              if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
              if(snode->IsLeaf() && snode->pt_cnt[0]<=Nsrf) continue;
              in_node.push_back(snode_id);
              scal_idx.push_back(snode->depth);
              {
                ivec3& rel_coord=interacList->rel_coord[type][j];
                const real_t* scoord=snode->Coord();
                const real_t* tcoord=tnode->Coord();
                real_t shift[3];
                shift[0]=rel_coord[0]*0.25*s-(0+0.25*s)+(tcoord[0]+0.5*s);
                shift[1]=rel_coord[1]*0.25*s-(0+0.25*s)+(tcoord[1]+0.5*s);
                shift[2]=rel_coord[2]*0.25*s-(0+0.25*s)+(tcoord[2]+0.5*s);
                coord_shift.push_back(shift[0]);
                coord_shift.push_back(shift[1]);
                coord_shift.push_back(shift[2]);
              }
              interac_cnt_++;
            }
          }
          interac_cnt.push_back(interac_cnt_);
        }
      }
      {
        InteracData& pt_interac_data=data.pt_interac_data;
	CopyVec(in_node_,pt_interac_data.in_node);
	CopyVec(scal_idx_,pt_interac_data.scal_idx);
	CopyVec(coord_shift_,pt_interac_data.coord_shift);
	CopyVec(interac_cnt_,pt_interac_data.interac_cnt);
        {
          pvfmm::Vector<size_t>& cnt=pt_interac_data.interac_cnt;
          pvfmm::Vector<size_t>& dsp=pt_interac_data.interac_dsp;
          dsp.Resize(cnt.Dim()); if(dsp.Dim()) dsp[0]=0;
          scan(&cnt[0],&dsp[0],dsp.Dim());
        }
      }
    }
    PtSetup(setup_data, &data);
  }

  void X_ListSetup(BodiesSetup&  setup_data, std::vector<Matrix<real_t> >& buff, std::vector<std::vector<FMM_Node*> >& n_list){
    if(!multipole_order) return;
    {
      setup_data.kernel=kernel->k_s2l;
      setup_data. input_data=&buff[4];         // src_value
      setup_data.output_data=&buff[1];         // dnward_equiv
      setup_data. coord_data=&buff[6];         // src & trg coords, upward & downward equiv
      std::vector<FMM_Node*>& nodes_in =n_list[4];
      std::vector<FMM_Node*>& nodes_out=n_list[1];
      setup_data.nodes_in .clear();
      setup_data.nodes_out.clear();
      for(FMM_Node* node : nodes_in)
        if(node->src_coord.Dim() && node->IsLeaf ()) setup_data.nodes_in.push_back(node);
      for(FMM_Node* node : nodes_out)
        if(node->pt_cnt[1]) setup_data.nodes_out.push_back(node);
    }
    ptSetupData data;
    data.kernel=setup_data.kernel;
    data.src_coord = PackedData(setup_data.coord_data, setup_data.nodes_in, SrcCoord);
    data.src_value = PackedData(setup_data.input_data, setup_data.nodes_in, SrcValue);
    data.trg_coord = PackedData(setup_data.coord_data, setup_data.nodes_out, DnwardCheckCoord);
    data.trg_value = PackedData(setup_data.output_data, setup_data.nodes_out, DnwardEquivValue);
    std::vector<FMM_Node*>& nodes_in =setup_data.nodes_in ;
    std::vector<FMM_Node*>& nodes_out=setup_data.nodes_out;

    {
      int omp_p=omp_get_max_threads();
      std::vector<std::vector<size_t> > in_node_(omp_p);
      std::vector<std::vector<size_t> > scal_idx_(omp_p);
      std::vector<std::vector<real_t> > coord_shift_(omp_p);
      std::vector<std::vector<size_t> > interac_cnt_(omp_p);
      size_t m=multipole_order;
      size_t Nsrf=(6*(m-1)*(m-1)+2);
#pragma omp parallel for
      for(size_t tid=0;tid<omp_p;tid++){
        std::vector<size_t>& in_node    =in_node_[tid];
        std::vector<size_t>& scal_idx   =scal_idx_[tid];
        std::vector<real_t>& coord_shift=coord_shift_[tid];
        std::vector<size_t>& interac_cnt=interac_cnt_[tid];
        size_t a=(nodes_out.size()*(tid+0))/omp_p;
        size_t b=(nodes_out.size()*(tid+1))/omp_p;
        for(size_t i=a;i<b;i++){
          FMM_Node* tnode=nodes_out[i];
          if(tnode->IsLeaf() && tnode->pt_cnt[1]<=Nsrf){
            interac_cnt.push_back(0);
            continue;
          }
          real_t s=powf(0.5,tnode->depth);
          size_t interac_cnt_=0;
          {
            Mat_Type type=X_Type;
            std::vector<FMM_Node*>& intlst=tnode->interac_list[type];
            for(size_t j=0;j<intlst.size();j++) if(intlst[j]){
              FMM_Node* snode=intlst[j];
              size_t snode_id=snode->node_id;
              if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
              in_node.push_back(snode_id);
              scal_idx.push_back(snode->depth);
              {
                ivec3& rel_coord=interacList->rel_coord[type][j];
                const real_t* scoord=snode->Coord();
                const real_t* tcoord=tnode->Coord();
                real_t shift[3];
                shift[0]=rel_coord[0]*0.5*s-(scoord[0]+1.0*s)+(0+0.5*s);
                shift[1]=rel_coord[1]*0.5*s-(scoord[1]+1.0*s)+(0+0.5*s);
                shift[2]=rel_coord[2]*0.5*s-(scoord[2]+1.0*s)+(0+0.5*s);
                coord_shift.push_back(shift[0]);
                coord_shift.push_back(shift[1]);
                coord_shift.push_back(shift[2]);
              }
              interac_cnt_++;
            }
          }
          interac_cnt.push_back(interac_cnt_);
        }
      }
      {
        InteracData& pt_interac_data=data.pt_interac_data;
	CopyVec(in_node_,pt_interac_data.in_node);
	CopyVec(scal_idx_,pt_interac_data.scal_idx);
	CopyVec(coord_shift_,pt_interac_data.coord_shift);
	CopyVec(interac_cnt_,pt_interac_data.interac_cnt);
        {
          pvfmm::Vector<size_t>& cnt=pt_interac_data.interac_cnt;
          pvfmm::Vector<size_t>& dsp=pt_interac_data.interac_dsp;
          dsp.Resize(cnt.Dim()); if(dsp.Dim()) dsp[0]=0;
          scan(&cnt[0],&dsp[0],dsp.Dim());
        }
      }
    }
    PtSetup(setup_data, &data);
  }

  void V_ListSetup(M2LSetup&  setup_data, std::vector<Matrix<real_t> >& buff, std::vector<std::vector<FMM_Node*> >& n_list){
    if(!multipole_order) return;
    {
      setup_data.kernel=kernel->k_m2l;
      setup_data.interac_type = V1_Type;
      setup_data. input_data=&buff[0];
      setup_data.output_data=&buff[1];
      std::vector<FMM_Node*>& nodes_in =n_list[2];
      std::vector<FMM_Node*>& nodes_out=n_list[3];
      setup_data.nodes_in .clear();
      setup_data.nodes_out.clear();
      for(FMM_Node* node : nodes_in)
        if(node->pt_cnt[0]) setup_data.nodes_in.push_back(node);
      for(FMM_Node* node : nodes_out) 
        if(node->pt_cnt[1]) setup_data.nodes_out.push_back(node);
    }
    std::vector<FMM_Node*>& nodes_in =setup_data.nodes_in ;
    std::vector<FMM_Node*>& nodes_out=setup_data.nodes_out;
    std::vector<Vector<real_t>*>&  input_vector=setup_data. input_vector;  input_vector.clear();
    std::vector<Vector<real_t>*>& output_vector=setup_data.output_vector; output_vector.clear();
    for(FMM_Node* node : nodes_in)  input_vector.push_back(&(node->Child(0)->upward_equiv));
    for(FMM_Node* node : nodes_out) output_vector.push_back(&(node->Child(0)->dnward_equiv));
    size_t n_in =nodes_in .size();
    size_t n_out=nodes_out.size();
    Profile::Tic("Interac-Data",true,25);
    if(n_out>0 && n_in >0){
      size_t precomp_offset=0;
      Mat_Type& interac_type = setup_data.interac_type;
      size_t mat_cnt = interacList->rel_coord[interac_type].size();
      std::vector<real_t*> precomp_mat;
      for(size_t mat_id=0;mat_id<mat_cnt;mat_id++){
        Matrix<real_t>& M = mat->mat[interac_type][mat_id];
        precomp_mat.push_back(&M[0][0]);
      }
      size_t m=multipole_order;
      size_t ker_dim0=setup_data.kernel->ker_dim[0];
      size_t ker_dim1=setup_data.kernel->ker_dim[1];
      size_t fftsize;
      {
        size_t n1=m*2;
        size_t n2=n1*n1;
        size_t n3_=n2*(n1/2+1);
        size_t chld_cnt=1UL<<3;
        fftsize=2*n3_*chld_cnt;
      }
      int omp_p=omp_get_max_threads();
      size_t buff_size=1024l*1024l*1024l;
      size_t n_blk0=2*fftsize*(ker_dim0*n_in +ker_dim1*n_out)*sizeof(real_t)/buff_size;
      if(n_blk0==0) n_blk0=1;
      std::vector<std::vector<size_t> >  fft_vec(n_blk0);
      std::vector<std::vector<size_t> > ifft_vec(n_blk0);
      std::vector<std::vector<real_t> >  fft_scl(n_blk0);
      std::vector<std::vector<real_t> > ifft_scl(n_blk0);
      std::vector<std::vector<size_t> > interac_vec(n_blk0);
      std::vector<std::vector<size_t> > interac_dsp(n_blk0);
      {
        Matrix<real_t>&  input_data=*setup_data. input_data;
        Matrix<real_t>& output_data=*setup_data.output_data;
        std::vector<std::vector<FMM_Node*> > nodes_blk_in (n_blk0);
        std::vector<std::vector<FMM_Node*> > nodes_blk_out(n_blk0);
        std::vector<real_t> src_scal=kernel->k_m2l->src_scal;
        std::vector<real_t> trg_scal=kernel->k_m2l->trg_scal;

        for(size_t i=0;i<n_in;i++) nodes_in[i]->node_id=i;
        for(size_t blk0=0;blk0<n_blk0;blk0++){
          size_t blk0_start=(n_out* blk0   )/n_blk0;
          size_t blk0_end  =(n_out*(blk0+1))/n_blk0;
          std::vector<FMM_Node*>& nodes_in_ =nodes_blk_in [blk0];
          std::vector<FMM_Node*>& nodes_out_=nodes_blk_out[blk0];
          {
            std::set<FMM_Node*> nodes_in;
            for(size_t i=blk0_start;i<blk0_end;i++){
              nodes_out_.push_back(nodes_out[i]);
              std::vector<FMM_Node*>& lst=nodes_out[i]->interac_list[interac_type];
              for(size_t k=0;k<mat_cnt;k++) if(lst[k]!=NULL && lst[k]->pt_cnt[0]) nodes_in.insert(lst[k]);
            }
            for(typename std::set<FMM_Node*>::iterator node=nodes_in.begin(); node != nodes_in.end(); node++){
              nodes_in_.push_back(*node);
            }
            size_t  input_dim=nodes_in_ .size()*ker_dim0*fftsize;
            size_t output_dim=nodes_out_.size()*ker_dim1*fftsize;
            size_t buffer_dim=2*(ker_dim0+ker_dim1)*fftsize*omp_p;
            if(buff_size<(input_dim + output_dim + buffer_dim)*sizeof(real_t))
              buff_size=(input_dim + output_dim + buffer_dim)*sizeof(real_t);
          }
          {
            for(size_t i=0;i<nodes_in_ .size();i++) fft_vec[blk0].push_back((size_t)(& input_vector[nodes_in_[i]->node_id][0][0]- input_data[0]));
            for(size_t i=0;i<nodes_out_.size();i++)ifft_vec[blk0].push_back((size_t)(&output_vector[blk0_start   +     i ][0][0]-output_data[0]));
            size_t scal_dim0=src_scal.size();
            size_t scal_dim1=trg_scal.size();
            fft_scl [blk0].resize(nodes_in_ .size()*scal_dim0);
            ifft_scl[blk0].resize(nodes_out_.size()*scal_dim1);
            for(size_t i=0;i<nodes_in_ .size();i++){
              size_t depth=nodes_in_[i]->depth+1;
              for(size_t j=0;j<scal_dim0;j++){
                fft_scl[blk0][i*scal_dim0+j]=powf(2.0, src_scal[j]*depth);
              }
            }
            for(size_t i=0;i<nodes_out_.size();i++){
              size_t depth=nodes_out_[i]->depth+1;
              for(size_t j=0;j<scal_dim1;j++){
                ifft_scl[blk0][i*scal_dim1+j]=powf(2.0, trg_scal[j]*depth);
              }
            }
          }
        }
        for(size_t blk0=0;blk0<n_blk0;blk0++){
          std::vector<FMM_Node*>& nodes_in_ =nodes_blk_in [blk0];
          std::vector<FMM_Node*>& nodes_out_=nodes_blk_out[blk0];
          for(size_t i=0;i<nodes_in_.size();i++) nodes_in_[i]->node_id=i;
          {
            size_t n_blk1=nodes_out_.size()*sizeof(real_t)/CACHE_SIZE;
            if(n_blk1==0) n_blk1=1;
            size_t interac_dsp_=0;
            for(size_t blk1=0;blk1<n_blk1;blk1++){
              size_t blk1_start=(nodes_out_.size()* blk1   )/n_blk1;
              size_t blk1_end  =(nodes_out_.size()*(blk1+1))/n_blk1;
              for(size_t k=0;k<mat_cnt;k++){
                for(size_t i=blk1_start;i<blk1_end;i++){
                  std::vector<FMM_Node*>& lst=nodes_out_[i]->interac_list[interac_type];
                  if(lst[k]!=NULL && lst[k]->pt_cnt[0]){
                    interac_vec[blk0].push_back(lst[k]->node_id*fftsize*ker_dim0);
                    interac_vec[blk0].push_back(    i          *fftsize*ker_dim1);
                    interac_dsp_++;
                  }
                }
                interac_dsp[blk0].push_back(interac_dsp_);
              }
            }
          }
        }
      }
      setup_data.vlist_data.buff_size   = buff_size;
      setup_data.vlist_data.m           = m;
      setup_data.vlist_data.n_blk0      = n_blk0;
      setup_data.vlist_data.precomp_mat = precomp_mat;
      setup_data.vlist_data.fft_vec     = fft_vec;
      setup_data.vlist_data.ifft_vec    = ifft_vec;
      setup_data.vlist_data.fft_scl     = fft_scl;
      setup_data.vlist_data.ifft_scl    = ifft_scl;
      setup_data.vlist_data.interac_vec = interac_vec;
      setup_data.vlist_data.interac_dsp = interac_dsp;
    }
    Profile::Toc();
  }

  void P2MSetup(BodiesSetup& setup_data, std::vector<Matrix<real_t> >& buff, std::vector<std::vector<FMM_Node*> >& n_list) {
    if(!multipole_order) return;
    {
      setup_data.kernel=kernel->k_s2m;
      setup_data. input_data=&buff[4];            // src_value
      setup_data.output_data=&buff[0];            // upward_equiv
      setup_data. coord_data=&buff[6];
      std::vector<FMM_Node*>& nodes_in =n_list[4];
      std::vector<FMM_Node*>& nodes_out=n_list[0];
      setup_data.nodes_in .clear();
      setup_data.nodes_out.clear();
      for(FMM_Node* node : nodes_in) 
        if(node->src_coord.Dim() && node->IsLeaf()) setup_data.nodes_in.push_back(node);
      for(FMM_Node* node : nodes_out)
        if(node->src_coord.Dim() && node->IsLeaf()) setup_data.nodes_out.push_back(node);
    }
    ptSetupData data;
    data.kernel = setup_data.kernel;
    data.src_coord = PackedData(setup_data.coord_data, setup_data.nodes_in, SrcCoord);
    data.src_value = PackedData(setup_data.input_data, setup_data.nodes_in, SrcValue);
    data.trg_coord = PackedData(setup_data.coord_data, setup_data.nodes_out, UpwardCheckCoord);
    data.trg_value = PackedData(setup_data.output_data, setup_data.nodes_out, UpwardEquivValue);

    int i = 0;
    for(FMM_Node* node : setup_data.nodes_in) node->node_id = i++;
    std::vector<FMM_Node*>& nodes_in =setup_data.nodes_in ;
    std::vector<FMM_Node*>& nodes_out=setup_data.nodes_out;
    {
      int omp_p=omp_get_max_threads();
      std::vector<std::vector<size_t> > in_node_(omp_p);
      std::vector<std::vector<size_t> > scal_idx_(omp_p);
      std::vector<std::vector<real_t> > coord_shift_(omp_p);
      std::vector<std::vector<size_t> > interac_cnt_(omp_p);
      const Kernel* ker=kernel->k_m2m;
      for(size_t l=0;l<MAX_DEPTH;l++){
        Vector<real_t>& scal=data.pt_interac_data.scal[l*4+2];
        std::vector<real_t>& scal_exp=ker->trg_scal;
        scal.Resize(scal_exp.size());
        for(size_t i=0;i<scal.Dim();i++){
  assert(scal_exp[i] == 1);
          scal[i]=powf(2.0,-scal_exp[i]*l);
        }
      }
      for(size_t l=0;l<MAX_DEPTH;l++){
        Vector<real_t>& scal=data.pt_interac_data.scal[l*4+3];
        std::vector<real_t>& scal_exp=ker->src_scal;
        scal.Resize(scal_exp.size());
        for(size_t i=0;i<scal.Dim();i++){
  assert(scal_exp[i] == 0);
          scal[i]=powf(2.0,-scal_exp[i]*l);
        }
      }
#pragma omp parallel for
      for(size_t tid=0;tid<omp_p;tid++){
        std::vector<size_t>& in_node    =in_node_[tid]    ;
        std::vector<size_t>& scal_idx   =scal_idx_[tid]   ;
        std::vector<real_t>& coord_shift=coord_shift_[tid];
        std::vector<size_t>& interac_cnt=interac_cnt_[tid];
        size_t a=(nodes_out.size()*(tid+0))/omp_p;
        size_t b=(nodes_out.size()*(tid+1))/omp_p;
        for(size_t i=a;i<b;i++){
          FMM_Node* tnode=nodes_out[i];
          real_t s=powf(0.5,tnode->depth);
          size_t interac_cnt_=0;
          {
            Mat_Type type=P2M_Type;
            std::vector<FMM_Node*>& intlst=tnode->interac_list[type];
            for(size_t j=0;j<intlst.size();j++) if(intlst[j]){
              FMM_Node* snode=intlst[j];
              size_t snode_id=snode->node_id;
              if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
              in_node.push_back(snode_id);
              scal_idx.push_back(snode->depth);
              {
                ivec3& rel_coord=interacList->rel_coord[type][j];
  assert(rel_coord[0] ==0 && rel_coord[1] ==0 && rel_coord[2] ==0);
                const real_t* scoord=snode->Coord();
                real_t shift[3];
                shift[0] = -scoord[0];
                shift[1] = -scoord[1];
                shift[2] = -scoord[2];
                coord_shift.push_back(shift[0]);
                coord_shift.push_back(shift[1]);
                coord_shift.push_back(shift[2]);
              }
              interac_cnt_++;
            }
          }
          interac_cnt.push_back(interac_cnt_);
        }
      }
      {
        InteracData& pt_interac_data=data.pt_interac_data;
	CopyVec(in_node_,pt_interac_data.in_node);
	CopyVec(scal_idx_,pt_interac_data.scal_idx);
	CopyVec(coord_shift_,pt_interac_data.coord_shift);
	CopyVec(interac_cnt_,pt_interac_data.interac_cnt);
        {
          pvfmm::Vector<size_t>& cnt=pt_interac_data.interac_cnt;
          pvfmm::Vector<size_t>& dsp=pt_interac_data.interac_dsp;
          dsp.Resize(cnt.Dim()); if(dsp.Dim()) dsp[0]=0;
          scan(&cnt[0],&dsp[0],dsp.Dim());
        }
      }
      {
        InteracData& pt_interac_data=data.pt_interac_data;
        pvfmm::Vector<size_t>& cnt=pt_interac_data.interac_cnt;
        pvfmm::Vector<size_t>& dsp=pt_interac_data.interac_dsp;
        if(cnt.Dim() && cnt[cnt.Dim()-1]+dsp[dsp.Dim()-1]){
          data.pt_interac_data.M[2] = mat->mat[M2M_V_Type][0];
          data.pt_interac_data.M[3] = mat->mat[M2M_U_Type][0];
        }else{
          data.pt_interac_data.M[2].ReInit(0,0);
          data.pt_interac_data.M[3].ReInit(0,0);
        }
      }
    }
    PtSetup(setup_data, &data);
  }

  void M2MSetup(CellsSetup& setup_data, std::vector<Matrix<real_t> >& buff, std::vector<std::vector<FMM_Node*> >& n_list, int level){
    setup_data.level = level;
    setup_data.kernel = kernel->k_m2m;
    setup_data.interac_type = M2M_Type;
    setup_data.input_data = &buff[0];
    setup_data.output_data = &buff[0];

    setup_data.nodes_in.clear();
    setup_data.nodes_out.clear();
    std::vector<FMM_Node*>& nodes_in = n_list[0];
    std::vector<FMM_Node*>& nodes_out = n_list[0];
    for(FMM_Node* node : nodes_in)
      if(node->depth==level+1 && node->pt_cnt[0]) setup_data.nodes_in.push_back(node);
    for(FMM_Node* node : nodes_out)
      if(node->depth==level && node->pt_cnt[0]) setup_data.nodes_out.push_back(node);

    setup_data.input_vector.clear();
    setup_data.output_vector.clear();
    for(FMM_Node* node : setup_data.nodes_in)
      setup_data.input_vector.push_back(&(node->upward_equiv));
    for(FMM_Node* node : setup_data.nodes_out)
      setup_data.output_vector.push_back(&(node->upward_equiv));
    SetupInterac(setup_data);
  }

  void L2LSetup(CellsSetup& setup_data, std::vector<Matrix<real_t> >& buff, std::vector<std::vector<FMM_Node*> >& n_list, int level){
    setup_data.level = level;
    setup_data.kernel = kernel->k_l2l;
    setup_data.interac_type = L2L_Type;
    setup_data.input_data = &buff[1];
    setup_data.output_data = &buff[1];

    setup_data.nodes_in.clear();
    setup_data.nodes_out.clear();
    std::vector<FMM_Node*>& nodes_in =n_list[1];
    std::vector<FMM_Node*>& nodes_out=n_list[1];
    for(FMM_Node* node : nodes_in)
      if(node->depth==level-1 && node->pt_cnt[1]) setup_data.nodes_in.push_back(node);
    for(FMM_Node* node : nodes_out)
      if(node->depth==level && node->pt_cnt[1]) setup_data.nodes_out.push_back(node);

    setup_data.input_vector.clear();
    setup_data.output_vector.clear();
    for(FMM_Node* node : setup_data.nodes_in)
      setup_data.input_vector.push_back(&(node->dnward_equiv));
    for(FMM_Node* node : setup_data.nodes_out)
      setup_data.output_vector.push_back(&(node->dnward_equiv));
    SetupInterac(setup_data);
  }

  void L2PSetup(BodiesSetup&  setup_data, std::vector<Matrix<real_t> >& buff, std::vector<std::vector<FMM_Node*> >& n_list){
    if(!multipole_order) return;
    {
      setup_data.kernel=kernel->k_l2t;
      setup_data. input_data=&buff[1];        // dnward_equiv
      setup_data.output_data=&buff[5];        // trg_value
      setup_data. coord_data=&buff[6];        // coords & equiv surface
      std::vector<FMM_Node*>& nodes_in =n_list[1];
      std::vector<FMM_Node*>& nodes_out=n_list[5];
      setup_data.nodes_in .clear();
      setup_data.nodes_out.clear();
      for(FMM_Node* node : nodes_in)
        if(node->trg_coord.Dim() && node->IsLeaf()) setup_data.nodes_in.push_back(node);
      for(FMM_Node* node : nodes_out)
        if(node->trg_coord.Dim() && node->IsLeaf()) setup_data.nodes_out.push_back(node);
    }
    ptSetupData data;
    data.kernel=setup_data.kernel;
    data.src_coord = PackedData(setup_data.coord_data, setup_data.nodes_in, DnwardEquivCoord);
    data.src_value = PackedData(setup_data.input_data, setup_data.nodes_in, DnwardEquivValue);
    data.trg_coord = PackedData(setup_data.coord_data, setup_data.nodes_out, TrgCoord);
    data.trg_value = PackedData(setup_data.output_data, setup_data.nodes_out, TrgValue);
    int i = 0;
    for(FMM_Node* node : setup_data.nodes_in) node->node_id = i++;

    std::vector<FMM_Node*>& nodes_in =setup_data.nodes_in ;
    std::vector<FMM_Node*>& nodes_out=setup_data.nodes_out;
    {
      int omp_p=omp_get_max_threads();
      std::vector<std::vector<size_t> > in_node_(omp_p);
      std::vector<std::vector<size_t> > scal_idx_(omp_p);
      std::vector<std::vector<real_t> > coord_shift_(omp_p);
      std::vector<std::vector<size_t> > interac_cnt_(omp_p);
      const Kernel* ker=kernel->k_l2l;
      for(size_t l=0;l<MAX_DEPTH;l++){
        Vector<real_t>& scal=data.pt_interac_data.scal[l*4+0];
        std::vector<real_t>& scal_exp=ker->trg_scal;
        scal.Resize(scal_exp.size());
        for(size_t i=0;i<scal.Dim();i++){
          scal[i]=powf(2.0,-scal_exp[i]*l);
        }
      }
      for(size_t l=0;l<MAX_DEPTH;l++){
        Vector<real_t>& scal=data.pt_interac_data.scal[l*4+1];
        std::vector<real_t>& scal_exp=ker->src_scal;
        scal.Resize(scal_exp.size());
        for(size_t i=0;i<scal.Dim();i++){
          scal[i]=powf(2.0,-scal_exp[i]*l);
        }
      }
#pragma omp parallel for
      for(size_t tid=0;tid<omp_p;tid++){
        std::vector<size_t>& in_node    =in_node_[tid]    ;
        std::vector<size_t>& scal_idx   =scal_idx_[tid]   ;
        std::vector<real_t>& coord_shift=coord_shift_[tid];
        std::vector<size_t>& interac_cnt=interac_cnt_[tid];
        size_t a=(nodes_out.size()*(tid+0))/omp_p;
        size_t b=(nodes_out.size()*(tid+1))/omp_p;
        for(size_t i=a;i<b;i++){
          FMM_Node* tnode=nodes_out[i];
          real_t s=powf(0.5,tnode->depth);
          size_t interac_cnt_=0;
          {
            Mat_Type type=L2P_Type;
            std::vector<FMM_Node*>& intlst=tnode->interac_list[type];
            for(size_t j=0;j<intlst.size();j++) if(intlst[j]){
              FMM_Node* snode=intlst[j];
              size_t snode_id=snode->node_id;
              if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
              in_node.push_back(snode_id);
              scal_idx.push_back(snode->depth);
              {
                ivec3& rel_coord=interacList->rel_coord[type][j];
                const real_t* scoord=snode->Coord();
                const real_t* tcoord=tnode->Coord();
                real_t shift[3];
                shift[0]=rel_coord[0]*0.5*s-(0+0.5*s)+(tcoord[0]+0.5*s);
                shift[1]=rel_coord[1]*0.5*s-(0+0.5*s)+(tcoord[1]+0.5*s);
                shift[2]=rel_coord[2]*0.5*s-(0+0.5*s)+(tcoord[2]+0.5*s);
                coord_shift.push_back(shift[0]);
                coord_shift.push_back(shift[1]);
                coord_shift.push_back(shift[2]);
              }
              interac_cnt_++;
            }
          }
          interac_cnt.push_back(interac_cnt_);
        }
      }
      {
        InteracData& pt_interac_data=data.pt_interac_data;
	CopyVec(in_node_,pt_interac_data.in_node);
	CopyVec(scal_idx_,pt_interac_data.scal_idx);
	CopyVec(coord_shift_,pt_interac_data.coord_shift);
	CopyVec(interac_cnt_,pt_interac_data.interac_cnt);
        {
          pvfmm::Vector<size_t>& cnt=pt_interac_data.interac_cnt;
          pvfmm::Vector<size_t>& dsp=pt_interac_data.interac_dsp;
          dsp.Resize(cnt.Dim()); if(dsp.Dim()) dsp[0]=0;
          scan(&cnt[0],&dsp[0],dsp.Dim());
        }
      }
      {
        InteracData& pt_interac_data=data.pt_interac_data;
        pvfmm::Vector<size_t>& cnt=pt_interac_data.interac_cnt;
        pvfmm::Vector<size_t>& dsp=pt_interac_data.interac_dsp;
        if(cnt.Dim() && cnt[cnt.Dim()-1]+dsp[dsp.Dim()-1]){
          data.pt_interac_data.M[0]=mat->mat[L2L_V_Type][0];
          data.pt_interac_data.M[1]=mat->mat[L2L_U_Type][0];
        }else{
          data.pt_interac_data.M[0].ReInit(0,0);
          data.pt_interac_data.M[1].ReInit(0,0);
        }
      }
    }
    PtSetup(setup_data, &data);
  }

public:
  void SetupFMM() {
    Profile::Tic("SetupFMM",true);{
    Profile::Tic("SetColleagues",false,3);
    SetColleagues();
    Profile::Toc();
    Profile::Tic("CollectNodeData",false,3);
    FMM_Node* n = PostorderFirst();
    std::vector<FMM_Node*> all_nodes;
    while(n!=NULL){
      n->pt_cnt[0]=0;
      n->pt_cnt[1]=0;
      all_nodes.push_back(n);        // all_nodes: postorder tree traversal
      n = PostorderNxt(n);
    }
    std::vector<std::vector<FMM_Node*> > node_lists; // TODO: Remove this parameter, not really needed
    CollectNodeData(all_nodes, node_data_buff, node_lists);
    Profile::Toc();

    M2M_data.resize(MAX_DEPTH);
    L2L_data.resize(MAX_DEPTH);

    Profile::Tic("BuildLists",false,3);
    BuildInteracLists();
    Profile::Toc();

    Profile::Tic("UListSetup",false,3);
    U_ListSetup(U_data, node_data_buff, node_lists);
    Profile::Toc();
    Profile::Tic("WListSetup",false,3);
    W_ListSetup(W_data, node_data_buff, node_lists);
    Profile::Toc();
    Profile::Tic("XListSetup",false,3);
    X_ListSetup(X_data, node_data_buff, node_lists);
    Profile::Toc();

    Profile::Tic("VListSetup",false,3);
    V_ListSetup(M2L_data, node_data_buff, node_lists);
    Profile::Toc();

    Profile::Tic("L2LSetup",false,3);
    for(int i=0;i<MAX_DEPTH;i++){
      L2L_data[i].precomp_data = &L2L_precomp_lst;
      L2LSetup(L2L_data[i], node_data_buff, node_lists, i);
    }
    Profile::Toc();

    Profile::Tic("L2PSetup",false,3);
    L2PSetup(L2P_data, node_data_buff, node_lists);
    Profile::Toc();

    Profile::Tic("P2MSetup",false,3);
    P2MSetup(P2M_data, node_data_buff, node_lists);
    Profile::Toc();

    Profile::Tic("M2MSetup",false,3);
    for(size_t i=0;i<MAX_DEPTH;i++){
      M2M_data[i].precomp_data = &M2M_precomp_lst;
      M2MSetup(M2M_data[i], node_data_buff, node_lists, i);
    }
    Profile::Toc();

    ClearFMMData();
    }Profile::Toc();
  }
/* End of 2nd Part: Setup FMM */

/* 3rd Part: Evaluation */
private:
  void evalP2P(BodiesSetup& setup_data) {
    ptSetupData data = setup_data.pt_setup_data;
    InteracData& intdata = data.pt_interac_data;
    int omp_p=omp_get_max_threads();
#pragma omp parallel for
    for(size_t tid=0;tid<omp_p;tid++) {
      Matrix<real_t> src_coord, src_value;
      Matrix<real_t> trg_coord, trg_value;
      size_t trg_a=0, trg_b=0;
      if(intdata.interac_cst.Dim()){
        Vector<size_t>& interac_cst=intdata.interac_cst;
        size_t cost=interac_cst[interac_cst.Dim()-1];
        trg_a = std::lower_bound(&interac_cst[0], &interac_cst[interac_cst.Dim()-1], (cost*(tid+0))/omp_p) - &interac_cst[0]+1;
        trg_b = std::lower_bound(&interac_cst[0], &interac_cst[interac_cst.Dim()-1], (cost*(tid+1))/omp_p) - &interac_cst[0]+1;
        if(tid==omp_p-1) trg_b=interac_cst.Dim();
        if(tid==0) trg_a=0;
      }
      for(int trg=trg_a; trg<trg_b; trg++){
        trg_coord.ReInit(1, data.trg_coord.cnt[trg], &data.trg_coord.ptr[0][0][data.trg_coord.dsp[trg]], false);
        trg_value.ReInit(1, data.trg_value.cnt[trg], &data.trg_value.ptr[0][0][data.trg_value.dsp[trg]], false);
        for(size_t i=0;i<intdata.interac_cnt[trg];i++){
          size_t int_id = intdata.interac_dsp[trg]+i;
          size_t src = intdata.in_node[int_id];
          src_coord.ReInit(1, data.src_coord.cnt[src], &data.src_coord.ptr[0][0][data.src_coord.dsp[src]], false);
          src_value.ReInit(1, data.src_value.cnt[src], &data.src_value.ptr[0][0][data.src_value.dsp[src]], false);
          if(src_coord.Dim(1)){
            setup_data.kernel->ker_poten(src_coord[0], src_coord.Dim(1)/3, src_value[0],
                                         trg_coord[0], trg_coord.Dim(1)/3, trg_value[0]);
          }
        }
      }
    }
  }

  void P2M(BodiesSetup& setup_data) {
    if(setup_data.kernel->ker_dim[0]*setup_data.kernel->ker_dim[1]==0) return;
    char* dev_buff = dev_buffer.data_ptr;
    ptSetupData data = setup_data.pt_setup_data;
    InteracData& intdata = data.pt_interac_data;
    assert(intdata.M[0].Dim(1)==intdata.M[1].Dim(0));
    assert(intdata.M[2].Dim(1)==intdata.M[3].Dim(0));

    int omp_p=omp_get_max_threads();
#pragma omp parallel for
    for(size_t tid=0;tid<omp_p;tid++) {
      Matrix<real_t> src_coord, src_value;
      Matrix<real_t> trg_coord, trg_value;
      Vector<real_t> buff;
      size_t thread_buff_size = setup_data.output_data->Dim(0)*setup_data.output_data->Dim(1) / omp_p;
      buff.ReInit3(thread_buff_size, (real_t*)(dev_buff + tid*thread_buff_size*sizeof(real_t)), false);

      std::vector<Matrix<real_t> > vbuff(6);
      int vdim_ = intdata.M[2].Dim(0) + intdata.M[2].Dim(1) + intdata.M[3].Dim(1);
      int vcnt  = buff.Dim() / vdim_ / 2;
      {
        std::vector<int> vdim(6, 0);
        vdim[0] = intdata.M[0].Dim(0);
        vdim[1] = intdata.M[0].Dim(1);
        vdim[2] = intdata.M[1].Dim(1);
        vdim[3] = intdata.M[2].Dim(0);
        vdim[4] = intdata.M[2].Dim(1);
        vdim[5] = intdata.M[3].Dim(1);
        for(size_t indx=0;indx<6;indx++){
          vbuff[indx].ReInit(vcnt,vdim[indx],&buff[0],false);
          buff.ReInit3(buff.Dim()-vdim[indx]*vcnt, &buff[vdim[indx]*vcnt], false);
        }
      }
      // assign a chunk of target boxes to each thread
      size_t trg_a=0, trg_b=0;
      if(intdata.interac_cst.Dim()){
        Vector<size_t>& interac_cst=intdata.interac_cst;
        size_t cost=interac_cst[interac_cst.Dim()-1];
        trg_a = std::lower_bound(&interac_cst[0], &interac_cst[interac_cst.Dim()-1], (cost*(tid+0))/omp_p) - &interac_cst[0]+1;
        trg_b = std::lower_bound(&interac_cst[0], &interac_cst[interac_cst.Dim()-1], (cost*(tid+1))/omp_p) - &interac_cst[0]+1;
        if(tid==omp_p-1) trg_b=interac_cst.Dim();
        if(tid==0) trg_a=0;
      }
      for(size_t trg0=trg_a; trg0<trg_b; ){
        // calculate num of nodes evaluated per iteration based on buffer size
        int trg1_max;
        if(vcnt){
          if (trg0 + vcnt < trg_b) trg1_max = vcnt;
          else trg1_max = trg_b - trg0;
          assert(trg1_max <= vcnt);
          for(size_t k=0;k<6;k++){
            if(vbuff[k].Dim(0)*vbuff[k].Dim(1)){
              vbuff[k].ReInit(trg1_max, vbuff[k].Dim(1), vbuff[k][0], false);
            }
          }
        } else trg1_max = trg_b - trg0;

        vbuff[3].SetZero();    // vbuff[3]: n_trg * n_ue
        // evaluate potential at upward check surface
        for(size_t trg1=0; trg1<trg1_max; trg1++) {
          size_t trg = trg0 + trg1;
          size_t int_id = intdata.interac_dsp[trg];
          size_t src = intdata.in_node[int_id];
          assert(intdata.interac_cnt[trg] == 1);
          trg_coord.ReInit(1, data.trg_coord.cnt[trg], &data.trg_coord.ptr[0][0][data.trg_coord.dsp[trg]], false);
          trg_value.ReInit(1, data.trg_value.cnt[trg], &data.trg_value.ptr[0][0][data.trg_value.dsp[trg]], false);
          src_coord.ReInit(1, data.src_coord.cnt[src], &data.src_coord.ptr[0][0][data.src_coord.dsp[src]], false);
          src_value.ReInit(1, data.src_value.cnt[src], &data.src_value.ptr[0][0][data.src_value.dsp[src]], false);
          real_t* trg_value = vbuff[3][trg1];
          real_t* shift = &intdata.coord_shift[int_id*3];
          if(shift[0]!=0 || shift[1]!=0 || shift[2]!=0){
            size_t vdim = src_coord.Dim(1);
            Vector<real_t> new_coord(vdim, &buff[0], false);
            assert(buff.Dim()>=vdim);
            for(int i=0; i<src_coord.Dim(1); i++) new_coord[i] = src_coord[0][i] + shift[i%3];
            src_coord.ReInit(1, vdim, &new_coord[0], false);
          }
          setup_data.kernel->ker_poten(src_coord[0], src_coord.Dim(1)/3, src_value[0],
                                       trg_coord[0], trg_coord.Dim(1)/3, trg_value);

          int scal_idx = intdata.scal_idx[int_id];
          Vector<real_t>& scal = intdata.scal[scal_idx*4+2];   // level factor 2*(-l) of UE2UC matrix
          size_t scal_dim = scal.Dim();
          if(scal_dim){
            size_t vdim = vbuff[3].Dim(1);
            for(size_t j=0;j<vdim;j+=scal_dim){
              for(size_t k=0;k<scal_dim;k++){
                vbuff[3][trg1][j+k]*=scal[k];
              }
            }
          }
        }
        // upward_equiv_charge = upward_check_potential * inv(Mat_UE2UC)
        Matrix<real_t>::GEMM(vbuff[4],vbuff[3],intdata.M[2]);
        Matrix<real_t>::GEMM(vbuff[5],vbuff[4],intdata.M[3]);
        // copy equivalent charge's values from buffer back to ptSetupData
        for(size_t trg1=0; trg1<trg1_max; trg1++){
          size_t trg = trg0+trg1;
          trg_value.ReInit(1, data.trg_value.cnt[trg], &data.trg_value.ptr[0][0][data.trg_value.dsp[trg]], false);
          assert(trg_value.Dim(1) == vbuff[5].Dim(1));
          for(size_t i=0; i<trg_value.Dim(1); i++) trg_value[0][i] += vbuff[5][trg1][i];
        }
        trg0+=trg1_max;
      }
    }
  }

  void L2P(BodiesSetup& setup_data) {
    if(setup_data.kernel->ker_dim[0]*setup_data.kernel->ker_dim[1]==0) return;
    char* dev_buff = dev_buffer.data_ptr;
    ptSetupData data = setup_data.pt_setup_data;
    InteracData& intdata = data.pt_interac_data;
    assert(intdata.M[0].Dim(1)==intdata.M[1].Dim(0));
    assert(intdata.M[2].Dim(1)==intdata.M[3].Dim(0));

    int omp_p=omp_get_max_threads();
#pragma omp parallel for
    for(size_t tid=0;tid<omp_p;tid++) {
      Matrix<real_t> src_coord, src_value;
      Matrix<real_t> trg_coord, trg_value;
      Vector<real_t> buff;
      size_t thread_buff_size = setup_data.output_data->Dim(0)*setup_data.output_data->Dim(1) / omp_p;
      buff.ReInit3(thread_buff_size, (real_t*)(dev_buff + tid*thread_buff_size*sizeof(real_t)), false);
      std::vector<Matrix<real_t> > vbuff(6);
      int vdim_ = intdata.M[0].Dim(0) + intdata.M[0].Dim(1) + intdata.M[1].Dim(1);
      int vcnt  = buff.Dim() / vdim_ / 2;
      {
        std::vector<int> vdim(6, 0);
        vdim[0] = intdata.M[0].Dim(0);
        vdim[1] = intdata.M[0].Dim(1);
        vdim[2] = intdata.M[1].Dim(1);
        vdim[3] = intdata.M[2].Dim(0);
        vdim[4] = intdata.M[2].Dim(1);
        vdim[5] = intdata.M[3].Dim(1);
        for(size_t indx=0;indx<6;indx++){
          vbuff[indx].ReInit(vcnt,vdim[indx],&buff[0],false);
          buff.ReInit3(buff.Dim()-vdim[indx]*vcnt, &buff[vdim[indx]*vcnt], false);
        }
      }
      // assign a chunk of target boxes to each thread
      size_t trg_a=0, trg_b=0;
      if(intdata.interac_cst.Dim()){
        Vector<size_t>& interac_cst=intdata.interac_cst;
        size_t cost=interac_cst[interac_cst.Dim()-1];
        trg_a = std::lower_bound(&interac_cst[0], &interac_cst[interac_cst.Dim()-1], (cost*(tid+0))/omp_p) - &interac_cst[0]+1;
        trg_b = std::lower_bound(&interac_cst[0], &interac_cst[interac_cst.Dim()-1], (cost*(tid+1))/omp_p) - &interac_cst[0]+1;
        if(tid==omp_p-1) trg_b=interac_cst.Dim();
        if(tid==0) trg_a=0;
      }
      for(size_t trg0=trg_a; trg0<trg_b; ){
        // calculate num of nodes evaluated per iteration based on buffer size
        int trg1_max;
        if(vcnt){
          if (trg0 + vcnt < trg_b) trg1_max = vcnt;
          else trg1_max = trg_b - trg0;
          assert(trg1_max <= vcnt);
          for(size_t k=0;k<6;k++){
            if(vbuff[k].Dim(0)*vbuff[k].Dim(1)){
              vbuff[k].ReInit(trg1_max, vbuff[k].Dim(1), vbuff[k][0], false);
            }
          }
        } else trg1_max = trg_b - trg0;

        for(size_t trg1=0;trg1<trg1_max;trg1++){
          size_t trg=trg0+trg1;
          size_t int_id=intdata.interac_dsp[trg];
          size_t src=intdata.in_node[int_id];
          src_value.ReInit(1, data.src_value.cnt[src], &data.src_value.ptr[0][0][data.src_value.dsp[src]], false);
          assert(src_value.Dim(1) == vbuff[0].Dim(1));
          for(size_t j=0; j<vbuff[0].Dim(1); j++) vbuff[0][trg1][j]=src_value[0][j];

          int scal_idx=intdata.scal_idx[int_id];
          Vector<real_t>& scal=intdata.scal[scal_idx*4+0];  // level factor 2*(-l) of DE2DC matrix
          size_t scal_dim=scal.Dim();
          if(scal_dim){
            size_t vdim = vbuff[0].Dim(1);
            for(size_t j=0;j<vdim;j+=scal_dim){
              for(size_t k=0;k<scal_dim;k++){
                vbuff[0][trg1][j+k]*=scal[k];
              }
            }
          }
        }
        // downward check potentials to downward equivalent charges
        Matrix<real_t>::GEMM(vbuff[1],vbuff[0],intdata.M[0]);
        Matrix<real_t>::GEMM(vbuff[2],vbuff[1],intdata.M[1]);
        // downward equivalent charges to targets
        for(size_t trg1=0; trg1<trg1_max; trg1++){
          size_t trg = trg0+trg1;
          size_t int_id = intdata.interac_dsp[trg];
          size_t src = intdata.in_node[int_id];
          trg_coord.ReInit(1, data.trg_coord.cnt[trg], &data.trg_coord.ptr[0][0][data.trg_coord.dsp[trg]], false);
          trg_value.ReInit(1, data.trg_value.cnt[trg], &data.trg_value.ptr[0][0][data.trg_value.dsp[trg]], false);
          src_coord.ReInit(1, data.src_coord.cnt[src], &data.src_coord.ptr[0][0][data.src_coord.dsp[src]], false);
          src_value.ReInit(1, data.src_value.cnt[src], &data.src_value.ptr[0][0][data.src_value.dsp[src]], false);
          real_t* src_value = vbuff[2][trg1];
          real_t* shift=&intdata.coord_shift[int_id*3];
          if(shift[0]!=0 || shift[1]!=0 || shift[2]!=0) {
            size_t vdim = src_coord.Dim(1);
            Vector<real_t> new_coord(vdim, &buff[0], false);
            assert(buff.Dim()>=vdim);
            for(int i=0; i<src_coord.Dim(1); i++) new_coord[i] = src_coord[0][i] + shift[i%3];
            src_coord.ReInit(1, vdim, &new_coord[0], false);
          }
          setup_data.kernel->ker_poten(src_coord[0], src_coord.Dim(1)/3, src_value,
                                       trg_coord[0], trg_coord.Dim(1)/3, trg_value[0]);
        }
        trg0+=trg1_max;
      }
    }
  }

  void EvalList(CellsSetup& setup_data){
    if(setup_data.interac_cnt.empty()) return;
    char* buff = dev_buffer.data_ptr;
    char* precomp_data = &(*setup_data.precomp_data)[0];
    real_t* input_data = setup_data.input_data->data_ptr;
    real_t* output_data = setup_data.output_data->data_ptr;
    size_t mat_cnt = interacList->rel_coord[setup_data.interac_type].size();
    Profile::Tic("DeviceComp",false,20);
    {
      size_t M_dim0 = setup_data.M_dim0;
      size_t M_dim1 = setup_data.M_dim1;
      Vector<size_t> interac_cnt = setup_data.interac_cnt;
      Vector<size_t> interac_mat = setup_data.interac_mat;
      Vector<size_t>  input_perm = setup_data.input_perm;
      Vector<size_t> output_perm = setup_data.output_perm;

      int omp_p=omp_get_max_threads();
      size_t vec_cnt=0;
      for(size_t j=0; j<mat_cnt; j++) vec_cnt += interac_cnt[j];
      char* buff_in =buff;
      char* buff_out=buff+vec_cnt*M_dim0*sizeof(real_t);

#pragma omp parallel for
      for(size_t i=0; i<vec_cnt; i++) {
        const size_t*  perm=(size_t*)(precomp_data+input_perm[i*4+0]);
        const real_t*  scal=(real_t*)(precomp_data+input_perm[i*4+1]);
        const real_t* v_in =(real_t*)(  input_data+input_perm[i*4+3]);
        real_t*       v_out=(real_t*)(     buff_in+input_perm[i*4+2]);
        for(size_t j=0; j<M_dim0; j++) {
          v_out[j] = v_in[perm[j]]*scal[j];
        }
      }

      size_t vec_cnt0=0;
      for(size_t j=0; j<mat_cnt; ){
        size_t vec_cnt1=0;
        size_t interac_mat0=interac_mat[j];
        for(; j<mat_cnt && interac_mat[j]==interac_mat0; j++) vec_cnt1+=interac_cnt[j];
        Matrix<real_t> M(M_dim0, M_dim1, (real_t*)(precomp_data+interac_mat0), false);
#pragma omp parallel for
        for(int tid=0;tid<omp_p;tid++){
          size_t a=(vec_cnt1*(tid  ))/omp_p;
          size_t b=(vec_cnt1*(tid+1))/omp_p;
          Matrix<real_t> Ms(b-a, M_dim0, (real_t*)(buff_in +M_dim0*vec_cnt0*sizeof(real_t))+M_dim0*a, false);
          Matrix<real_t> Mt(b-a, M_dim1, (real_t*)(buff_out+M_dim1*vec_cnt0*sizeof(real_t))+M_dim1*a, false);
          Matrix<real_t>::GEMM(Mt,Ms,M);
        }
        vec_cnt0+=vec_cnt1;
      }
#pragma omp parallel for
      for(int tid=0;tid<omp_p;tid++){
        size_t a=( tid   *vec_cnt)/omp_p;
        size_t b=((tid+1)*vec_cnt)/omp_p;
        if(tid>      0 && a<vec_cnt){
          size_t out_ptr=output_perm[a*4+3];
          if(tid>      0) while(a<vec_cnt && out_ptr==output_perm[a*4+3]) a++;
        }
        if(tid<omp_p-1 && b<vec_cnt){
          size_t out_ptr=output_perm[b*4+3];
          if(tid<omp_p-1) while(b<vec_cnt && out_ptr==output_perm[b*4+3]) b++;
        }
        for(size_t i=a;i<b;i++){ // Compute permutations.
          const size_t*  perm=(size_t*)(precomp_data+output_perm[i*4+0]);
          const real_t*  scal=(real_t*)(precomp_data+output_perm[i*4+1]);
          const real_t* v_in =(real_t*)(    buff_out+output_perm[i*4+2]);
          real_t*       v_out=(real_t*)( output_data+output_perm[i*4+3]);
          for(size_t j=0;j<M_dim1;j++ ){
            v_out[j]+=v_in[perm[j]]*scal[j];
          }
        }
      }
    }
    Profile::Toc();
  }

  void VListHadamard(size_t M_dim, std::vector<size_t>& interac_dsp,
                     std::vector<size_t>& interac_vec, std::vector<real_t*>& precomp_mat, Vector<real_t>& fft_in, Vector<real_t>& fft_out){
    size_t chld_cnt=1UL<<3;
    size_t fftsize_in =M_dim*chld_cnt*2;
    size_t fftsize_out=M_dim*chld_cnt*2;
    int err;
    real_t * zero_vec0, * zero_vec1;
    err = posix_memalign((void**)&zero_vec0, MEM_ALIGN, fftsize_in *sizeof(real_t));
    err = posix_memalign((void**)&zero_vec1, MEM_ALIGN, fftsize_out*sizeof(real_t));
    size_t n_out=fft_out.Dim()/fftsize_out;
#pragma omp parallel for
    for(size_t k=0;k<n_out;k++){
      Vector<real_t> dnward_check_fft(fftsize_out, &fft_out[k*fftsize_out], false);
      dnward_check_fft.SetZero();
    }
    size_t mat_cnt=precomp_mat.size();
    size_t blk1_cnt=interac_dsp.size()/mat_cnt;
    int BLOCK_SIZE = CACHE_SIZE * 4 / sizeof(real_t);
    real_t **IN_, **OUT_;
    err = posix_memalign((void**)&IN_ , MEM_ALIGN, BLOCK_SIZE*blk1_cnt*mat_cnt*sizeof(real_t*));
    err = posix_memalign((void**)&OUT_, MEM_ALIGN, BLOCK_SIZE*blk1_cnt*mat_cnt*sizeof(real_t*));
#pragma omp parallel for
    for(size_t interac_blk1=0; interac_blk1<blk1_cnt*mat_cnt; interac_blk1++){
      size_t interac_dsp0 = (interac_blk1==0?0:interac_dsp[interac_blk1-1]);
      size_t interac_dsp1 =                    interac_dsp[interac_blk1  ] ;
      size_t interac_cnt  = interac_dsp1-interac_dsp0;
      for(size_t j=0;j<interac_cnt;j++){
        IN_ [BLOCK_SIZE*interac_blk1 +j]=&fft_in [interac_vec[(interac_dsp0+j)*2+0]];
        OUT_[BLOCK_SIZE*interac_blk1 +j]=&fft_out[interac_vec[(interac_dsp0+j)*2+1]];
      }
      IN_ [BLOCK_SIZE*interac_blk1 +interac_cnt]=zero_vec0;
      OUT_[BLOCK_SIZE*interac_blk1 +interac_cnt]=zero_vec1;
    }
    int omp_p=omp_get_max_threads();
#pragma omp parallel for
    for(int pid=0; pid<omp_p; pid++){
      size_t a=( pid   *M_dim)/omp_p;
      size_t b=((pid+1)*M_dim)/omp_p;
      for(size_t     blk1=0;     blk1<blk1_cnt;    blk1++)
      for(size_t        k=a;        k<       b;       k++)
      for(size_t mat_indx=0; mat_indx< mat_cnt;mat_indx++){
        size_t interac_blk1 = blk1*mat_cnt+mat_indx;
        size_t interac_dsp0 = (interac_blk1==0?0:interac_dsp[interac_blk1-1]);
        size_t interac_dsp1 =                    interac_dsp[interac_blk1  ] ;
        size_t interac_cnt  = interac_dsp1-interac_dsp0;
        real_t** IN = IN_ + BLOCK_SIZE*interac_blk1;
        real_t** OUT= OUT_+ BLOCK_SIZE*interac_blk1;
        real_t* M = precomp_mat[mat_indx] + k*chld_cnt*chld_cnt*2;
        for(size_t j=0;j<interac_cnt;j+=2){
          real_t* M_   = M;
          real_t* IN0  = IN [j+0] + k*chld_cnt*2;
          real_t* IN1  = IN [j+1] + k*chld_cnt*2;
          real_t* OUT0 = OUT[j+0] + k*chld_cnt*2;
          real_t* OUT1 = OUT[j+1] + k*chld_cnt*2;
#ifdef __SSE__
          if (j+2 < interac_cnt) {
            _mm_prefetch(((char *)(IN[j+2] + k*chld_cnt*2)), _MM_HINT_T0);
            _mm_prefetch(((char *)(IN[j+2] + k*chld_cnt*2) + 64), _MM_HINT_T0);
            _mm_prefetch(((char *)(IN[j+3] + k*chld_cnt*2)), _MM_HINT_T0);
            _mm_prefetch(((char *)(IN[j+3] + k*chld_cnt*2) + 64), _MM_HINT_T0);
            _mm_prefetch(((char *)(OUT[j+2] + k*chld_cnt*2)), _MM_HINT_T0);
            _mm_prefetch(((char *)(OUT[j+2] + k*chld_cnt*2) + 64), _MM_HINT_T0);
            _mm_prefetch(((char *)(OUT[j+3] + k*chld_cnt*2)), _MM_HINT_T0);
            _mm_prefetch(((char *)(OUT[j+3] + k*chld_cnt*2) + 64), _MM_HINT_T0);
          }
#endif
          matmult_8x8x2(M_, IN0, IN1, OUT0, OUT1);
        }
      }
    }
    Profile::Add_FLOP(8*8*8*(interac_vec.size()/2)*M_dim);
    free(IN_ );
    free(OUT_);
    free(zero_vec0);
    free(zero_vec1);
  }

  void FFT_UpEquiv(size_t m, std::vector<size_t>& fft_vec, std::vector<real_t>& fft_scal,
		   Vector<real_t>& input_data, Vector<real_t>& output_data, Vector<real_t>& buffer_) {
    size_t n1=m*2;
    size_t n2=n1*n1;
    size_t n3=n1*n2;
    size_t n3_=n2*(n1/2+1);
    size_t chld_cnt=1UL<<3;
    size_t fftsize_in =2*n3_*chld_cnt;
    int omp_p=omp_get_max_threads();
    size_t n=6*(m-1)*(m-1)+2;
    static Vector<size_t> map;
    {
      size_t n_old=map.Dim();
      if(n_old!=n){
        real_t c[3]={0,0,0};
        Vector<real_t> surf=surface(m, c, (real_t)(m-1), 0);
        map.Resize(surf.Dim()/3);
        for(size_t i=0;i<map.Dim();i++)
          map[i]=((size_t)(m-1-surf[i*3]+0.5))+((size_t)(m-1-surf[i*3+1]+0.5))*n1+((size_t)(m-1-surf[i*3+2]+0.5))*n2;
      }
    }
    {
      if(!vlist_fft_flag){
        int err, nnn[3]={(int)n1,(int)n1,(int)n1};
        real_t *fftw_in, *fftw_out;
        err = posix_memalign((void**)&fftw_in,  MEM_ALIGN,   n3 *chld_cnt*sizeof(real_t));
        err = posix_memalign((void**)&fftw_out, MEM_ALIGN, 2*n3_*chld_cnt*sizeof(real_t));
        vlist_fftplan = fft_plan_many_dft_r2c(3,nnn,chld_cnt,
					      (real_t*)fftw_in, NULL, 1, n3,
					      (fft_complex*)(fftw_out),NULL, 1, n3_,
					      FFTW_ESTIMATE);
        free(fftw_in );
        free(fftw_out);
        vlist_fft_flag=true;
      }
    }
    {
      size_t n_in = fft_vec.size();
#pragma omp parallel for
      for(int pid=0; pid<omp_p; pid++){
        size_t node_start=(n_in*(pid  ))/omp_p;
        size_t node_end  =(n_in*(pid+1))/omp_p;
        Vector<real_t> buffer(fftsize_in, &buffer_[fftsize_in*pid], false);
        for(size_t node_idx=node_start; node_idx<node_end; node_idx++){
          Matrix<real_t>  upward_equiv(chld_cnt,n,&input_data[0] + fft_vec[node_idx],false);
          Vector<real_t> upward_equiv_fft(fftsize_in, &output_data[fftsize_in *node_idx], false);
          upward_equiv_fft.SetZero();
          for(size_t k=0;k<n;k++){
            size_t idx=map[k];
            int j1=0;
            for(int j0=0;j0<(int)chld_cnt;j0++)
              upward_equiv_fft[idx+j0*n3]=upward_equiv[j0][k]*fft_scal[node_idx];
          }
          fft_execute_dft_r2c(vlist_fftplan, (real_t*)&upward_equiv_fft[0], (fft_complex*)&buffer[0]);
          for(size_t j=0;j<n3_;j++)
          for(size_t k=0;k<chld_cnt;k++){
            upward_equiv_fft[2*(chld_cnt*j+k)+0]=buffer[2*(n3_*k+j)+0];
            upward_equiv_fft[2*(chld_cnt*j+k)+1]=buffer[2*(n3_*k+j)+1];
          }
        }
      }
    }
  }

  void FFT_Check2Equiv(size_t m, std::vector<size_t>& ifft_vec, std::vector<real_t>& ifft_scal,
		       Vector<real_t>& input_data, Vector<real_t>& output_data, Vector<real_t>& buffer_) {
    size_t n1=m*2;
    size_t n2=n1*n1;
    size_t n3=n1*n2;
    size_t n3_=n2*(n1/2+1);
    size_t chld_cnt=1UL<<3;
    size_t fftsize_out=2*n3_*chld_cnt;
    int omp_p=omp_get_max_threads();
    size_t n=6*(m-1)*(m-1)+2;
    static Vector<size_t> map;
    {
      size_t n_old=map.Dim();
      if(n_old!=n){
        real_t c[3]={0,0,0};
        Vector<real_t> surf=surface(m, c, (real_t)(m-1), 0);
        map.Resize(surf.Dim()/3);
        for(size_t i=0;i<map.Dim();i++)
          map[i]=((size_t)(m*2-0.5-surf[i*3]))+((size_t)(m*2-0.5-surf[i*3+1]))*n1+((size_t)(m*2-0.5-surf[i*3+2]))*n2;
      }
    }
    {
      if(!vlist_ifft_flag){
        int err, nnn[3]={(int)n1,(int)n1,(int)n1};
        real_t *fftw_in, *fftw_out;
        err = posix_memalign((void**)&fftw_in,  MEM_ALIGN, 2*n3_*chld_cnt*sizeof(real_t));
        err = posix_memalign((void**)&fftw_out, MEM_ALIGN,   n3 *chld_cnt*sizeof(real_t));
        vlist_ifftplan = fft_plan_many_dft_c2r(3,nnn,chld_cnt,
					       (fft_complex*)fftw_in, NULL, 1, n3_,
					       (real_t*)(fftw_out),NULL, 1, n3,
					       FFTW_ESTIMATE);
        free(fftw_in);
        free(fftw_out);
        vlist_ifft_flag=true;
      }
    }
    {
      assert(buffer_.Dim()>=2*fftsize_out*omp_p);
      size_t n_out=ifft_vec.size();
#pragma omp parallel for
      for(int pid=0; pid<omp_p; pid++){
        size_t node_start=(n_out*(pid  ))/omp_p;
        size_t node_end  =(n_out*(pid+1))/omp_p;
        Vector<real_t> buffer0(fftsize_out, &buffer_[fftsize_out*(2*pid+0)], false);
        Vector<real_t> buffer1(fftsize_out, &buffer_[fftsize_out*(2*pid+1)], false);
        for(size_t node_idx=node_start; node_idx<node_end; node_idx++){
          Vector<real_t> dnward_check_fft(fftsize_out, &input_data[fftsize_out*node_idx], false);
          Vector<real_t> dnward_equiv(n*chld_cnt,&output_data[0] + ifft_vec[node_idx],false);
          for(size_t j=0;j<n3_;j++)
          for(size_t k=0;k<chld_cnt;k++){
            buffer0[2*(n3_*k+j)+0]=dnward_check_fft[2*(chld_cnt*j+k)+0];
            buffer0[2*(n3_*k+j)+1]=dnward_check_fft[2*(chld_cnt*j+k)+1];
          }
          fft_execute_dft_c2r(vlist_ifftplan, (fft_complex*)&buffer0[0], (real_t*)&buffer1[0]);
          for(size_t k=0;k<n;k++){
            size_t idx=map[k];
            for(int j0=0;j0<(int)chld_cnt;j0++)
              dnward_equiv[n*j0+k]+=buffer1[idx+j0*n3]*ifft_scal[node_idx];
          }
        }
      }
    }
  }

  void M2M(CellsSetup& setup_data){
    if(!multipole_order) return;
    EvalList(setup_data);
  }

  void X_List(BodiesSetup&  setup_data){
    if(!multipole_order) return;
    evalP2P(setup_data);
  }

  void W_List(BodiesSetup&  setup_data){
    if(!multipole_order) return;
    evalP2P(setup_data);
  }

  void U_List(BodiesSetup&  setup_data){
    evalP2P(setup_data);
  }

  void V_List(M2LSetup&  setup_data){
    if(!multipole_order) return;
    int np=1;
    Profile::Tic("Host2Device",false,25);
    int dim0=setup_data.input_data->dim[0];
    int dim1=setup_data.input_data->dim[1];
    size_t buff_size=*((size_t*)&setup_data.vlist_data.buff_size);
    if(dev_buffer.Dim()<buff_size) dev_buffer.Resize(buff_size);
    char * buff=dev_buffer.data_ptr;
    VListData vlist_data=setup_data.vlist_data;
    real_t * input_data=setup_data.input_data->data_ptr;
    real_t * output_data=setup_data.output_data->data_ptr;
    Profile::Toc();
    buff_size     = vlist_data.buff_size;
    size_t m      = vlist_data.m;
    size_t n_blk0 = vlist_data.n_blk0;
    size_t n1 = m * 2;
    size_t n2 = n1 * n1;
    size_t n3_ = n2 * (n1 / 2 + 1);
    size_t chld_cnt = 8;
    size_t fftsize = 2 * n3_ * chld_cnt;
    size_t M_dim = n3_;
    std::vector<real_t*> precomp_mat = vlist_data.precomp_mat;
    std::vector<std::vector<size_t> >  fft_vec = vlist_data.fft_vec;
    std::vector<std::vector<size_t> > ifft_vec = vlist_data.ifft_vec;
    std::vector<std::vector<real_t> >  fft_scl = vlist_data.fft_scl;
    std::vector<std::vector<real_t> > ifft_scl = vlist_data.ifft_scl;
    std::vector<std::vector<size_t> > interac_vec = vlist_data.interac_vec;
    std::vector<std::vector<size_t> > interac_dsp = vlist_data.interac_dsp;
    int omp_p=omp_get_max_threads();
    for(size_t blk0=0;blk0<n_blk0;blk0++){
      size_t n_in = fft_vec[blk0].size();
      size_t n_out=ifft_vec[blk0].size();
      size_t  input_dim=n_in *fftsize;
      size_t output_dim=n_out*fftsize;
      size_t buffer_dim=4*fftsize*omp_p;
      Vector<real_t> fft_in ( input_dim, (real_t*)buff,false);
      Vector<real_t> fft_out(output_dim, (real_t*)(buff+input_dim*sizeof(real_t)),false);
      Vector<real_t>  buffer(buffer_dim, (real_t*)(buff+(input_dim+output_dim)*sizeof(real_t)),false);
      Vector<real_t>  input_data_(dim0*dim1,input_data,false);
      FFT_UpEquiv(m, fft_vec[blk0],  fft_scl[blk0],  input_data_, fft_in, buffer);
      VListHadamard(M_dim, interac_dsp[blk0], interac_vec[blk0], precomp_mat, fft_in, fft_out);
      Vector<real_t> output_data_(dim0*dim1, output_data, false);
      FFT_Check2Equiv(m, ifft_vec[blk0], ifft_scl[blk0], fft_out, output_data_, buffer);
    }
  }


  void L2L(CellsSetup& setup_data){
    if(!multipole_order) return;
    EvalList(setup_data);
  }

  void UpwardPass() {
    int depth=0;
    std::vector<FMM_Node*>& nodes=GetNodeList();
    for(size_t i=0;i<nodes.size();i++){
      FMM_Node* n=nodes[i];
      if(n->depth>depth) depth=n->depth;
    }
    Profile::Tic("P2M",false,5);
    P2M(P2M_data);
    Profile::Toc();
    Profile::Tic("M2M",false,5);
    for(int i=depth-1; i>=0; i--){
      M2M(M2M_data[i]);
    }
    Profile::Toc();
  }

  void DownwardPass() {
    Profile::Tic("Setup",true,3);
    std::vector<FMM_Node*> leaf_nodes;
    int depth=0;
    std::vector<FMM_Node*>& nodes=GetNodeList();
    for(size_t i=0;i<nodes.size();i++){
      FMM_Node* n=nodes[i];
      if(n->IsLeaf()) leaf_nodes.push_back(n);
      if(n->depth>depth) depth=n->depth;
    }
    Profile::Toc();
    Profile::Tic("X-List",false,5);
    X_List(X_data);
    Profile::Toc();
    Profile::Tic("W-List",false,5);
    W_List(W_data);
    Profile::Toc();
    Profile::Tic("U-List",false,5);
    U_List(U_data);
    Profile::Toc();
    Profile::Tic("V-List",false,5);
    V_List(M2L_data);
    Profile::Toc();
    Profile::Tic("L2L",false,5);
    for(size_t i=0; i<=depth; i++) {
      L2L(L2L_data[i]);
    }
    Profile::Toc();
    Profile::Tic("L2P",false,5);
    L2P(L2P_data);
    Profile::Toc();
  }

public:
  void RunFMM() {
    Profile::Tic("RunFMM",true);
    {
      Profile::Tic("UpwardPass",false,2);
      UpwardPass();
      Profile::Toc();
      Profile::Tic("DownwardPass",true,2);
      DownwardPass();
      Profile::Toc();
    }
    Profile::Toc();
  }
/* End of 3rd part: Evaluation */

  void CheckFMMOutput(std::string t_name){
    int np=omp_get_max_threads();

    std::vector<real_t> src_coord;
    std::vector<real_t> src_value;
    FMM_Node* n=root_node;
    while(n!=NULL){
      if(n->IsLeaf()){
        Vector<real_t>& coord_vec=n->src_coord;
        Vector<real_t>& value_vec=n->src_value;
        for(size_t i=0;i<coord_vec.Dim();i++) src_coord.push_back(coord_vec[i]);
        for(size_t i=0;i<value_vec.Dim();i++) src_value.push_back(value_vec[i]);
      }
      n=static_cast<FMM_Node*>(PreorderNxt(n));
    }
    size_t src_cnt = src_coord.size()/3;

    int trg_dof = kernel->ker_dim[1];
    std::vector<real_t> trg_coord;
    std::vector<real_t> trg_poten_fmm;
    size_t step_size = 1 + src_cnt*src_cnt*1e-9;
    n = root_node;
    long long trg_iter=0;
    while(n!=NULL){
      if(n->IsLeaf()){
        Vector<real_t>& coord_vec=n->trg_coord;
        Vector<real_t>& poten_vec=n->trg_value;
        for(size_t i=0; i<coord_vec.Dim()/3; i++){
          if(trg_iter%step_size == 0){
            for(int j=0;j<3        ;j++) trg_coord    .push_back(coord_vec[i*3        +j]);
            for(int j=0;j<trg_dof  ;j++) trg_poten_fmm.push_back(poten_vec[i*trg_dof  +j]);
          }
          trg_iter++;
        }
      }
      n=static_cast<FMM_Node*>(PreorderNxt(n));
    }
    size_t trg_cnt = trg_coord.size()/3;

    std::vector<real_t> trg_poten_dir(trg_cnt*trg_dof ,0);
    pvfmm::Profile::Tic("N-Body Direct",false,1);
#pragma omp parallel for
    for(int i=0;i<np;i++){
      size_t a=(i*trg_cnt)/np;
      size_t b=((i+1)*trg_cnt)/np;
      kernel->ker_poten(&src_coord[0], src_cnt, &src_value[0], &trg_coord[a*3], b-a, &trg_poten_dir[a*trg_dof  ]);
    }
    pvfmm::Profile::Toc();
    {
      real_t max_=0;
      real_t max_err=0;
      for(size_t i=0;i<trg_poten_fmm.size();i++){
        real_t err=fabs(trg_poten_dir[i]-trg_poten_fmm[i]);
        real_t max=fabs(trg_poten_dir[i]);
        if(err>max_err) max_err=err;
        if(max>max_) max_=max;
      }
        std::cout<<"Error      : "<<std::scientific<<max_err/max_<<'\n';
    }
    real_t trg_diff = 0, trg_norm = 0.;
    assert(trg_poten_dir.size() == trg_poten_fmm.size());
    for(size_t i=0; i<trg_poten_fmm.size(); i++) {
      trg_diff += (trg_poten_dir[i]-trg_poten_fmm[i])*(trg_poten_dir[i]-trg_poten_fmm[i]);
      trg_norm += trg_poten_dir[i] * trg_poten_dir[i];
    }
    std::cout << "L2 Error   : " << std::scientific << sqrt(trg_diff/trg_norm) << std::endl;
  }
};

}//end namespace
#endif //_PVFMM_FMM_TREE_HPP_
