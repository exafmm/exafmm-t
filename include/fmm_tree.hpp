#ifndef _PVFMM_FMM_TREE_HPP_
#define _PVFMM_FMM_TREE_HPP_
#include "intrinsics.h"
#include "pvfmm.h"
#include <queue>
#include "geometry.h"
#include "precomp_mat.hpp"

namespace pvfmm {
class FMM_Tree {
 public:
  FMM_Node* root_node;
  const Kernel* kernel;
  InteracList* interacList;
  PrecompMat* mat;
  std::vector<FMM_Node*> node_lst;
  M2LData M2Ldata;

  FMM_Tree(const Kernel* kernel_, InteracList* interacList_, PrecompMat* mat_):
    kernel(kernel_), interacList(interacList_), mat(mat_),
    root_node(NULL) {
    m2l_precomp_fft_flag = false;
    m2l_list_fft_flag = false;
    m2l_list_ifft_flag = false;
  }

  ~FMM_Tree() {
    if(root_node!=NULL) delete root_node;
    if(m2l_precomp_fft_flag) fft_destroy_plan(m2l_precomp_fftplan);
    if(m2l_list_fft_flag) fft_destroy_plan(m2l_list_fftplan);
    if(m2l_list_ifft_flag) fft_destroy_plan(m2l_list_ifftplan);
    m2l_list_fft_flag = false;
    m2l_list_ifft_flag = false;
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
    while(next_pt <= num_pts) {
      next_node = curr_node.NextId();
      while( next_pt < num_pts && next_node > nodes[next_pt] && curr_node.GetDepth() < maxDepth-1 ) {
        curr_node = curr_node.getDFD(curr_node.GetDepth()+1);
        next_node = curr_node.NextId();
      }
      leaves_lst.push_back(curr_node);
      curr_node = next_node;
      unsigned int inc=maxNumPts;
      while(next_pt < num_pts && curr_node > nodes[next_pt]) {
        inc=inc<<1;
        next_pt+=inc;
        if(next_pt > num_pts) {
          next_pt = num_pts;
          break;
        }
      }
      curr_pt = std::lower_bound(&nodes[0]+curr_pt, &nodes[0]+next_pt, curr_node,
                                 std::less<MortonId>())-&nodes[0];
      if(curr_pt >= num_pts) break;
      next_pt = curr_pt + maxNumPts;
      if(next_pt > num_pts) next_pt = num_pts;
    }
    if(complete) {
      while(curr_node<last_node) {
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

  FMM_Node* PreorderNxt(FMM_Node* curr_node) {
    assert(curr_node!=NULL);
    int n=(1UL<<3);
    if(!curr_node->IsLeaf())
      for(int i=0; i<n; i++)
        if(curr_node->Child(i)!=NULL)
          return curr_node->Child(i);
    FMM_Node* node=curr_node;
    while(true) {
      int i=node->octant+1;
      node=node->Parent();
      if(node==NULL) return NULL;
      for(; i<n; i++)
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
    for(; j<n; j++) {
      if(node->Child(j)!=NULL) {
        node=node->Child(j);
        while(true) {
          if(node->IsLeaf()) return node;
          for(int i=0; i<n; i++) {
            if(node->Child(i)!=NULL) {
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
    while(true) {
      if(node->IsLeaf()) return node;
      for(int i=0; i<n; i++) {
        if(node->Child(i)!=NULL) {
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
    while(n!=NULL) {
      node_lst.push_back(n);
      n=PreorderNxt(n);
    }
    return node_lst;
  }

  void Initialize(InitData* init_data) {
    Profile::Tic("InitTree", true);
    {
      Profile::Tic("InitRoot", false, 5);
      int max_depth=init_data->max_depth;
      if(max_depth>MAX_DEPTH) max_depth=MAX_DEPTH;
      if(root_node) delete root_node;
      root_node=new FMM_Node();
      root_node->Initialize(NULL, 0, init_data);
      Profile::Toc();
      Profile::Tic("Points2Octee", true, 5);
      std::vector<MortonId> lin_oct;
      std::vector<MortonId> pt_mid;
      Vector<real_t>& pt_c=root_node->pt_coord;
      size_t pt_cnt=pt_c.Dim()/3;
      pt_mid.resize(pt_cnt);
      #pragma omp parallel for
      for(size_t i=0; i<pt_cnt; i++)
        pt_mid[i]=MortonId(pt_c[i*3+0], pt_c[i*3+1], pt_c[i*3+2], max_depth);
      points2Octree(pt_mid, lin_oct, max_depth, init_data->max_pts);
      Profile::Toc();
      Profile::Tic("SortPoints", true, 5);
      std::vector<Vector<real_t>*> coord_lst;
      std::vector<Vector<real_t>*> value_lst;
      root_node->NodeDataVec(coord_lst, value_lst);
      assert(coord_lst.size()==value_lst.size());
      std::vector<size_t> index;
      for(size_t i=0; i<coord_lst.size(); i++) {
        if(!coord_lst[i]) continue;
        Vector<real_t>& pt_c=*coord_lst[i];
        size_t pt_cnt=pt_c.Dim()/3;
        pt_mid.resize(pt_cnt);
        #pragma omp parallel for
        for(size_t i=0; i<pt_cnt; i++)
          pt_mid[i]=MortonId(pt_c[i*3+0], pt_c[i*3+1], pt_c[i*3+2], max_depth);
        SortIndex(pt_mid, index);
        Forward  (pt_c, index);
        if(value_lst[i]!=NULL) {
          Vector<real_t>& pt_v=*value_lst[i];
          Forward(pt_v, index);
        }
      }
      Profile::Toc();
      Profile::Tic("PointerTree", false, 5);
      int omp_p=1;
      for(int i=0; i<1; i++) {
        size_t size=lin_oct.size();
        size_t idx=0;
        FMM_Node* n=root_node;
        while(n!=NULL && idx<size) {
          MortonId mortonId=n->GetMortonId();
          if(mortonId.isAncestor(lin_oct[idx])) {
            if(n->IsLeaf()) n->Subdivide();
          } else if(mortonId==lin_oct[idx]) {
            if(!n->IsLeaf()) n->Truncate();
            assert(n->IsLeaf());
            idx++;
          } else
            n->Truncate();
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
    if(node==NULL) {        // for root node
      FMM_Node* curr_node=root_node;
      if(curr_node!=NULL) {
        curr_node->SetColleague(curr_node, (n1-1)/2);
        curr_node=PreorderNxt(curr_node);
      }
      std::vector<std::vector<FMM_Node*> > nodes(MAX_DEPTH);
      // traverse all nodes, store nodes at each level in a vector
      while(curr_node!=NULL) {
        nodes[curr_node->depth].push_back(curr_node);
        curr_node=PreorderNxt(curr_node);
      }
      for(size_t i=0; i<MAX_DEPTH; i++) { // loop over each level
        size_t j0=nodes[i].size();        // j0 is num of nodes at level i
        FMM_Node** nodes_=&nodes[i][0];
        #pragma omp parallel for
        for(size_t j=0; j<j0; j++)
          SetColleagues(nodes_[j]);
      }
    } else {
      FMM_Node* parent_node;
      FMM_Node* tmp_node1;
      FMM_Node* tmp_node2;
      for(int i=0; i<n1; i++)node->SetColleague(NULL, i);
      parent_node=node->Parent();
      if(parent_node==NULL) return;
      int l=node->octant;         // l is octant
      for(int i=0; i<n1; i++) {
        tmp_node1=parent_node->Colleague(i);  // loop over parent's colleagues
        if(tmp_node1!=NULL)
          if(!tmp_node1->IsLeaf()) {
            for(int j=0; j<n2; j++) {
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
                  node->SetColleague(tmp_node2, new_indx);
              }
            }
          }
      }
    }
  }

  // generate node_list for different operators and node_data_buff (src trg information)
  void CollectNodeData(std::vector<FMM_Node*>& nodes) {
    // post-order traversal
    leafs.clear();
    nonleafs.clear();                       // input "nodes" is post-order
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
    nodesLevelOrder.clear();            // level order traversal with leafs
    nonleafsLevelOrder.clear();         // level order traversal without leafs
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
    nodesLevelOrder.push_back(
      root_node);   // level 0 root is the last one instead of the first elem in pvfmm's level order TT
    // fill in vec_list
    for(int i=0; i<nodesLevelOrder.size(); i++) {
      FMM_Node* node = nodesLevelOrder[i];
      node->idx = i;
      node->upward_equiv.Resize(NSURF);
      node->upward_equiv.SetZero();
      node->dnward_equiv.Resize(NSURF);
      node->dnward_equiv.SetZero();
    }
    size_t numNodes = nodesLevelOrder.size();
    allUpwardEquiv.ReInit(1, numNodes*NSURF);
    allDnwardEquiv.ReInit(1, numNodes*NSURF);
    int trg_dof = kernel->ker_dim[1];
    for(int i=0; i<leafs.size(); i++) {
      FMM_Node* leaf = leafs[i];
      int n_trg_val = (leaf->trg_coord.Dim()/3) * trg_dof;
      leaf->trg_value.Resize(n_trg_val);
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
    case P2P0_Type:
      if(p == NULL || !n->IsLeaf()) return;
      for(int i=0; i<n_collg; i++) {
        FMM_Node* pc = p->Colleague(i);
        if(pc!=NULL && pc->IsLeaf()) {
          rel_coord[0]=( i %3)*4-4-(p2n & 1?2:0)+1;
          rel_coord[1]=((i/3)%3)*4-4-(p2n & 2?2:0)+1;
          rel_coord[2]=((i/9)%3)*4-4-(p2n & 4?2:0)+1;
          c_hash = interacList->coord_hash(rel_coord);
          idx = interacList->hash_lut[t][c_hash];
          if(idx>=0) interac_list[idx] = pc;
        }
      }
      break;
    case P2P1_Type:
      if(!n->IsLeaf()) return;
      for(int i=0; i<n_collg; i++) {
        FMM_Node* col=(FMM_Node*)n->Colleague(i);
        if(col!=NULL && col->IsLeaf()) {
          rel_coord[0]=( i %3)-1;
          rel_coord[1]=((i/3)%3)-1;
          rel_coord[2]=((i/9)%3)-1;
          c_hash = interacList->coord_hash(rel_coord);
          idx = interacList->hash_lut[t][c_hash];
          if(idx>=0) interac_list[idx] = col;
        }
      }
      break;
    case P2P2_Type:
      if(!n->IsLeaf()) return;
      for(int i=0; i<n_collg; i++) {
        FMM_Node* col=(FMM_Node*)n->Colleague(i);
        if(col!=NULL && !col->IsLeaf()) {
          for(int j=0; j<n_child; j++) {
            rel_coord[0]=( i %3)*4-4+(j & 1?2:0)-1;
            rel_coord[1]=((i/3)%3)*4-4+(j & 2?2:0)-1;
            rel_coord[2]=((i/9)%3)*4-4+(j & 4?2:0)-1;
            c_hash = interacList->coord_hash(rel_coord);
            idx = interacList->hash_lut[t][c_hash];
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
        FMM_Node* col=(FMM_Node*)n->Colleague(i);
        if(col!=NULL && !col->IsLeaf()) {
          rel_coord[0]=( i %3)-1;
          rel_coord[1]=((i/3)%3)-1;
          rel_coord[2]=((i/9)%3)-1;
          c_hash = interacList->coord_hash(rel_coord);
          idx=interacList->hash_lut[t][c_hash];
          if(idx>=0) interac_list[idx]=col;
        }
      }
      break;
    case M2P_Type:
      if(!n->IsLeaf()) return;
      for(int i=0; i<n_collg; i++) {
        FMM_Node* col=(FMM_Node*)n->Colleague(i);
        if(col!=NULL && !col->IsLeaf()) {
          for(int j=0; j<n_child; j++) {
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
    case P2L_Type:
      if(p == NULL) return;
      for(int i=0; i<n_collg; i++) {
        FMM_Node* pc=(FMM_Node*)p->Colleague(i);
        if(pc!=NULL && pc->IsLeaf()) {
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
    std::vector<Mat_Type> interactionTypes = {P2P0_Type, P2P1_Type, P2P2_Type,
                                              M2P_Type, P2L_Type, M2L_Type
                                             };
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

  void ClearFMMData() {
    for(size_t i=0; i<nodesLevelOrder.size(); i++) {
      nodesLevelOrder[i]->upward_equiv.SetZero();
      nodesLevelOrder[i]->dnward_equiv.SetZero();
      nodesLevelOrder[i]->trg_value.SetZero();
    }
  }

  void M2LSetup(M2LData& M2Ldata) {
    std::vector<FMM_Node*>& nodes_in = nonleafsLevelOrder;
    std::vector<FMM_Node*>& nodes_out = nonleafsLevelOrder;
    size_t n_in = nodes_in.size();
    size_t n_out = nodes_out.size();
    // build ptrs of precompmat
    size_t mat_cnt = interacList->rel_coord[M2L_Type].size();
    std::vector<real_t*>
    precomp_mat;                    // vector of ptrs which points to Precomputation matrix of each M2L relative position
    for(size_t mat_id=0; mat_id<mat_cnt; mat_id++) {
      Matrix<real_t>& M = mat->mat[M2L_Type][mat_id];
      precomp_mat.push_back(&M[0][0]);                   // precomp_mat.size == M2L's numRelCoords
    }
    // calculate buff_size & numBlocks
    size_t ker_dim0 = kernel->k_m2l->ker_dim[0];
    size_t ker_dim1 = kernel->k_m2l->ker_dim[1];
    size_t n1 = MULTIPOLE_ORDER*2;
    size_t n2 = n1*n1;
    size_t n3_ = n2*(n1/2+1);
    size_t chld_cnt = 8;
    size_t fftsize = 2 * n3_ * chld_cnt;
    size_t buff_size = 1024l*1024l*1024l;    // 1Gb buffer
    size_t n_blk0 = 2*fftsize*(ker_dim0*n_in +ker_dim1*n_out)*sizeof(real_t)/buff_size;
    if(n_blk0==0) n_blk0 = 1;
    // calculate fft_dsp(fft_vec) & fft_scal
    int omp_p = omp_get_max_threads();
    std::vector<std::vector<size_t> >  fft_vec(n_blk0);
    std::vector<std::vector<size_t> > ifft_vec(n_blk0);
    std::vector<std::vector<real_t> >  fft_scl(n_blk0);
    std::vector<std::vector<real_t> > ifft_scl(n_blk0);
    std::vector<std::vector<FMM_Node*> > nodes_blk_in (n_blk0);  // node_in in each block
    std::vector<std::vector<FMM_Node*> > nodes_blk_out(n_blk0);  // node_out in each block
    std::vector<real_t> src_scal=kernel->k_m2l->src_scal;  // src_scal is 0 for Laplace
    std::vector<real_t> trg_scal=kernel->k_m2l->trg_scal;  // trg_scal is 1 for Laplace
    for(size_t i=0; i<n_in; i++) nodes_in[i]->node_id=i;
    for(size_t blk0=0; blk0<n_blk0; blk0++) {
      // prepare nodes_in_ & out_ for this block
      size_t blk0_start=(n_out* blk0   )/n_blk0;
      size_t blk0_end  =(n_out*(blk0+1))/n_blk0;
      std::vector<FMM_Node*>& nodes_in_ =nodes_blk_in [blk0];
      std::vector<FMM_Node*>& nodes_out_=nodes_blk_out[blk0];
      std::set<FMM_Node*> nodes_in;
      for(size_t i=blk0_start; i<blk0_end; i++) {
        nodes_out_.push_back(nodes_out[i]);
        std::vector<FMM_Node*>& lst=nodes_out[i]->interac_list[M2L_Type];
        for(size_t k=0; k<mat_cnt; k++) if(lst[k]!=NULL && lst[k]->pt_cnt[0]) nodes_in.insert(lst[k]);
      }
      for(typename std::set<FMM_Node*>::iterator node=nodes_in.begin(); node != nodes_in.end(); node++)
        nodes_in_.push_back(*node);
      // reevaluate buff_size
      size_t  input_dim=nodes_in_ .size()*ker_dim0*fftsize;
      size_t output_dim=nodes_out_.size()*ker_dim1*fftsize;
      size_t buffer_dim=2*(ker_dim0+ker_dim1)*fftsize*omp_p;
      if(buff_size<(input_dim + output_dim + buffer_dim)*sizeof(real_t))
        buff_size=(input_dim + output_dim + buffer_dim)*sizeof(real_t);
      // calculate fft_vec (dsp) and fft_sca
      for(size_t i=0; i<nodes_in_ .size(); i++)
        fft_vec[blk0].push_back(nodes_in_[i]->child[0]->idx * NSURF);
      for(size_t i=0; i<nodes_out_.size(); i++)
        ifft_vec[blk0].push_back(nodes_out[blk0_start+i]->child[0]->idx * NSURF);
      size_t scal_dim0=src_scal.size();
      size_t scal_dim1=trg_scal.size();
      fft_scl [blk0].resize(nodes_in_ .size()*scal_dim0);
      ifft_scl[blk0].resize(nodes_out_.size()*scal_dim1);
      for(size_t i=0; i<nodes_in_ .size(); i++) {
        size_t depth=nodes_in_[i]->depth+1;
        for(size_t j=0; j<scal_dim0; j++)
          fft_scl[blk0][i*scal_dim0+j]=powf(2.0, src_scal[j]*depth);
      }
      for(size_t i=0; i<nodes_out_.size(); i++) {
        size_t depth=nodes_out_[i]->depth+1;
        for(size_t j=0; j<scal_dim1; j++)
          ifft_scl[blk0][i*scal_dim1+j]=powf(2.0, trg_scal[j]*depth);
      }
    }
    // calculate interac_vec & interac_dsp
    std::vector<std::vector<size_t> > interac_vec(n_blk0);
    std::vector<std::vector<size_t> > interac_dsp(n_blk0);
    for(size_t blk0=0; blk0<n_blk0; blk0++) {
      std::vector<FMM_Node*>& nodes_in_ =nodes_blk_in [blk0];
      std::vector<FMM_Node*>& nodes_out_=nodes_blk_out[blk0];
      for(size_t i=0; i<nodes_in_.size(); i++) nodes_in_[i]->node_id=i;
      {
        size_t n_blk1=nodes_out_.size()*sizeof(real_t)/CACHE_SIZE;
        if(n_blk1==0) n_blk1=1;
        size_t interac_dsp_=0;
        for(size_t blk1=0; blk1<n_blk1; blk1++) {
          size_t blk1_start=(nodes_out_.size()* blk1   )/n_blk1;
          size_t blk1_end  =(nodes_out_.size()*(blk1+1))/n_blk1;
          for(size_t k=0; k<mat_cnt; k++) {
            for(size_t i=blk1_start; i<blk1_end; i++) {
              std::vector<FMM_Node*>& lst=nodes_out_[i]->interac_list[M2L_Type];
              if(lst[k]!=NULL && lst[k]->pt_cnt[0]) {
                interac_vec[blk0].push_back(lst[k]->node_id*fftsize*ker_dim0);   // node_in dspl
                interac_vec[blk0].push_back(    i          *fftsize*ker_dim1);   // node_out dspl
                interac_dsp_++;
              }
            }
            interac_dsp[blk0].push_back(interac_dsp_);
          }
        }
      }
    }
    M2Ldata.buff_size   = buff_size;
    M2Ldata.n_blk0      = n_blk0;
    M2Ldata.precomp_mat = precomp_mat;
    M2Ldata.fft_vec     = fft_vec;
    M2Ldata.ifft_vec    = ifft_vec;
    M2Ldata.fft_scl     = fft_scl;
    M2Ldata.ifft_scl    = ifft_scl;
    M2Ldata.interac_vec = interac_vec;
    M2Ldata.interac_dsp = interac_dsp;
  }

 public:
  void SetupFMM() {
    Profile::Tic("SetupFMM", true);
    {
      Profile::Tic("SetColleagues", false, 3);
      SetColleagues();
      Profile::Toc();
      Profile::Tic("CollectNodeData", false, 3);
      FMM_Node* n = PostorderFirst();
      std::vector<FMM_Node*> all_nodes;
      LEVEL = 0;
      while(n!=NULL) {
        LEVEL = n->depth > LEVEL ? n->depth : LEVEL;
        n->pt_cnt[0]=0;
        n->pt_cnt[1]=0;
        all_nodes.push_back(n);        // all_nodes: postorder tree traversal
        n = PostorderNxt(n);
      }
      CollectNodeData(all_nodes);
      Profile::Toc();
      Profile::Tic("BuildLists", false, 3);
      BuildInteracLists();
      Profile::Toc();
      Profile::Tic("M2LListSetup", false, 3);
      M2LSetup(M2Ldata);
      Profile::Toc();
      ClearFMMData();
    }
    Profile::Toc();
  }
  /* End of 2nd Part: Setup FMM */

  /* 3rd Part: Evaluation */
 private:
  void P2M() {
    #pragma omp parallel for
    for(int i=0; i<leafs.size(); i++) {
      FMM_Node* leaf = leafs[i];
      int level = leaf->depth;
      real_t scal = pow(0.5, level);    // scaling factor of UC2UE precomputation matrix
      // source charge -> check surface potential
      std::vector<real_t> checkCoord(NSURF*3);
      for(int k=0; k<NSURF; k++) {
        checkCoord[3*k+0] = upwd_check_surf[level][3*k+0] + leaf->coord[0];
        checkCoord[3*k+1] = upwd_check_surf[level][3*k+1] + leaf->coord[1];
        checkCoord[3*k+2] = upwd_check_surf[level][3*k+2] + leaf->coord[2];
      }
      kernel->k_s2m->ker_poten(&(leaf->src_coord[0]), leaf->pt_cnt[0], &(leaf->src_value[0]),
                               &checkCoord[0], NSURF, &
                               (leaf->upward_equiv[0]));  // save check potentials in upward_equiv temporarily
      // check surface potential -> equivalent surface charge
      Matrix<real_t> check(1, NSURF, &(leaf->upward_equiv[0]), true);  // check surface potential
      Matrix<real_t> buffer(1, NSURF);
      Matrix<real_t>::GEMM(buffer, check, mat->mat[M2M_V_Type][0]);
      Matrix<real_t> equiv(1, NSURF);  // equivalent surface charge
      Matrix<real_t>::GEMM(equiv, buffer, mat->mat[M2M_U_Type][0]);
      for(int k=0; k<NSURF; k++)
        leaf->upward_equiv[k] = scal * equiv[0][k];
    }
  }

  void L2P() {
    #pragma omp parallel for
    for(int i=0; i<leafs.size(); i++) {
      FMM_Node* leaf = leafs[i];
      int level = leaf->depth;
      real_t scal = pow(0.5, level);
      // check surface potential -> equivalent surface charge
      Matrix<real_t> check(1, NSURF, &(leaf->dnward_equiv[0]), true);  // check surface potential
      Matrix<real_t> buffer(1, NSURF);
      Matrix<real_t>::GEMM(buffer, check, mat->mat[L2L_V_Type][0]);
      Matrix<real_t> equiv(1, NSURF);  // equivalent surface charge
      Matrix<real_t>::GEMM(equiv, buffer, mat->mat[L2L_U_Type][0]);
      for(int k=0; k<NSURF; k++)
        leaf->dnward_equiv[k] = scal * equiv[0][k];
      // equivalent surface charge -> target potential
      std::vector<real_t> equivCoord(NSURF*3);
      for(int k=0; k<NSURF; k++) {
        equivCoord[3*k+0] = dnwd_equiv_surf[level][3*k+0] + leaf->coord[0];
        equivCoord[3*k+1] = dnwd_equiv_surf[level][3*k+1] + leaf->coord[1];
        equivCoord[3*k+2] = dnwd_equiv_surf[level][3*k+2] + leaf->coord[2];
      }
      kernel->k_l2t->ker_poten(&equivCoord[0], NSURF, &(leaf->dnward_equiv[0]),
                               &(leaf->trg_coord[0]), leaf->pt_cnt[1], &(leaf->trg_value[0]));
    }
  }

  void M2M(FMM_Node* node) {
    if(node->IsLeaf()) return;
    Matrix<real_t>& M = mat->mat[M2M_Type][7];  // 7 is the class coord, will generalize it later
    for(int octant=0; octant<8; octant++) {
      if(node->child[octant] != NULL)
        #pragma omp task untied
        M2M(node->child[octant]);
    }
    #pragma omp taskwait
    for(int octant=0; octant<8; octant++) {
      if(node->child[octant] != NULL) {
        FMM_Node* child = node->child[octant];
        std::vector<size_t>& perm_in = mat->perm_r[M2M_Type][octant].perm;
        std::vector<size_t>& perm_out = mat->perm_c[M2M_Type][octant].perm;
        Matrix<real_t> buffer_in(1, NSURF);
        Matrix<real_t> buffer_out(1, NSURF);
        for(int k=0; k<NSURF; k++) {
          buffer_in[0][k] = child->upward_equiv[perm_in[k]]; // input perm
        }
        Matrix<real_t>::GEMM(buffer_out, buffer_in, M);
        for(int k=0; k<NSURF; k++)
          node->upward_equiv[k] += buffer_out[0][perm_out[k]];
      }
    }
  }

  void M2M() {
    #pragma omp parallel
    #pragma omp single nowait
    M2M(root_node);
  }

  void L2L(FMM_Node* node) {
    if(node->IsLeaf()) return;
    Matrix<real_t>& M = mat->mat[L2L_Type][7];  // 7 is the class coord, will generalize it later
    for(int octant=0; octant<8; octant++) {
      if(node->child[octant] != NULL) {
        FMM_Node* child = node->child[octant];
        std::vector<size_t>& perm_in = mat->perm_r[L2L_Type][octant].perm;
        std::vector<size_t>& perm_out = mat->perm_c[L2L_Type][octant].perm;
        Matrix<real_t> buffer_in(1, NSURF);
        Matrix<real_t> buffer_out(1, NSURF);
        for(int k=0; k<NSURF; k++) {
          buffer_in[0][k] = node->dnward_equiv[perm_in[k]]; // input perm
        }
        Matrix<real_t>::GEMM(buffer_out, buffer_in, M);
        for(int k=0; k<NSURF; k++)
          child->dnward_equiv[k] += buffer_out[0][perm_out[k]];
      }
    }
    for(int octant=0; octant<8; octant++) {
      if(node->child[octant] != NULL)
        #pragma omp task untied
        L2L(node->child[octant]);
    }
    #pragma omp taskwait
  }

  void L2L() {
    #pragma omp parallel
    #pragma omp single nowait
    L2L(root_node);
  }

  void gatherEquiv() {
    size_t numNodes = nodesLevelOrder.size();
    #pragma omp parallel for collapse(2)
    for(int i=0; i<numNodes; i++) {
      for(int j=0; j<NSURF; j++) {
        allUpwardEquiv[0][i*NSURF+j] = nodesLevelOrder[i]->upward_equiv[j];
        allDnwardEquiv[0][i*NSURF+j] = nodesLevelOrder[i]->dnward_equiv[j];
      }
    }
  }

  void scatterEquiv() {
    size_t numNodes = nodesLevelOrder.size();
    #pragma omp parallel for collapse(2)
    for(int i=0; i<numNodes; i++) {
      for(int j=0; j<NSURF; j++) {
        nodesLevelOrder[i]->upward_equiv[j] = allUpwardEquiv[0][i*NSURF+j];
        nodesLevelOrder[i]->dnward_equiv[j] = allDnwardEquiv[0][i*NSURF+j];
      }
    }
  }

  void M2LListHadamard(size_t M_dim, std::vector<size_t>& interac_dsp,
                       std::vector<size_t>& interac_vec,
                       std::vector<real_t*>& precomp_mat, std::vector<real_t>& fft_in, Vector<real_t>& fft_out) {
    size_t chld_cnt=1UL<<3;
    size_t fftsize_in =M_dim*chld_cnt*2;
    size_t fftsize_out=M_dim*chld_cnt*2;
    int err;
    real_t * zero_vec0, * zero_vec1;
    err = posix_memalign((void**)&zero_vec0, MEM_ALIGN, fftsize_in *sizeof(real_t));
    err = posix_memalign((void**)&zero_vec1, MEM_ALIGN, fftsize_out*sizeof(real_t));
    size_t n_out=fft_out.Dim()/fftsize_out;
    #pragma omp parallel for
    for(size_t k=0; k<n_out; k++) {
      Vector<real_t> dnward_check_fft(fftsize_out, &fft_out[k*fftsize_out], false);
      dnward_check_fft.SetZero();
    }
    size_t mat_cnt=precomp_mat.size();
    size_t blk1_cnt=interac_dsp.size()/mat_cnt;
    int BLOCK_SIZE = CACHE_SIZE * 4 / sizeof(real_t);
    real_t **IN_, **OUT_;
    err = posix_memalign((void**)&IN_, MEM_ALIGN, BLOCK_SIZE*blk1_cnt*mat_cnt*sizeof(real_t*));
    err = posix_memalign((void**)&OUT_, MEM_ALIGN, BLOCK_SIZE*blk1_cnt*mat_cnt*sizeof(real_t*));
    #pragma omp parallel for
    for(size_t interac_blk1=0; interac_blk1<blk1_cnt*mat_cnt; interac_blk1++) {
      size_t interac_dsp0 = (interac_blk1==0?0:interac_dsp[interac_blk1-1]);
      size_t interac_dsp1 =                    interac_dsp[interac_blk1  ] ;
      size_t interac_cnt  = interac_dsp1-interac_dsp0;
      for(size_t j=0; j<interac_cnt; j++) {
        IN_ [BLOCK_SIZE*interac_blk1 +j]=&fft_in [interac_vec[(interac_dsp0+j)*2+0]];
        OUT_[BLOCK_SIZE*interac_blk1 +j]=&fft_out[interac_vec[(interac_dsp0+j)*2+1]];
      }
      IN_ [BLOCK_SIZE*interac_blk1 +interac_cnt]=zero_vec0;
      OUT_[BLOCK_SIZE*interac_blk1 +interac_cnt]=zero_vec1;
    }
    int omp_p=omp_get_max_threads();
    #pragma omp parallel for
    for(int pid=0; pid<omp_p; pid++) {
      size_t a=( pid   *M_dim)/omp_p;
      size_t b=((pid+1)*M_dim)/omp_p;
      for(size_t     blk1=0;     blk1<blk1_cnt;    blk1++)
        for(size_t        k=a;        k<       b;       k++)
          for(size_t mat_indx=0; mat_indx< mat_cnt; mat_indx++) {
            size_t interac_blk1 = blk1*mat_cnt+mat_indx;
            size_t interac_dsp0 = (interac_blk1==0?0:interac_dsp[interac_blk1-1]);
            size_t interac_dsp1 =                    interac_dsp[interac_blk1  ] ;
            size_t interac_cnt  = interac_dsp1-interac_dsp0;
            real_t** IN = IN_ + BLOCK_SIZE*interac_blk1;
            real_t** OUT= OUT_+ BLOCK_SIZE*interac_blk1;
            real_t* M = precomp_mat[mat_indx] + k*chld_cnt*chld_cnt*2;
            for(size_t j=0; j<interac_cnt; j+=2) {
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
                   Vector<real_t>& input_data, std::vector<real_t>& output_data) {
    size_t n1=m*2;
    size_t n2=n1*n1;
    size_t n3=n1*n2;
    size_t n3_=n2*(n1/2+1);
    size_t chld_cnt=1UL<<3;
    size_t fftsize_in =2*n3_*chld_cnt;
    int omp_p=omp_get_max_threads();
    size_t n=6*(m-1)*(m-1)+2;
    static std::vector<size_t> map;
    {
      // Remove
      size_t n_old = map.size();
      if(n_old!=n) {
        real_t c[3]= {0, 0, 0};
        std::vector<real_t> surf = surface(m, c, (real_t)(m-1), 0);
        map.resize(surf.size()/3);
        for(size_t i=0; i<map.size(); i++)
          map[i]=((size_t)(m-1-surf[i*3]+0.5))+((size_t)(m-1-surf[i*3+1]+0.5))*n1+((size_t)(
                   m-1-surf[i*3+2]+0.5))*n2;
      }
    }
    {
      // Remove
      if(!m2l_list_fft_flag) {
        int err, nnn[3]= {(int)n1, (int)n1, (int)n1};
        real_t *fftw_in, *fftw_out;
        err = posix_memalign((void**)&fftw_in,  MEM_ALIGN,   n3 *chld_cnt*sizeof(real_t));
        err = posix_memalign((void**)&fftw_out, MEM_ALIGN, 2*n3_*chld_cnt*sizeof(real_t));
        m2l_list_fftplan = fft_plan_many_dft_r2c(3, nnn, chld_cnt,
                           (real_t*)fftw_in, NULL, 1, n3,
                           (fft_complex*)(fftw_out), NULL, 1, n3_,
                           FFTW_ESTIMATE);
        free(fftw_in );
        free(fftw_out);
        m2l_list_fft_flag=true;
      }
    }
    {
      size_t n_in = fft_vec.size();
      #pragma omp parallel for
      for(int pid=0; pid<omp_p; pid++) {
        size_t node_start=(n_in*(pid  ))/omp_p;
        size_t node_end  =(n_in*(pid+1))/omp_p;
        std::vector<real_t> buffer(fftsize_in, 0);
        for(size_t node_idx=node_start; node_idx<node_end; node_idx++) {
          Matrix<real_t> upward_equiv(chld_cnt, n, &input_data[0] + fft_vec[node_idx], false);
          real_t* upward_equiv_fft =
            &output_data[fftsize_in*node_idx];  // offset ptr for node_idx in fft_in vector
          for(size_t k=0; k<n; k++) {
            size_t idx=map[k];
            int j1=0;
            for(int j0=0; j0<(int)chld_cnt; j0++)
              upward_equiv_fft[idx+j0*n3]=upward_equiv[j0][k]*fft_scal[node_idx];
          }
          fft_execute_dft_r2c(m2l_list_fftplan, (real_t*)&upward_equiv_fft[0], (fft_complex*)&buffer[0]);
          for(size_t j=0; j<n3_; j++)
            for(size_t k=0; k<chld_cnt; k++) {
              upward_equiv_fft[2*(chld_cnt*j+k)+0]=buffer[2*(n3_*k+j)+0];
              upward_equiv_fft[2*(chld_cnt*j+k)+1]=buffer[2*(n3_*k+j)+1];
            }
        }
      }
    }
  }

  void FFT_Check2Equiv(size_t m, std::vector<size_t>& ifft_vec, std::vector<real_t>& ifft_scal,
                       Vector<real_t>& input_data, Vector<real_t>& output_data) {
    size_t n1=m*2;
    size_t n2=n1*n1;
    size_t n3=n1*n2;
    size_t n3_=n2*(n1/2+1);
    size_t chld_cnt=1UL<<3;
    size_t fftsize_out=2*n3_*chld_cnt;
    int omp_p=omp_get_max_threads();
    size_t n=6*(m-1)*(m-1)+2;
    static std::vector<size_t> map;
    {
      // Remove
      size_t n_old = map.size();
      if(n_old!=n) {
        real_t c[3]= {0, 0, 0};
        std::vector<real_t> surf = surface(m, c, (real_t)(m-1), 0);
        map.resize(surf.size()/3);
        for(size_t i=0; i<map.size(); i++)
          map[i]=((size_t)(m*2-0.5-surf[i*3]))+((size_t)(m*2-0.5-surf[i*3+1]))*n1+((size_t)(
                   m*2-0.5-surf[i*3+2]))*n2;
      }
    }
    {
      // Remove
      if(!m2l_list_ifft_flag) {
        int err, nnn[3]= {(int)n1, (int)n1, (int)n1};
        real_t *fftw_in, *fftw_out;
        err = posix_memalign((void**)&fftw_in,  MEM_ALIGN, 2*n3_*chld_cnt*sizeof(real_t));
        err = posix_memalign((void**)&fftw_out, MEM_ALIGN,   n3 *chld_cnt*sizeof(real_t));
        m2l_list_ifftplan = fft_plan_many_dft_c2r(3, nnn, chld_cnt,
                            (fft_complex*)fftw_in, NULL, 1, n3_,
                            (real_t*)(fftw_out), NULL, 1, n3,
                            FFTW_ESTIMATE);
        free(fftw_in);
        free(fftw_out);
        m2l_list_ifft_flag=true;
      }
    }
    {
      size_t n_out=ifft_vec.size();
      #pragma omp parallel for
      for(int pid=0; pid<omp_p; pid++) {
        size_t node_start=(n_out*(pid  ))/omp_p;
        size_t node_end  =(n_out*(pid+1))/omp_p;
        std::vector<real_t> buffer0(fftsize_out, 0);
        std::vector<real_t> buffer1(fftsize_out, 0);
        for(size_t node_idx=node_start; node_idx<node_end; node_idx++) {
          Vector<real_t> dnward_check_fft(fftsize_out, &input_data[fftsize_out*node_idx], false);
          //real_t* dnward_check_fft = &input_data[fftsize_out*node_idx];
          Vector<real_t> dnward_equiv(n*chld_cnt, &output_data[0] + ifft_vec[node_idx], false);
          for(size_t j=0; j<n3_; j++)
            for(size_t k=0; k<chld_cnt; k++) {
              buffer0[2*(n3_*k+j)+0]=dnward_check_fft[2*(chld_cnt*j+k)+0];
              buffer0[2*(n3_*k+j)+1]=dnward_check_fft[2*(chld_cnt*j+k)+1];
            }
          fft_execute_dft_c2r(m2l_list_ifftplan, (fft_complex*)&buffer0[0], (real_t*)&buffer1[0]);
          for(size_t k=0; k<n; k++) {
            size_t idx=map[k];
            for(int j0=0; j0<(int)chld_cnt; j0++)
              dnward_equiv[n*j0+k]+=buffer1[idx+j0*n3]*ifft_scal[node_idx];
          }
        }
      }
    }
  }

  void P2P() {
    std::vector<FMM_Node*>& targets = leafs;   // leafs, assume sources == targets
    std::vector<Mat_Type> types = {P2P0_Type, P2P1_Type, P2P2_Type, P2L_Type, M2P_Type};
    #pragma omp parallel for
    for(int i=0; i<targets.size(); i++) {
      FMM_Node* target = targets[i];
      for(int k=0; k<types.size(); k++) {
        Mat_Type type = types[k];
        std::vector<FMM_Node*>& sources = target->interac_list[type];
        if (type == P2L_Type)
          if (target->pt_cnt[1] > NSURF)
            continue;
        for(int j=0; j<sources.size(); j++) {
          FMM_Node* source = sources[j];
          if (source != NULL) {
            if (type == M2P_Type)
              if (source->pt_cnt[0] > NSURF)
                continue;
            kernel->k_s2t->ker_poten(&(source->src_coord[0]), source->pt_cnt[0], &(source->src_value[0]),
                                     &(target->trg_coord[0]), target->pt_cnt[1], &(target->trg_value[0]));
          }
        }
      }
    }
  }

  void M2P() {
    std::vector<FMM_Node*>& targets = leafs;  // leafs
    #pragma omp parallel for
    for(int i=0; i<targets.size(); i++) {
      FMM_Node* target = targets[i];
      std::vector<FMM_Node*>& sources = target->interac_list[M2P_Type];
      for(int j=0; j<sources.size(); j++) {
        FMM_Node* source = sources[j];
        if (source != NULL) {
          if (source->IsLeaf() && source->pt_cnt[0]<=NSURF)
            continue;
          std::vector<real_t> sourceEquivCoord(NSURF*3);
          int level = source->depth;
          // source cell's equiv coord = relative equiv coord + cell's origin
          for(int k=0; k<NSURF; k++) {
            sourceEquivCoord[3*k+0] = upwd_equiv_surf[level][3*k+0] + source->coord[0];
            sourceEquivCoord[3*k+1] = upwd_equiv_surf[level][3*k+1] + source->coord[1];
            sourceEquivCoord[3*k+2] = upwd_equiv_surf[level][3*k+2] + source->coord[2];
          }
          kernel->k_m2t->ker_poten(&sourceEquivCoord[0], NSURF, &(source->upward_equiv[0]),
                                   &(target->trg_coord[0]), target->pt_cnt[1], &(target->trg_value[0]));
        }
      }
    }
  }

  void P2L() {
    std::vector<FMM_Node*>& targets = nodesLevelOrder;
    #pragma omp parallel for
    for(int i=0; i<targets.size(); i++) {
      FMM_Node* target = targets[i];
      if (target->IsLeaf() && target->pt_cnt[1]<=NSURF)
        continue;
      std::vector<FMM_Node*>& sources = target->interac_list[P2L_Type];
      for(int j=0; j<sources.size(); j++) {
        FMM_Node* source = sources[j];
        if (source != NULL) {
          std::vector<real_t> targetCheckCoord(NSURF*3);
          int level = target->depth;
          // target cell's check coord = relative check coord + cell's origin
          for(int k=0; k<NSURF; k++) {
            targetCheckCoord[3*k+0] = dnwd_check_surf[level][3*k+0] + target->coord[0];
            targetCheckCoord[3*k+1] = dnwd_check_surf[level][3*k+1] + target->coord[1];
            targetCheckCoord[3*k+2] = dnwd_check_surf[level][3*k+2] + target->coord[2];
          }
          kernel->k_s2l->ker_poten(&(source->src_coord[0]), source->pt_cnt[0], &(source->src_value[0]),
                                   &targetCheckCoord[0], NSURF, &(target->dnward_equiv[0]));
        }
      }
    }
  }

  void M2L(M2LData& M2Ldata) {
    size_t buff_size = M2Ldata.buff_size;
    size_t n_blk0 = M2Ldata.n_blk0;
    if(dev_buffer.Dim()<buff_size) dev_buffer.Resize(buff_size);
    int dim0 = allUpwardEquiv.dim[0];
    int dim1 = allUpwardEquiv.dim[1];
    char * buff=dev_buffer.data_ptr;
    real_t * input_data = allUpwardEquiv.data_ptr;
    real_t * output_data = allDnwardEquiv.data_ptr;
    size_t m      = MULTIPOLE_ORDER;
    size_t n1 = m * 2;
    size_t n2 = n1 * n1;
    size_t n3_ = n2 * (n1 / 2 + 1);
    size_t chld_cnt = 8;
    size_t fftsize = 2 * n3_ * chld_cnt;
    size_t M_dim = n3_;
    std::vector<real_t*> precomp_mat = M2Ldata.precomp_mat;
    std::vector<std::vector<size_t> >&  fft_vec = M2Ldata.fft_vec;
    std::vector<std::vector<size_t> >& ifft_vec = M2Ldata.ifft_vec;
    std::vector<std::vector<real_t> >&  fft_scl = M2Ldata.fft_scl;
    std::vector<std::vector<real_t> >& ifft_scl = M2Ldata.ifft_scl;
    std::vector<std::vector<size_t> >& interac_vec = M2Ldata.interac_vec;
    std::vector<std::vector<size_t> >& interac_dsp = M2Ldata.interac_dsp;
    int omp_p=omp_get_max_threads();
    for(size_t blk0=0; blk0<n_blk0; blk0++) {
      size_t n_in = fft_vec[blk0].size();  // num of nodes_in in this block
      size_t n_out=ifft_vec[blk0].size();  // num of nodes_out in this block
      size_t  input_dim=n_in *fftsize;
      size_t output_dim=n_out*fftsize;
      std::vector<real_t> fft_in(n_in * fftsize, 0);
      Vector<real_t> fft_out(output_dim, (real_t*)(buff+input_dim*sizeof(real_t)), false);
      //Vector<real_t> fft_in ( input_dim, (real_t*)buff,false);
      //std::vector<real_t> fft_out(n_out * fftsize, 0);
      Vector<real_t>  input_data_(dim0*dim1, input_data, false);    // upward equiv ptr (node_data_buff)
      Vector<real_t> output_data_(dim0*dim1, output_data, false);
      Profile::Tic("FFT_UpEquiv", false, 5);
      FFT_UpEquiv(m, fft_vec[blk0],  fft_scl[blk0],  input_data_, fft_in);
      Profile::Toc();
      Profile::Tic("M2LHadamard", false, 5);
      M2LListHadamard(M_dim, interac_dsp[blk0], interac_vec[blk0], precomp_mat, fft_in, fft_out);
      Profile::Toc();
      Profile::Tic("FFT_Check2Equiv", false, 5);
      FFT_Check2Equiv(m, ifft_vec[blk0], ifft_scl[blk0], fft_out, output_data_);
      Profile::Toc();
    }
  }

  void UpwardPass() {
    Profile::Tic("P2M", false, 5);
    P2M();
    Profile::Toc();
    Profile::Tic("M2M", false, 5);
    M2M();
    Profile::Toc();
  }

  void DownwardPass() {
    Profile::Tic("P2L", false, 5);
    P2L();
    Profile::Toc();
    Profile::Tic("M2P", false, 5);
    M2P();
    Profile::Toc();
    Profile::Tic("P2P", false, 5);
    P2P();
    Profile::Toc();
    Profile::Tic("M2L", false, 5);
    gatherEquiv();
    M2L(M2Ldata);
    scatterEquiv();
    Profile::Toc();
    Profile::Tic("L2L", false, 5);
    L2L();
    Profile::Toc();
    Profile::Tic("L2P", false, 5);
    L2P();
    Profile::Toc();
  }

 public:
  void RunFMM() {
    Profile::Tic("RunFMM", true);
    Profile::Tic("UpwardPass", false, 2);
    UpwardPass();
    Profile::Toc();
    Profile::Tic("DownwardPass", true, 2);
    DownwardPass();
    Profile::Toc();
    Profile::Toc();
  }
  /* End of 3rd part: Evaluation */

  void CheckFMMOutput(std::string t_name) {
    int np=omp_get_max_threads();
    std::vector<real_t> src_coord;
    std::vector<real_t> src_value;
    FMM_Node* n=root_node;
    while(n!=NULL) {
      if(n->IsLeaf()) {
        Vector<real_t>& coord_vec=n->src_coord;
        Vector<real_t>& value_vec=n->src_value;
        for(size_t i=0; i<coord_vec.Dim(); i++) src_coord.push_back(coord_vec[i]);
        for(size_t i=0; i<value_vec.Dim(); i++) src_value.push_back(value_vec[i]);
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
    while(n!=NULL) {
      if(n->IsLeaf()) {
        Vector<real_t>& coord_vec=n->trg_coord;
        Vector<real_t>& poten_vec=n->trg_value;
        for(size_t i=0; i<coord_vec.Dim()/3; i++) {
          if(trg_iter%step_size == 0) {
            for(int j=0; j<3        ; j++) trg_coord    .push_back(coord_vec[i*3        +j]);
            for(int j=0; j<trg_dof  ; j++) trg_poten_fmm.push_back(poten_vec[i*trg_dof  +j]);
          }
          trg_iter++;
        }
      }
      n=static_cast<FMM_Node*>(PreorderNxt(n));
    }
    size_t trg_cnt = trg_coord.size()/3;
    std::vector<real_t> trg_poten_dir(trg_cnt*trg_dof, 0);
    pvfmm::Profile::Tic("N-Body Direct", false, 1);
    #pragma omp parallel for
    for(int i=0; i<np; i++) {
      size_t a=(i*trg_cnt)/np;
      size_t b=((i+1)*trg_cnt)/np;
      kernel->ker_poten(&src_coord[0], src_cnt, &src_value[0], &trg_coord[a*3], b-a,
                        &trg_poten_dir[a*trg_dof  ]);
    }
    pvfmm::Profile::Toc();
    {
      // Remove
      real_t max_=0;
      real_t max_err=0;
      for(size_t i=0; i<trg_poten_fmm.size(); i++) {
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
