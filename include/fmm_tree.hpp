#ifndef _PVFMM_FMM_TREE_HPP_
#define _PVFMM_FMM_TREE_HPP_
#include "intrinsics.h"
#include "pvfmm.h"
#include <queue>
#include "geometry.h"
#include "precomp_mat.hpp"
#include "build_tree.h"

namespace pvfmm {
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
      node = node->parent;
      if(node==NULL) return NULL;
      for(; i<n; i++)
        if(node->Child(i)!=NULL)
          return node->Child(i);
    }
  }

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
      parent_node = node->parent;
      if(parent_node==NULL) return;
      int l=node->octant;         // l is octant
      for(int i=0; i<n1; i++) {
        tmp_node1 = parent_node->colleague[i];  // loop over parent's colleagues
        if(tmp_node1!=NULL && !tmp_node1->IsLeaf()) {
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

  // Construct list of leafs, nonleafs and initialize members
  void CollectNodeData() {
    leafs.clear();
    nonleafs.clear();
    allnodes.clear();
    std::queue<FMM_Node*> nodesQueue;
    nodesQueue.push(root_node);
    while (!nodesQueue.empty()) {
      FMM_Node* curr = nodesQueue.front();
      nodesQueue.pop();
      if (curr != root_node)  allnodes.push_back(curr);
      if (!curr->IsLeaf()) nonleafs.push_back(curr);
      else leafs.push_back(curr);
      for (int i=0; i<8; i++) {
        FMM_Node* child = curr->Child(i);
        if (child!=NULL) nodesQueue.push(child);
      }
    }
    allnodes.push_back(root_node);   // level 0 root is the last one

    for (long i=0; i<leafs.size(); i++) {
      FMM_Node* leaf = leafs[i];
      leaf->pt_trg.resize(leaf->numBodies * TRG_DIM);
    }

    for(long i=0; i<allnodes.size(); i++) {
      FMM_Node* node = allnodes[i];
      node->idx = i;
    }
    size_t numNodes = allnodes.size();
    allUpwardEquiv.resize(numNodes*NSURF);
    allDnwardEquiv.resize(numNodes*NSURF);
    LEVEL = leafs.back()->depth;
  }


  void ClearFMMData() {
    for(size_t i=0; i<allnodes.size(); i++) {
      std::vector<real_t>& upward_equiv = allnodes[i]->upward_equiv;
      std::vector<real_t>& dnward_equiv = allnodes[i]->dnward_equiv;
      std::vector<real_t>& pt_trg = allnodes[i]->pt_trg;
      std::fill(upward_equiv.begin(), upward_equiv.end(), 0);
      std::fill(dnward_equiv.begin(), dnward_equiv.end(), 0);
      std::fill(pt_trg.begin(), pt_trg.end(), 0);
    }
  }

  void M2LSetup(M2LData& M2Ldata) {
    std::vector<FMM_Node*>& nodes_in = nonleafs;
    std::vector<FMM_Node*>& nodes_out = nonleafs;
    size_t n_in = nodes_in.size();
    size_t n_out = nodes_out.size();
    // build ptrs of precompmat
    size_t mat_cnt = rel_coord[M2L_Type].size();
    std::vector<real_t*> precomp_mat;                    // vector of ptrs which points to Precomputation matrix of each M2L relative position
    for(size_t mat_id=0; mat_id<mat_cnt; mat_id++) {
      Matrix<real_t>& M = mat_M2L[mat_id];
      precomp_mat.push_back(&M[0][0]);                   // precomp_mat.size == M2L's numRelCoords
    }
    // calculate buff_size & numBlocks
    size_t n1 = MULTIPOLE_ORDER*2;
    size_t n2 = n1*n1;
    size_t n3_ = n2*(n1/2+1);
    size_t chld_cnt = 8;
    size_t fftsize = 2 * n3_ * chld_cnt;
    size_t buff_size = 1024l*1024l*1024l;    // 1Gb buffer
    size_t n_blk0 = 2*fftsize*(SRC_DIM*n_in +POT_DIM*n_out)*sizeof(real_t)/buff_size;
    if(n_blk0==0) n_blk0 = 1;
    // calculate fft_dsp(fft_vec) & fft_scal
    int omp_p = omp_get_max_threads();
    std::vector<std::vector<size_t> >  fft_vec(n_blk0);
    std::vector<std::vector<size_t> > ifft_vec(n_blk0);
    std::vector<std::vector<real_t> >  fft_scl(n_blk0);
    std::vector<std::vector<real_t> > ifft_scl(n_blk0);
    std::vector<std::vector<FMM_Node*> > nodes_blk_in (n_blk0);  // node_in in each block
    std::vector<std::vector<FMM_Node*> > nodes_blk_out(n_blk0);  // node_out in each block
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
        for(size_t k=0; k<mat_cnt; k++) if(lst[k]!=NULL) nodes_in.insert(lst[k]);
      }
      for(typename std::set<FMM_Node*>::iterator node=nodes_in.begin(); node != nodes_in.end(); node++)
        nodes_in_.push_back(*node);
      // calculate fft_vec (dsp) and fft_sca
      for(size_t i=0; i<nodes_in_ .size(); i++)
        fft_vec[blk0].push_back(nodes_in_[i]->child[0]->idx * NSURF);  // nodes_in_ is local to the current block
      for(size_t i=0; i<nodes_out_.size(); i++)
        ifft_vec[blk0].push_back(nodes_out[blk0_start+i]->child[0]->idx * NSURF); // nodes_out here is global
      fft_scl [blk0].resize(nodes_in_ .size());
      ifft_scl[blk0].resize(nodes_out_.size());
      for(size_t i=0; i<nodes_in_ .size(); i++) {
        size_t depth=nodes_in_[i]->depth+1;
        fft_scl[blk0][i]=1;
      }
      for(size_t i=0; i<nodes_out_.size(); i++) {
        size_t depth=nodes_out_[i]->depth+1;
        ifft_scl[blk0][i]=powf(2.0, depth);
      }
    }
    // calculate interac_vec & interac_dsp
    std::vector<std::vector<size_t> > interac_vec(n_blk0);
    std::vector<std::vector<size_t> > interac_dsp(n_blk0);
    for(size_t blk0=0; blk0<n_blk0; blk0++) {
      std::vector<FMM_Node*>& nodes_in_ =nodes_blk_in [blk0];
      std::vector<FMM_Node*>& nodes_out_=nodes_blk_out[blk0];
      for(size_t i=0; i<nodes_in_.size(); i++) nodes_in_[i]->node_id=i;
      size_t n_blk1=nodes_out_.size()*sizeof(real_t)/CACHE_SIZE;
      if(n_blk1==0) n_blk1=1;
      size_t interac_dsp_=0;
      for(size_t blk1=0; blk1<n_blk1; blk1++) {
        size_t blk1_start=(nodes_out_.size()* blk1   )/n_blk1;
        size_t blk1_end  =(nodes_out_.size()*(blk1+1))/n_blk1;
        for(size_t k=0; k<mat_cnt; k++) {
          for(size_t i=blk1_start; i<blk1_end; i++) {
            std::vector<FMM_Node*>& lst=nodes_out_[i]->interac_list[M2L_Type];
            if(lst[k]!=NULL) {
              interac_vec[blk0].push_back(lst[k]->node_id*fftsize*SRC_DIM);   // node_in dspl
              interac_vec[blk0].push_back(    i          *fftsize*POT_DIM);   // node_out dspl
              interac_dsp_++;
            }
          }
          interac_dsp[blk0].push_back(interac_dsp_);
        }
      }
    }
    M2Ldata.n_blk0      = n_blk0;
    M2Ldata.precomp_mat = precomp_mat;
    M2Ldata.fft_vec     = fft_vec;
    M2Ldata.ifft_vec    = ifft_vec;
    M2Ldata.fft_scl     = fft_scl;
    M2Ldata.ifft_scl    = ifft_scl;
    M2Ldata.interac_vec = interac_vec;
    M2Ldata.interac_dsp = interac_dsp;
  }

  void SetupFMM(FMM_Nodes& cells) {
    Profile::Tic("SetupFMM", true);
    Profile::Tic("SetColleagues", false, 3);
    SetColleagues();
    Profile::Toc();
    Profile::Tic("BuildLists", false, 3);
    BuildInteracLists(cells);
    Profile::Toc();
    Profile::Tic("CollectNodeData", false, 3);
    CollectNodeData();
    Profile::Toc();
    Profile::Tic("M2LListSetup", false, 3);
    M2LSetup(M2Ldata);
    Profile::Toc();
    ClearFMMData();
    Profile::Toc();
  }

  void UpwardPass() {
    Profile::Tic("P2M", false, 5);
    P2M();
    Profile::Toc();
    Profile::Tic("M2M", false, 5);
    #pragma omp parallel
    #pragma omp single nowait
    M2M(root_node);
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
    M2L(M2Ldata);
    Profile::Toc();
    Profile::Tic("L2L", false, 5);
    #pragma omp parallel
    #pragma omp single nowait
    L2L(root_node);
    Profile::Toc();
    Profile::Tic("L2P", false, 5);
    L2P();
    Profile::Toc();
  }

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
        std::vector<real_t>& coord_vec=n->pt_coord;
        std::vector<real_t>& value_vec=n->pt_src;
        for(size_t i=0; i<coord_vec.size(); i++) src_coord.push_back(coord_vec[i]);
        for(size_t i=0; i<value_vec.size(); i++) src_value.push_back(value_vec[i]);
      }
      n=static_cast<FMM_Node*>(PreorderNxt(n));
    }
    size_t src_cnt = src_coord.size()/3;
    int trg_dof = TRG_DIM;
    std::vector<real_t> trg_coord;
    std::vector<real_t> trg_poten_fmm;
    size_t step_size = 1 + src_cnt*src_cnt*1e-9;
    n = root_node;
    long long trg_iter=0;
    while(n!=NULL) {
      if(n->IsLeaf()) {
        std::vector<real_t>& coord_vec=n->pt_coord;
        std::vector<real_t>& poten_vec=n->pt_trg;
        for(size_t i=0; i<coord_vec.size()/3; i++) {
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
      gradientP2P(&src_coord[0], src_cnt, &src_value[0], &trg_coord[a*3], b-a,
                        &trg_poten_dir[a*trg_dof  ]);
    }
    pvfmm::Profile::Toc();
    real_t p_diff = 0, p_norm = 0, g_diff = 0, g_norm=0;
    assert(trg_poten_dir.size() == trg_poten_fmm.size());
    for(size_t i=0; i<trg_poten_fmm.size(); i+=4) {
      p_diff += (trg_poten_dir[i]-trg_poten_fmm[i])*(trg_poten_dir[i]-trg_poten_fmm[i]);
      p_norm += trg_poten_dir[i] * trg_poten_dir[i];
      for (int d=1; d<4; d++) {
        g_diff += (trg_poten_dir[i+d]-trg_poten_fmm[i+d])*(trg_poten_dir[i+d]-trg_poten_fmm[i+d]);
        g_norm += trg_poten_dir[i+d] * trg_poten_dir[i+d];
      }
    }
    std::cout << std::setw(20) << std::left << "Potn Error" << " : " << std::scientific << sqrt(p_diff/p_norm) << std::endl;
    std::cout << std::setw(20) << std::left << "Grad Error" << " : " << std::scientific << sqrt(g_diff/g_norm) << std::endl;
  }
}//end namespace
#endif //_PVFMM_FMM_TREE_HPP_
