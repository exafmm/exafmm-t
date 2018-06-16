#ifndef fmm_tree 
#define fmm_tree
#include <queue>
#include "exafmm_t.h"
#include "kernel.h"
#include "interaction_list.h"

namespace exafmm_t {
  // Construct list of leafs, nonleafs and initialize members
  void CollectNodeData() {
    leafs.clear();
    nonleafs.clear();
    allnodes.clear();
    std::queue<Node*> nodesQueue;
    nodesQueue.push(root_node);
    while (!nodesQueue.empty()) {
      Node* curr = nodesQueue.front();
      nodesQueue.pop();
      if (curr != root_node)  allnodes.push_back(curr);
      if (!curr->IsLeaf()) nonleafs.push_back(curr);
      else leafs.push_back(curr);
      for (int i=0; i<8; i++) {
        Node* child = curr->Child(i);
        if (child!=NULL) nodesQueue.push(child);
      }
    }
    allnodes.push_back(root_node);   // level 0 root is the last one

    for (long i=0; i<leafs.size(); i++) {
      Node* leaf = leafs[i];
      leaf->pt_trg.resize(leaf->numBodies * 4);
    }

    for(long i=0; i<allnodes.size(); i++) {
      Node* node = allnodes[i];
      node->idx = i;
    }
    LEVEL = leafs.back()->depth;
  }


  void ClearFMMData() {
    for(size_t i=0; i<allnodes.size(); i++) {
      RealVec& upward_equiv = allnodes[i]->upward_equiv;
      RealVec& dnward_equiv = allnodes[i]->dnward_equiv;
      RealVec& pt_trg = allnodes[i]->pt_trg;
      std::fill(upward_equiv.begin(), upward_equiv.end(), 0);
      std::fill(dnward_equiv.begin(), dnward_equiv.end(), 0);
      std::fill(pt_trg.begin(), pt_trg.end(), 0);
    }
  }

  void SetupFMM(Nodes& nodes) {
    Profile::Tic("SetupFMM", true);
    SetColleagues(nodes);
    Profile::Tic("BuildLists", false, 3);
    BuildInteracLists(nodes);
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

  void CheckFMMOutput(std::vector<Node*>& leafs) {
    int numTargets = 10;
    int stride = leafs.size() / numTargets;
    Nodes targets;
    for(size_t i=0; i<numTargets; i++) {
      targets.push_back(*(leafs[i*stride]));
    }
    Nodes targets2 = targets;    // used for direct summation
#pragma omp parallel for
    for(size_t i=0; i<targets2.size(); i++) {
      Node *target = &targets2[i];
      std::fill(target->pt_trg.begin(), target->pt_trg.end(), 0.);
      for(size_t j=0; j<leafs.size(); j++) {
        gradientP2P(leafs[j]->pt_coord, leafs[j]->pt_src, target->pt_coord, target->pt_trg);
      }
    }

    real_t p_diff = 0, p_norm = 0, g_diff = 0, g_norm = 0;
    for(size_t i=0; i<targets.size(); i++) {
      p_norm += targets2[i].pt_trg[0] * targets2[i].pt_trg[0];
      p_diff += (targets2[i].pt_trg[0] - targets[i].pt_trg[0]) * (targets2[i].pt_trg[0] - targets[i].pt_trg[0]);
      for(int d=1; d<4; d++) {
        g_diff += (targets2[i].pt_trg[d] - targets[i].pt_trg[d]) * (targets2[i].pt_trg[d] - targets[i].pt_trg[d]);
        g_norm += targets2[i].pt_trg[d] * targets2[i].pt_trg[d];
      }
    }
    std::cout << std::setw(20) << std::left << "Potn Error" << " : " << std::scientific << sqrt(p_diff/p_norm) << std::endl;
    std::cout << std::setw(20) << std::left << "Grad Error" << " : " << std::scientific << sqrt(g_diff/g_norm) << std::endl;
  }
}//end namespace
#endif
