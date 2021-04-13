#ifndef local_essential_tree_h
#define local_essential_tree_h
#include <map>
#include "alltoall.h"
#include "hilbert.h"
#include "timer.h"
#define SEND_ALL 0 //! Set to 1 for debugging
#define VERBOSE 1

namespace exafmm_t {
  using std::cout;
  using std::endl;
  template <typename T> using BodyMap = std::multimap<uint64_t, Body<T>>;
  template <typename T> using NodeMap = std::map<uint64_t, Node<T>>;

  //! Distance between cell center and edge of a remote domain
  template <typename T>
  real_t getDistance(Node<T>* C, int irank, std::vector<int>& OFFSET, int LEVEL, vec3 X0, real_t R0) {
    real_t distance = R0;
    real_t R = R0 / (1 << LEVEL);
    for (int key=OFFSET[irank]; key<OFFSET[irank+1]; key++) {
      ivec3 iX = get3DIndex(key, LEVEL);
      vec3 X = getCoordinates(iX, LEVEL, X0, R0);
      vec3 Xmin = X - R;
      vec3 Xmax = X + R;
      vec3 dX;
			for (int d=0; d<3; d++) {
				dX[d] = (C->x[d] > Xmax[d]) * (C->x[d] - Xmax[d]) + (C->x[d] < Xmin[d]) * (C->x[d] - Xmin[d]);
			}
			distance = std::min(distance, norm(dX));
    }
    return distance;
  }

  //! Recursive call to pre-order tree traversal for selecting cells to send
  template <typename T>
  void selectCells(Node<T>* Cj, int irank, Bodies<T>& bodyBuffer, std::vector<int> & sendBodyCount,
                   Nodes<T>& cellBuffer, std::vector<int> & sendCellCount,
                   std::vector<int>& OFFSET, int LEVEL, vec3 X0, real_t R0) {
    real_t R = getDistance(Cj, irank, OFFSET, LEVEL, X0, R0);
    real_t THETA = 0.5;
    real_t R2 = R * R * THETA * THETA;
    sendCellCount[irank]++;
    cellBuffer.push_back(*Cj);

    if (R2 <= (Cj->r + Cj->r) * (Cj->r + Cj->r)) {   // if near range
      if (Cj->is_leaf) {
        sendBodyCount[irank] += Cj->nsrcs;
        for (int b=0; b<Cj->nsrcs; b++) {
          bodyBuffer.push_back(Cj->first_src[b]);
        }
      } else {
        for (auto & child : Cj->children) {
          selectCells(child, irank, bodyBuffer, sendBodyCount, cellBuffer, sendCellCount,
                      OFFSET, LEVEL, X0, R0);
        }
      }
    }
  }

  template <typename T>
  void whatToSend(Nodes<T> & cells, Bodies<T> & bodyBuffer, std::vector<int> & sendBodyCount,
                  Nodes<T> & cellBuffer, std::vector<int> & sendCellCount,
                  std::vector<int>& OFFSET, int LEVEL, vec3 X0, real_t R0) {
#if SEND_ALL //! Send everything (for debugging)
    for (int irank=0; irank<MPISIZE; irank++) {
      sendCellCount[irank] = cells.size();
      for (size_t i=0; i<cells.size(); i++) {
        if (cells[i].is_leaf) {
          sendBodyCount[irank] += cells[i].nsrcs;
          for (int b=0; b<cells[i].nsrcs; b++) {
            bodyBuffer.push_back(cells[i].first_src[b]);
          }
        }
      }
      cellBuffer.insert(cellBuffer.end(), cells.begin(), cells.end());
    }
#else //! Send only necessary cells
    for (int irank=0; irank<MPISIZE; irank++) {
      selectCells(&cells[0], irank, bodyBuffer, sendBodyCount, cellBuffer, sendCellCount,
                  OFFSET, LEVEL, X0, R0);
    }
#endif
  }

  // Reapply ncrit to account for bodies from other ranks
  template <typename T>
  void reapplyNcrit(BodyMap<T> & bodyMap, NodeMap<T> & nodeMap, uint64_t key,
                    int ncrit, int nsurf, vec3 x0, real_t r0) {
    // recursion exit condition: key is leaf
    // that is, nsrcs < ncrit and no children in nodeMap
    bool noChildSent = true;
    for (int i=0; i<8; i++) {
      uint64_t childKey = getChild(key) + i;
      if (nodeMap.find(childKey) != nodeMap.end()) noChildSent = false;
    }
    if (nodeMap[key].nsrcs <= ncrit && noChildSent) {
      nodeMap[key].is_leaf = true;
      nodeMap[key].nsrcs = bodyMap.count(key);
      return;
    }

    // current node K is not a leaf. If K happens to be a leaf in
    // some ranks, we should update bodyMap by assigning K's sources with
    // new keys that correspond to their octants
    nodeMap[key].is_leaf = false;
    int level = getLevel(key);
    std::vector<int> counter(8, 0);
    auto range = bodyMap.equal_range(key);

    Bodies<T> bodies(bodyMap.count(key));
    size_t b = 0;
    for (auto B=range.first; B!=range.second; B++, b++) {
      bodies[b] = B->second;
    }
    for (b=0; b<bodies.size(); b++) {
      ivec3 iX = get3DIndex(bodies[b].X, level+1, x0, r0);
      uint64_t childKey = getKey(iX, level+1);
      int octant = getOctant(childKey);
      counter[octant]++;
      bodies[b].key = childKey;
      bodyMap.insert(std::pair<uint64_t, Body<T>>(childKey, bodies[b]));
    }
    if (bodyMap.count(key) != 0) bodyMap.erase(key);

    // and then update nodeMap
    for (int i=0; i<8; i++) {
      uint64_t childKey = getChild(key) + i;

      if (nodeMap.find(childKey) == nodeMap.end()) {  // if child doesn't exist
        Node<T> node;
        node.nsrcs = counter[i];
        ivec3 iX = get3DIndex(childKey);
        node.x = getCoordinates(iX, level+1, x0, r0);
        node.r = r0 / (1 << (level+1));
        node.key = childKey;
        node.up_equiv.resize(nsurf, 0.0);
        // node.L.resize(nsurf, 0.0);
        nodeMap[childKey] = node;
      } else if (counter[i] != 0) { // if child exists, need to recompute up_equiv
        nodeMap[childKey].nsrcs += counter[i];
        nodeMap[childKey].up_equiv.resize(nsurf, 0);
      }
    
      if (nodeMap.find(childKey) != nodeMap.end()) {
        reapplyNcrit(bodyMap, nodeMap, childKey, ncrit, nsurf, x0, r0);
      }
    }
    //! Update number of sources
    int nsrcs = 0;
    for (int i=0; i<8; i++) {
      uint64_t childKey = getChild(key) + i;
      if (nodeMap.find(childKey) != nodeMap.end()) {
        nsrcs += nodeMap[childKey].nsrcs;
      }
    }
    nodeMap[key].nsrcs = nsrcs;
  }

  //! Check integrity of local essential tree
  template <typename T>
  void sanityCheck(BodyMap<T> & bodyMap, NodeMap<T> & nodeMap, uint64_t key) {
    Node<T> node = nodeMap[key];
    assert(node.key == key);
    // verify nsrcs in leaf
    if (node.is_leaf)
      assert(node.nsrcs == int(bodyMap.count(key)));
    if (bodyMap.count(key) != 0) {
      assert(node.is_leaf);   // if bodyMap[key] not empty, key must be leaf
      auto range = bodyMap.equal_range(key);
      for (auto B=range.first; B!=range.second; B++) {
        assert(B->second.key == key);
      }
    }
    // verify nsrcs in non-leaf
    int nsrcs = 0;
    for (int i=0; i<8; i++) {
      uint64_t childKey = getChild(key) + i;
      if (nodeMap.find(childKey) != nodeMap.end()) {
        sanityCheck(bodyMap, nodeMap, childKey);
        nsrcs += nodeMap[childKey].nsrcs;
      }
    }
    if (!node.is_leaf)
      assert((node.nsrcs == nsrcs));
  }

  //! Build cells of LET recursively
  template <typename T>
  void buildCells(BodyMap<T> & bodyMap, NodeMap<T> & nodeMap, uint64_t key, Bodies<T> & bodies, Node<T> * node, Nodes<T> & nodes) {
    *node = nodeMap[key];
    node->idx = int(node-&nodes[0]);  // current node's index in nodes
    node->level = getLevel(key);
    if (bodyMap.count(key) != 0) {   // if node is leaf & has sources in it
      auto range = bodyMap.equal_range(key);
      bodies.resize(bodies.size()+node->nsrcs);
      Body<T>* first_src = &bodies.back() - node->nsrcs + 1;
      node->first_src = first_src;
      int b = 0;
      for (auto B=range.first; B!=range.second; B++, b++) {
        first_src[b] = B->second;
      }
    } else {
      node->first_src = nullptr;
    }
    if (!node->is_leaf) {   // if node is not a leaf
      nodes.resize(nodes.size()+NCHILD);  // add all 8 children
      node->children.resize(8, nullptr);
      Node<T>* child = &nodes.back() - NCHILD + 1;
      for (int c=0; c<8; c++) {
        uint64_t childKey = getChild(key) + c;
        assert(nodeMap.find(childKey) != nodeMap.end());  // must have all 8 chilren in nodeMap
        node->children[c] = &child[c];
        child[c].parent = node;
        buildCells(bodyMap, nodeMap, childKey, bodies, &child[c], nodes);
      }
    }
    /*else {
      node->child = nullptr;
    }*/
    //if (!node->is_leaf)
    //  node->body = node->child->body;
  }

  //! MPI communication for local essential tree
  template <typename T>
  void localEssentialTree(Bodies<T>& sources, Bodies<T>& targets, Nodes<T>& nodes,
                          NodePtrs<T>& leafs, NodePtrs<T>& nonleafs,
                          FmmBase<T>& fmm, std::vector<int>& OFFSET) {
    vec3 x0 = nodes[0].x;
    real_t r0 = nodes[0].r;
    int nsurf = nodes[0].up_equiv.size();
    std::vector<int> sendBodyCount(MPISIZE, 0);
    std::vector<int> recvBodyCount(MPISIZE, 0);
    std::vector<int> sendBodyDispl(MPISIZE, 0);
    std::vector<int> recvBodyDispl(MPISIZE, 0);
    std::vector<int> sendCellCount(MPISIZE, 0);
    std::vector<int> recvCellCount(MPISIZE, 0);
    std::vector<int> sendCellDispl(MPISIZE, 0);
    std::vector<int> recvCellDispl(MPISIZE, 0);
    Bodies<T> sendBodies, recvBodies;
    Nodes<T> sendCells, recvCells;
    //! Decide which nodes & bodies to send
    whatToSend(nodes, sendBodies, sendBodyCount, sendCells, sendCellCount,
               OFFSET, fmm.depth, x0, r0);
    //! Use alltoall to get recv count and calculate displacement (defined in alltoall.h)
    getCountAndDispl(sendBodyCount, sendBodyDispl, recvBodyCount, recvBodyDispl);
    getCountAndDispl(sendCellCount, sendCellDispl, recvCellCount, recvCellDispl);
    //! Alltoallv for nodes (defined in alltoall.h)
    alltoallCells(sendCells, sendCellCount, sendCellDispl, recvCells, recvCellCount, recvCellDispl);
    //! Alltoallv for sources (defined in alltoall.h)
    alltoallBodies(sendBodies, sendBodyCount, sendBodyDispl, recvBodies, recvBodyCount, recvBodyDispl);

#if VERBOSE
    printMPI("after sending sources and nodes");
    for (int i=0; i<MPISIZE; i++)
      printMPI("recv node count from rank "+std::to_string(i), recvCellCount[i]);
    for (int i=0; i<MPISIZE; i++)
      printMPI("recv source count from rank "+std::to_string(i), recvBodyCount[i]);
#endif
    // Build local essential tree
    // create bodyMap and nodeMap
    BodyMap<T> bodyMap;
    NodeMap<T> nodeMap;
    // insert bodies to bodyMap
    for (size_t i=0; i<recvBodies.size(); i++) {
      bodyMap.insert(std::pair<uint64_t, Body<T>>(recvBodies[i].key, recvBodies[i]));
    }
    // insert/merge nodes (NodeBase) to nodeMap
    for (size_t i=0; i<recvCells.size(); i++) {
      uint64_t key = recvCells[i].key;    // key with level offset
      if (nodeMap.find(key) == nodeMap.end()) {
        nodeMap[key] = recvCells[i];
      } else {
        for (int n=0; n<nsurf; n++) {
          nodeMap[key].up_equiv[n] += recvCells[i].up_equiv[n];
        }
        nodeMap[key].nsrcs += recvCells[i].nsrcs;
      }
    }

    printMPI("LET root nsrcs", nodeMap[0].nsrcs);
    printMPI("LET root monopole", nodeMap[0].up_equiv[0]);

    printMPI("before reapply ncrit");
    printMPI("size of nodeMap", nodeMap.size());

    //! Reapply Ncrit recursively to account for bodies from other ranks
    reapplyNcrit(bodyMap, nodeMap, 0, fmm.ncrit, nsurf, x0, r0);

    printMPI("after reapply ncrit");
    printMPI("size of nodeMap", nodeMap.size());

    // sanity check
    sanityCheck(bodyMap, nodeMap, 0);

    // build tree using nodeMap
    sources.clear();
    sources.reserve(bodyMap.size());
    nodes.reserve(nodeMap.size());
    nodes.resize(1);
    buildCells(bodyMap, nodeMap, 0, sources, &nodes[0], nodes);

    printMPI("build LET");
    printMPI("size of nodes", nodes.size());
  }
}
#endif
