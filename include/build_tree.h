#ifndef buildtree_h
#define buildtree_h
#include <unordered_map>
#include <queue>
#include "exafmm_t.h"
#include "hilbert.h"

namespace exafmm_t {
  //! Build nodes of tree adaptively using a top-down approach based on recursion
  void buildTree(Body * bodies, Body * buffer, int begin, int end, Node * node, Nodes & nodes, const vec3 & X, real_t R, std::vector<int> &leafs_idx, Args & args, const Keys & leafkeys, int level=0, bool direction=false) {
    node->depth = level;         // depth
    node->idx = int(node-&nodes[0]);  // current node's index in nodes
    //! Create a tree node
    node->body = bodies + begin;
    if(direction) node->body = buffer + begin;
    node->numBodies = end - begin;
    node->numChilds = 0;
    for(int d=0; d<3; d++) {
      node->coord[d] = X[d]-R; 
    }
    ivec3 iX = get3DIndex(X, level);
    node->key = getKey(iX, level);
    //! Count number of bodies in each octant
    int size[8] = {0};
    vec3 x;
    for (int i=begin; i<end; i++) {
      x = bodies[i].X;
      int octant = (x[0] > X[0]) + ((x[1] > X[1]) << 1) + ((x[2] > X[2]) << 2);
      size[octant]++;
    }
    //! Exclusive scan to get offsets
    int offset = begin;
    int offsets[8], counter[8];
    for (int i=0; i<8; i++) {
      offsets[i] = offset;
      offset += size[i];
      if (size[i]) node->numChilds++;
    }
    //! If node is a leaf
    bool isLeafKey = 1;
    if (!leafkeys.empty()) {  // when leafkeys is given (when balancing tree) 
      std::set<uint64_t>::iterator it = leafkeys[level].find(node->key);
      if (it == leafkeys[level].end()) {  // if current key is not a leaf key
        isLeafKey = 0;
      }
    }
    if (end-begin<=args.ncrit && isLeafKey) {
      node->numChilds = 0;
      leafs_idx.push_back(node->idx);
      node->idx_leafs = leafs_idx.size()-1;
      if (direction) {
        for (int i=begin; i<end; i++) {
          buffer[i].X = bodies[i].X;
          buffer[i].q = bodies[i].q;
        }
      }
      return;
    }
    //! Sort bodies by octant
    for (int i=0; i<8; i++) counter[i] = offsets[i];
    for (int i=begin; i<end; i++) {
      x = bodies[i].X;
      int octant = (x[0] > X[0]) + ((x[1] > X[1]) << 1) + ((x[2] > X[2]) << 2);
      buffer[counter[octant]].X = bodies[i].X;
      buffer[counter[octant]].q = bodies[i].q;
      counter[octant]++;
    }
    //! Loop over children and recurse
    vec3 Xchild;
    assert(nodes.capacity() >= nodes.size()+node->numChilds);
    nodes.resize(nodes.size()+node->numChilds);
    Node * child = &nodes.back() - node->numChilds + 1;
    node->fchild = child;
    node->child.resize(8, NULL);
    for (int i=0, c=0; i<8; i++) {
      Xchild = X;
      real_t Rchild = R / 2;
      for (int d=0; d<3; d++) {
        Xchild[d] += Rchild * (((i & 1 << d) >> d) * 2 - 1);
      }
      if (size[i]) {
        child[c].parent = node;
        child[c].octant = i;
        node->child[i] = &child[c];
        buildTree(buffer, bodies, offsets[i], offsets[i] + size[i], &child[c++], nodes, Xchild, Rchild, leafs_idx, args, leafkeys, level+1, !direction);
      }
    }
  }

  Nodes buildTree(Bodies & bodies, std::vector<int> &leafs_idx, Args & args, const Keys & leafkeys=Keys()) {
    real_t R0 = 0.5;
    vec3 X0(0.5);
    Bodies buffer = bodies;
    Nodes nodes(1);
    nodes[0].parent = NULL;
    nodes[0].octant = 0;
    nodes.reserve(bodies.size()*(32/args.ncrit+1));
    buildTree(&bodies[0], &buffer[0], 0, bodies.size(), &nodes[0], nodes, X0, R0, leafs_idx, args, leafkeys);
    return nodes;
  }

  // Given root, generate a level-order Morton keys
  Keys breadthFirstTraversal(Node* root, std::unordered_map<uint64_t, size_t>& key2id) {
    assert(root != NULL);
    Keys keys;
    std::queue<Node*> buffer;
    buffer.push(root);
    int level = 0;
    std::set<uint64_t> keys_;
    while (!buffer.empty()) {
      Node* curr = buffer.front();
      if (curr->depth != level) {
        keys.push_back(keys_);
        keys_.clear();
        level = curr->depth;
      }
      keys_.insert(curr->key);
      key2id[curr->key] = curr->idx;
      buffer.pop();
      for (int i=0; i<curr->numChilds; i++) {
        buffer.push(curr->fchild+i);
      }
    }
    if (keys_.size())
      keys.push_back(keys_);
    return keys;
  }

  Keys balanceTree(Keys& keys, std::unordered_map<uint64_t, size_t>& key2id, Nodes& nodes) {
    int nlevels = keys.size();
    int maxlevel = nlevels - 1;
    Keys bkeys(keys.size());      // balanced Morton keys
    std::set<uint64_t> S, N;
    std::set<uint64_t>::iterator it;
    for (int l=maxlevel; l>0; --l) {
      // N <- S + nonleafs
      N.clear();
      for (it=keys[l].begin(); it!=keys[l].end(); ++it)
        if (!nodes[key2id[*it]].IsLeaf()) // choose nonleafs
          N.insert(*it); 
      N.insert(S.begin(), S.end());       // N = S + nonleafs
      S.clear();
      // S <- Parent(Colleagues(N))
      for (it=N.begin(); it!=N.end(); ++it) {
        ivec3 iX = get3DIndex(*it);       // find N's colleagues
        ivec3 ciX;
        for (int m=-1; m<=1; ++m) {
          for (int n=-1; n<=1; ++n) {
            for (int p=-1; p<=1; ++p) {
              if (m||n||p) {
                ciX[0] = iX[0] + m;
                ciX[1] = iX[1] + n;
                ciX[2] = iX[2] + p;
              }
              if (ciX[0]>=0 && ciX[0]<pow(2,l) &&  // boundary check
                  ciX[1]>=0 && ciX[1]<pow(2,l) &&
                  ciX[2]>=0 && ciX[2]<pow(2,l)) {
                uint64_t colleague = getKey(ciX, l);
                uint64_t parent = getParent(colleague);
                S.insert(parent);          // S: parent of N's colleague
              }
            }
          }
        }
      } 
      // T <- T + Children(N)
      if (l!=maxlevel) {
        std::set<uint64_t>& T = bkeys[l+1];
        for (it=N.begin(); it!=N.end(); ++it) {
          uint64_t child = getChild(*it);
          for (int i=0; i<8; ++i) {
            T.insert(child+i);
          }
        }
      }
    }
    // manually add keys for lvl 0 and 1
    bkeys[0].insert(0);
    for(int i=1; i<9; ++i) bkeys[1].insert(i);
    return bkeys;
  }

  Keys findLeafKeys(Keys& keys) {
    std::set<uint64_t>::iterator it;
    Keys leafkeys(keys.size());
    for (int l=keys.size()-1; l>=1; --l) {
      std::set<uint64_t> parentkeys = keys[l-1];
      // remove nonleaf keys
      for (it=keys[l].begin(); it!=keys[l].end(); ++it) {
        uint64_t parentkey = getParent(*it);
        std::set<uint64_t>::iterator it2 = parentkeys.find(parentkey);
        if (it2 != parentkeys.end()) parentkeys.erase(it2);
      }
      leafkeys[l-1] = parentkeys;
    }
    leafkeys[keys.size()-1] = keys.back();
    return leafkeys;
  }
}
#endif
