#ifndef buildtree_h
#define buildtree_h
#include <unordered_map>
#include <queue>
#include "exafmm_t.h"
#include "hilbert.h"

namespace exafmm_t {
  // Sort bodies in a node according to their octants
  void sortBodies(Node * node, Body * bodies, Body * buffer, int begin, int end, std::vector<int>& size, std::vector<int>& offsets) {
    // Count number of bodies in each octant
    size.resize(8, 0);
    vec3& X = node->X;
    for (int i=begin; i<end; i++) {
      vec3& x = bodies[i].X;
      int octant = (x[0] > X[0]) + ((x[1] > X[1]) << 1) + ((x[2] > X[2]) << 2);
      size[octant]++;
    }
    // Exclusive scan to get offsets
    offsets.resize(8);
    int offset = begin;
    for (int i=0; i<8; i++) {
      offsets[i] = offset;
      offset += size[i];
    }
    // Sort bodies by octant
    std::vector<int> counter(offsets);
    for (int i=begin; i<end; i++) {
      vec3& x = bodies[i].X;
      int octant = (x[0] > X[0]) + ((x[1] > X[1]) << 1) + ((x[2] > X[2]) << 2);
      buffer[counter[octant]].X = bodies[i].X;
      buffer[counter[octant]].q = bodies[i].q;
      counter[octant]++;
    }
  }

  //! Build nodes of tree adaptively using a top-down approach based on recursion
  void buildTree(Body * sources, Body * sources_buffer, int source_begin, int source_end, 
                 Body * targets, Body * targets_buffer, int target_begin, int target_end,
                 Node * node, Nodes & nodes, const vec3 & X, real_t R, 
                 std::vector<Node*> & leafs, std::vector<Node*> & nonleafs,
                 Args & args, const Keys & leafkeys, int level=0, bool direction=false) {
    //! Create a tree node
    node->level = level;         // level
    node->idx = int(node-&nodes[0]);  // current node's index in nodes
    node->fsource = sources + source_begin;
    node->ftarget = targets + target_begin;
    if(direction) {
      node->fsource = sources_buffer + source_begin;
      node->ftarget = targets_buffer + target_begin;
    }
    node->numSources = source_end - source_begin;
    node->numTargets = target_end - target_begin;
    node->numChilds = 0;
    node->X = X;
    node->R = R;
#if COMPLEX
    node->upward_equiv.resize(NSURF, complex_t(0.,0.));
    node->dnward_equiv.resize(NSURF, complex_t(0.,0.));
#else
    node->upward_equiv.resize(NSURF, 0.);
    node->dnward_equiv.resize(NSURF, 0.);
#endif
    ivec3 iX = get3DIndex(X, level);
    node->key = getKey(iX, level);

    //! If node is a leaf
    bool isLeafKey = 1;
    if (!leafkeys.empty()) {  // when leafkeys is given (when balancing tree) 
      std::set<uint64_t>::iterator it = leafkeys[level].find(node->key);
      if (it == leafkeys[level].end()) {  // if current key is not a leaf key
        isLeafKey = 0;
      }
    }
    if (node->numSources<=args.ncrit && node->numTargets<=args.ncrit && isLeafKey) {
      node->numChilds = 0;
#if COMPLEX
      node->trg_value.resize(node->numTargets*4, complex_t(0.,0.));   // initialize target result vector
#else
      node->trg_value.resize(node->numTargets*4, 0.);   // initialize target result vector
#endif
      leafs.push_back(node);
      if (direction) {
        for (int i=source_begin; i<source_end; i++) {
          sources_buffer[i].X = sources[i].X;
          sources_buffer[i].q = sources[i].q;
        }
        for (int i=target_begin; i<target_end; i++) {
          targets_buffer[i].X = targets[i].X;
        }
      }
      return;
    }
    // Sort bodies and save in buffer
    std::vector<int> source_size, source_offsets;
    std::vector<int> target_size, target_offsets;
    sortBodies(node, sources, sources_buffer, source_begin, source_end, source_size, source_offsets);
    sortBodies(node, targets, targets_buffer, target_begin, target_end, target_size, target_offsets);
    //! Loop over children and recurse
    nonleafs.push_back(node);
    node->numChilds = 8;
    vec3 Xchild;
    assert(nodes.capacity() >= nodes.size()+node->numChilds);
    nodes.resize(nodes.size()+node->numChilds);
    Node * child = &nodes.back() - node->numChilds + 1;
    node->fchild = child;
    node->child.resize(8, nullptr);
    for (int i=0, c=0; i<8; i++) {
      Xchild = X;
      real_t Rchild = R / 2;
      for (int d=0; d<3; d++) {
        Xchild[d] += Rchild * (((i & 1 << d) >> d) * 2 - 1);
      }
      child[c].parent = node;
      child[c].octant = i;
      node->child[i] = &child[c];
      buildTree(sources_buffer, sources, source_offsets[i], source_offsets[i] + source_size[i],
                targets_buffer, targets, target_offsets[i], target_offsets[i] + target_size[i],
                &child[c++], nodes, Xchild, Rchild, leafs, nonleafs,
                args, leafkeys, level+1, !direction);
    }
  }

  Nodes buildTree(Bodies & sources, Bodies & targets, std::vector<Node*> & leafs, std::vector<Node*> & nonleafs, Args & args, const Keys & leafkeys=Keys()) {
    real_t R0 = 0.5;
    vec3 X0(0.5);
    Bodies sources_buffer = sources;
    Bodies targets_buffer = targets;
    Nodes nodes(1);
    nodes[0].parent = nullptr;
    nodes[0].octant = 0;
    nodes.reserve((sources.size()+targets.size()) * (32/args.ncrit+1));
    buildTree(&sources[0], &sources_buffer[0], 0, sources.size(), 
              &targets[0], &targets_buffer[0], 0, targets.size(),
              &nodes[0], nodes, X0, R0, leafs, nonleafs, args, leafkeys);
    return nodes;
  }

  // Given root, generate a level-order Morton keys
  Keys breadthFirstTraversal(Node* root, std::unordered_map<uint64_t, size_t>& key2id) {
    assert(root);
    Keys keys;
    std::queue<Node*> buffer;
    buffer.push(root);
    int level = 0;
    std::set<uint64_t> keys_;
    while (!buffer.empty()) {
      Node* curr = buffer.front();
      if (curr->level != level) {
        keys.push_back(keys_);
        keys_.clear();
        level = curr->level;
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
