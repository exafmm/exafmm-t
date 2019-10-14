#ifndef build_tree_h
#define build_tree_h
#include <unordered_map>
#include <queue>
#include "exafmm_t.h"
#include "hilbert.h"

namespace exafmm_t {
  /**
   * @brief Get bounding box of sources and targets
   *
   * @param sources Vector of sources
   * @param targets Vector of targets
   * @param x0 Coordinates of the center of the bounding box
   * @param r0 Radius of the bounding box
   */
  template <typename T>
  void get_bounds(const Bodies<T>& sources, const Bodies<T>& targets, vec3& x0, real_t& r0) {
    vec3 Xmin = sources[0].X;
    vec3 Xmax = sources[0].X;
    for (size_t b=0; b<sources.size(); ++b) {
      Xmin = min(sources[b].X, Xmin);
      Xmax = max(sources[b].X, Xmax);
    }
    for (size_t b=0; b<targets.size(); ++b) {
      Xmin = min(targets[b].X, Xmin);
      Xmax = max(targets[b].X, Xmax);
    }
    x0 = (Xmax + Xmin) / 2;
    r0 = fmax(max(X0-Xmin), max(Xmax-X0));
    r0 *= 1.00001;
  } 

  /**
   * @brief Sort a chunk of bodies in a node according to their octants
   *
   * @param node The node that bodies are in 
   * @param bodies The bodies to be sorted
   * @param buffer The sorted bodies
   * @param begin Begin index of the chunk 
   * @param end End index of the chunk
   * @param size Vector of the counts of bodies in each octant after
   * @param offsets Vector of the offsets of sorted bodies in each octant
   */
  template <typename T>
  void sort_bodies(Node<T>* const node, Body<T>* const bodies, Body<T>* const buffer,
                   int begin, int end, std::vector<int>& size, std::vector<int>& offsets) {
    // Count number of bodies in each octant
    size.resize(8, 0);
    vec3 X = node->x;  // the center of the node
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
#if SORT_BACK
      buffer[counter[octant]].ibody = bodies[i].ibody;
#endif
      counter[octant]++;
    }
  }

  //! Build nodes of tree adaptively using a top-down approach based on recursion
  template <typename T>
  void build_tree(Body<T>* sources, Body<T>* sources_buffer, int source_begin, int source_end, 
                  Body<T>* targets, Body<T>* targets_buffer, int target_begin, int target_end,
                  Node<T>* node, Nodes<T>& nodes, NodePtrs<T>& leafs, NodePtrs<T>& nonleafs,
                  const Args& args, const Keys& leafkeys, bool direction=false) {
    //! Create a tree node
    node->idx = int(node-&nodes[0]);  // current node's index in nodes
    node->nsrcs = source_end - source_begin;
    node->ntrgs = target_end - target_begin;
#if COMPLEX
    node->up_equiv.resize(NSURF, complex_t(0.,0.));
    node->dn_equiv.resize(NSURF, complex_t(0.,0.));
#else
    node->up_equiv.resize(NSURF, 0.);
    node->dn_equiv.resize(NSURF, 0.);
#endif
    ivec3 iX = get3DIndex(node->x, node->level);
    node->key = getKey(iX, node->level);

    //! If node is a leaf
    bool is_leaf_key = 1;
    if (!leafkeys.empty()) {  // when leafkeys is given (when balancing tree) 
      std::set<uint64_t>::iterator it = leafkeys[node->level].find(node->key);
      if (it == leafkeys[node->level].end()) {  // if current key is not a leaf key
        is_leaf_key = 0;
      }
    }
    if (node->nsrcs<=args.ncrit && node->ntrgs<=args.ncrit && is_leaf_key) {
      node->is_leaf = true;
#if COMPLEX
      node->trg_value.resize(node->ntrgs*4, complex_t(0.,0.));   // initialize target result vector
#else
      node->trg_value.resize(node->ntrgs*4, 0.);   // initialize target result vector
#endif
      if (node->nsrcs || node->ntrgs)
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
      // Copy sources and targets' coords and values to leaf (only during 2:1 tree balancing)
      if (!leafkeys.empty()) {
        Body<T>* first_source = (direction ? sources_buffer : sources) + source_begin;
        Body<T>* first_target = (direction ? targets_buffer : targets) + target_begin;
        for (Body<T>* B=first_source; B<first_source+node->nsrcs; ++B) {
          for (int d=0; d<3; ++d) {
            node->src_coord.push_back(B->X[d]);
          }
#if SORT_BACK
          node->isrcs.push_back(B->ibody);
#endif
          node->src_value.push_back(B->q);
        }
        for (Body<T>* B=first_target; B<first_target+node->ntrgs; ++B) {
          for (int d=0; d<3; ++d) {
            node->trg_coord.push_back(B->X[d]);
          }
#if SORT_BACK
          node->itrgs.push_back(B->ibody);
#endif
        }
      }
      return;
    }
    // Sort bodies and save in buffer
    std::vector<int> source_size, source_offsets;
    std::vector<int> target_size, target_offsets;
    sort_bodies(node, sources, sources_buffer, source_begin, source_end, source_size, source_offsets);  // sources_buffer is sorted
    sort_bodies(node, targets, targets_buffer, target_begin, target_end, target_size, target_offsets);  // targets_buffer is sorted
    //! Loop over children and recurse
    node->is_leaf = false;
    nonleafs.push_back(node);
    assert(nodes.capacity() >= nodes.size()+NCHILD);
    nodes.resize(nodes.size()+NCHILD);
    Node<T> * child = &nodes.back() - NCHILD + 1;
    node->children.resize(8, nullptr);
    for (int c=0; c<8; c++) {
      node->children[c] = &child[c];
      child[c].x = node->x;
      child[c].r = node->r / 2;
      for (int d=0; d<3; d++) {
        child[c].x[d] += child[c].r * (((c & 1 << d) >> d) * 2 - 1);
      }
      child[c].parent = node;
      child[c].octant = c;
      child[c].level = node->level + 1;
      build_tree(sources_buffer, sources, source_offsets[c], source_offsets[c] + source_size[c],
                 targets_buffer, targets, target_offsets[c], target_offsets[c] + target_size[c],
                 &child[c], nodes,  leafs, nonleafs, args, leafkeys, !direction);
    }
  }

  /**
   * @brief Recursively build the tree and return the tree as a vector of nodes
   *
   * @param sources Vector of sources
   * @param targets Vector of targets
   * @param x0 Coordinates of the lower left bottom vertex of the bounding box
   * @param r0 Radius of the bounding box
   * @param leafs Vector of pointers of leaf nodes
   * @param nonleafs Vector of pointers of nonleaf nodes
   * @param args Args that contains tree information
   * @param leafkeys Vector of leaf Hilbert keys of each level, only used during 2:1 tree balancing 
   *
   * @return Vector of nodes that represents the tree
   */
  template <typename T>
  Nodes<T> build_tree(Bodies<T>& sources, Bodies<T>& targets, vec3 x0, real_t r0, NodePtrs<T>& leafs,
                      NodePtrs<T>& nonleafs, const Args& args, const Keys& leafkeys=Keys()) {
    Bodies<T> sources_buffer = sources;
    Bodies<T> targets_buffer = targets;
    Nodes<T> nodes(1);
    nodes[0].parent = nullptr;
    nodes[0].octant = 0;
    nodes[0].x = x0;
    nodes[0].r = r0;
    nodes[0].level = 0;
    nodes.reserve((sources.size()+targets.size()) * (32/args.ncrit+1));
    build_tree(&sources[0], &sources_buffer[0], 0, sources.size(), 
              &targets[0], &targets_buffer[0], 0, targets.size(),
              &nodes[0], nodes, leafs, nonleafs, args, leafkeys);
    return nodes;
  }

  /**
   * @brief Generate the set of Morton keys of nodes at each level using a breadth-first traversal
   * 
   * @param root Root node pointer
   * @param key2id A map from Morton key to node index
   * @return Vector of the set of Morton keys of nodes at each level (before balancing)
   */
  template <typename T>
  Keys breadth_first_traversal(Node<T>* const root, std::unordered_map<uint64_t, size_t>& key2id) {
    assert(root);
    Keys keys;
    std::queue<Node<T>*> buffer;
    buffer.push(root);
    int level = 0;
    std::set<uint64_t> keys_;
    while (!buffer.empty()) {
      Node<T>* curr = buffer.front();
      if (curr->level != level) {
        keys.push_back(keys_);
        keys_.clear();
        level = curr->level;
      }
      keys_.insert(curr->key);
      key2id[curr->key] = curr->idx;
      buffer.pop();
      if (!curr->is_leaf) {
        for (int i=0; i<NCHILD; i++) {
          buffer.push(curr->children[i]);
        }
      }
    }
    if (keys_.size())
      keys.push_back(keys_);
    return keys;
  }

  /**
   * @brief Generate the set of Morton keys of nodes at each level after 2:1 balancing
   * 
   * @param keys Vector of the set of Morton keys of nodes at each level (before balancing)
   * @param key2id A map from Morton key to node index
   * @param nodes Vector of nodes that represents the tree
   * @return Vector of the set of Morton keys of nodes at each level after 2:1 balancing
   */
  template <typename K>
  Keys balance_tree(const Keys& keys, const std::unordered_map<uint64_t, size_t>& key2id, const Nodes<K>& nodes) {
    int nlevels = keys.size();
    int maxlevel = nlevels - 1;
    Keys bkeys(keys.size());      // balanced Morton keys
    std::set<uint64_t> S, N;
    std::set<uint64_t>::iterator it;
    for (int l=maxlevel; l>0; --l) {
      // N <- S + nonleafs
      N.clear();
      for (it=keys[l].begin(); it!=keys[l].end(); ++it)
        if (!nodes[key2id.at(*it)].is_leaf) // choose nonleafs
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

  /**
   * @brief Find leaf keys at each level
   * 
   * @param keys Vector of the set of Morton keys of nodes at each level after 2:1 balancing
   * @return Vector of leaf keys at each level
   */
  Keys find_leaf_keys(const Keys& keys) {
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

  /**
   * @brief 
   * 
   * @param nodes Vector of nodes that represents the tree (after 2:1 balancing)
   * @param sources Vector of sources
   * @param targets Vector of targets
   * @param x0 Coordinates of the center of the root
   * @param r0 Radius of root node
   * @param leafs Vector of pointers of leaf nodes
   * @param nonleafs Vector of pointers of non-leaf nodes
   * @param args Args that contains tree information
   */
  template <typename T>
  void balance_tree(Nodes<T>& nodes, Bodies<T>& sources, Bodies<T>& targets, vec3 x0, real_t r0,
                    NodePtrs<T>& leafs, NodePtrs<T>& nonleafs, const Args& args) {
    std::unordered_map<uint64_t, size_t> key2id;
    Keys keys = breadth_first_traversal(&nodes[0], key2id);
    Keys balanced_keys = balance_tree(keys, key2id, nodes);
    Keys leaf_keys = find_leaf_keys(balanced_keys);
    nodes.clear();
    leafs.clear();
    nonleafs.clear();
    nodes = build_tree(sources, targets, x0, r0, leafs, nonleafs, args, leaf_keys);
    MAXLEVEL = keys.size() - 1;
  }
}
#endif
