#ifndef buildtree_h
#define buildtree_h
#include <unordered_map>
#include <queue>
#include "exafmm_t.h"
#include "hilbert.h"

using namespace std;
namespace exafmm_t {
  vec3 XMIN0;
  real_t R0;

  // Get bounding box of sources and targets
  void get_bounds(Bodies& sources, Bodies& targets, vec3& Xmin0, real_t& r0) {
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
    vec3 X0 = (Xmax + Xmin) / 2;
    r0 = fmax(max(X0-Xmin), max(Xmax-X0));
    r0 *= 1.00001;
    Xmin0 = X0 - r0;
  } 

  // Sort bodies in a node according to their octants
  void sort_bodies(Node * node, Body * bodies, Body * buffer, int begin, int end, std::vector<int>& size, std::vector<int>& offsets) {
    // Count number of bodies in each octant
    size.resize(8, 0);
    vec3 X = node->xmin + node->r;  // the center of the node
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
  void build_tree(Body * sources, Body * sources_buffer, int source_begin, int source_end, 
                 Body * targets, Body * targets_buffer, int target_begin, int target_end,
                 Node * node, Nodes & nodes, NodePtrs & leafs, NodePtrs & nonleafs,
                 Args & args, const Keys & leafkeys, bool direction=false) {
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
    ivec3 iX = get3DIndex(node->xmin+node->r, node->level);
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
        Body* first_source = (direction ? sources_buffer : sources) + source_begin;
        Body* first_target = (direction ? targets_buffer : targets) + target_begin;
        for (Body* B=first_source; B<first_source+node->nsrcs; ++B) {
          for (int d=0; d<3; ++d) {
            node->src_coord.push_back(B->X[d]);
          }
          node->src_value.push_back(B->q);
        }
        for (Body* B=first_target; B<first_target+node->ntrgs; ++B) {
          for (int d=0; d<3; ++d) {
            node->trg_coord.push_back(B->X[d]);
          }
        }
      }
      return;
    }
    // Sort bodies and save in buffer
    std::vector<int> source_size, source_offsets;
    std::vector<int> target_size, target_offsets;
    sort_bodies(node, sources, sources_buffer, source_begin, source_end, source_size, source_offsets);
    sort_bodies(node, targets, targets_buffer, target_begin, target_end, target_size, target_offsets);
    //! Loop over children and recurse
    node->is_leaf = false;
    nonleafs.push_back(node);
    assert(nodes.capacity() >= nodes.size()+NCHILD);
    nodes.resize(nodes.size()+NCHILD);
    Node * child = &nodes.back() - NCHILD + 1;
    node->children.resize(8, nullptr);
    for (int c=0; c<8; c++) {
      node->children[c] = &child[c];
      child[c].xmin = node->xmin;
      for (int d=0; d<3; d++) {
        child[c].xmin[d] += node->r * ((c & 1 << d) >> d);
      }
      child[c].r = node->r / 2;
      child[c].parent = node;
      child[c].octant = c;
      child[c].level = node->level + 1;
      build_tree(sources_buffer, sources, source_offsets[c], source_offsets[c] + source_size[c],
                targets_buffer, targets, target_offsets[c], target_offsets[c] + target_size[c],
                &child[c], nodes,  leafs, nonleafs, args, leafkeys, !direction);
    }
  }

  Nodes build_tree(Bodies & sources, Bodies & targets, vec3 Xmin0, real_t r0, NodePtrs & leafs, NodePtrs & nonleafs, Args & args, const Keys & leafkeys=Keys()) {
    Bodies sources_buffer = sources;
    Bodies targets_buffer = targets;
    Nodes nodes(1);
    nodes[0].parent = nullptr;
    nodes[0].octant = 0;
    nodes[0].xmin = Xmin0;
    nodes[0].r = r0;
    nodes[0].level = 0;
    nodes.reserve((sources.size()+targets.size()) * (32/args.ncrit+1));
    build_tree(&sources[0], &sources_buffer[0], 0, sources.size(), 
              &targets[0], &targets_buffer[0], 0, targets.size(),
              &nodes[0], nodes, leafs, nonleafs, args, leafkeys);
    return nodes;
  }

  // Given root, generate a level-order Morton keys
  Keys breadth_first_traversal(Node* root, std::unordered_map<uint64_t, size_t>& key2id) {
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

  Keys balance_tree(Keys& keys, std::unordered_map<uint64_t, size_t>& key2id, Nodes& nodes) {
    int nlevels = keys.size();
    int maxlevel = nlevels - 1;
    Keys bkeys(keys.size());      // balanced Morton keys
    std::set<uint64_t> S, N;
    std::set<uint64_t>::iterator it;
    for (int l=maxlevel; l>0; --l) {
      // N <- S + nonleafs
      N.clear();
      for (it=keys[l].begin(); it!=keys[l].end(); ++it)
        if (!nodes[key2id[*it]].is_leaf) // choose nonleafs
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

  Keys find_leaf_keys(Keys& keys) {
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

  void balance_tree(Nodes& nodes, Bodies& sources, Bodies& targets, vec3 Xmin0, real_t r0, NodePtrs& leafs, NodePtrs& nonleafs, Args& args) {
    std::unordered_map<uint64_t, size_t> key2id;
    Keys keys = breadth_first_traversal(&nodes[0], key2id);
    Keys balanced_keys = balance_tree(keys, key2id, nodes);
    Keys leaf_keys = find_leaf_keys(balanced_keys);
    nodes.clear();
    leafs.clear();
    nonleafs.clear();
    nodes = build_tree(sources, targets, Xmin0, r0, leafs, nonleafs, args, leaf_keys);
    MAXLEVEL = keys.size() - 1;
  }
}
#endif
