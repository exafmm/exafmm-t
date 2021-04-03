#ifndef build_list_h
#define build_list_h
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <queue>
#include "exafmm_t.h"
#include "geometry.h"
#include "hilbert.h"
#include "fmm_base.h"

namespace exafmm_t {
  using std::abs;
  using std::max;
  using std::unordered_map;
  using std::unordered_set;
  using std::set;
  using std::queue;

  /**
   * @brief Generate the mapping from Hilbert keys to node indices in the tree.
   *
   * @param nodes Tree.
   * @return Keys to indices mapping.
   */
  template <typename T>
  unordered_map<uint64_t, size_t> get_key2id(const Nodes<T>& nodes) {
    unordered_map<uint64_t, size_t> key2id;
    for (int i=0; i<nodes.size(); ++i) {
      key2id[nodes[i].key] = nodes[i].idx;
    }
    return key2id;
  }

  /**
   * @brief Generate the set of keys of all leaf nodes.
   *
   * @param nodes Tree.
   * @return Set of all leaf keys with level offset.
   */
  template <typename T>
  unordered_set<uint64_t> get_leaf_keys(const Nodes<T>& nodes) {
    // we cannot use leafs to generate leaf keys, since it does not include
    // empty leaf nodes where ntrgs and nsrcs are 0.
    unordered_set<uint64_t> leaf_keys;
    for (int i=0; i<nodes.size(); ++i) {
      if (nodes[i].is_leaf) {
        leaf_keys.insert(nodes[i].key);
      }
    }
    return leaf_keys;
  }

  /**
   * @brief Given the integer index of an octant and its depth, return the Hilbert
   * index of the leaf that contains the octant.
   *
   * @param iX Integer index of the octant.
   * @param level The level of the octant.
   *
   * @return Hilbert index with level offset.
   */
	uint64_t findgnt(const ivec3& iX, int level, const unordered_set<uint64_t>& leaf_keys) {
    uint64_t orig_key = getKey(iX, level, true);
    uint64_t curr_key = orig_key;
		while (level>0) {
      if (leaf_keys.find(curr_key) != leaf_keys.end()) {  // if key is leaf
        return curr_key;
      } else {     // else go 1 level up
        curr_key = getParent(curr_key);
        level--;
      }
		}
		return orig_key;
	}

  /**
   * @brief Check the adjacency of two nodes.
   *
   * @param key_a, key_b Hilbert keys with level offset.
   */
  bool is_adjacent(uint64_t key_a, uint64_t key_b) {
    int level_a = getLevel(key_a);
    int level_b = getLevel(key_b);
    int max_level = max(level_a, level_b);
    ivec3 iX_a = get3DIndex(key_a); 
    ivec3 iX_b = get3DIndex(key_b); 
    ivec3 iX_ac = (iX_a*2 + 1) * (1 << (max_level-level_a));  // center coordinates
    ivec3 iX_bc = (iX_b*2 + 1) * (1 << (max_level-level_b));  // center coordinates
    ivec3 diff = iX_ac - iX_bc;
    int max_diff = -1;   // L-infinity norm of diff
    for (int d=0; d<3; ++d) {
      diff[d] = abs(diff[d]);
      max_diff = max(max_diff, diff[d]);
    }
    int sum_radius = (1 << (max_level-level_a)) + (1 << (max_level-level_b));

    return (diff[0] <= sum_radius) &&
           (diff[1] <= sum_radius) &&
           (diff[2] <= sum_radius) &&
           (max_diff == sum_radius);
  }

  template <typename T>
  void build_list(Node<T>* node, const unordered_set<uint64_t>& leaf_keys,
                  const unordered_map<uint64_t, size_t>& key2id, Nodes<T>& nodes) {
    set<Node<T>*> Uset, Wset, Xset;
    Node<T>* curr = node;
    if (curr->key != 0) {
      Node<T>* parent = curr->parent;
      ivec3 minIdx = 0;
      ivec3 maxIdx = 1 << node->level;
      ivec3 curr_iX = get3DIndex(curr->key);
      ivec3 parent_iX = get3DIndex(parent->key); 
      // search in every direction
      for (int i=-2; i<4; i++) {
        for (int j=-2; j<4; j++) {
          for (int k=-2; k<4; k++) {
            // Index3 tryPath( 2*path2Node(parGNodeIdx) + Index3(i,j,k) );
            ivec3 tryPath;
            tryPath[0] = i;
            tryPath[1] = j;
            tryPath[2] = k;
            tryPath += parent_iX * 2;
            if (tryPath >= minIdx && tryPath < maxIdx && tryPath != curr_iX) {	
              // int resGNodeIdx = findgnt(depth(curGNodeIdx), tryPath);
              uint64_t res_key = findgnt(tryPath, curr->level, leaf_keys);
              bool adj = is_adjacent(res_key, curr->key);
              Node<T>* res = &nodes[key2id.at(res_key)];
              if (res->level < curr->level) {
                if (adj) {
                  if (curr->is_leaf) {
                    Uset.insert(res);
                  } else {
                    ;
                  }
                } else {
                  Xset.insert(res);
                }
              }
              if (res->level == curr->level) {
                if (!adj) {
                  ;
                } else {
                  if (curr->is_leaf) {
                    queue<Node<T>*> rest;
                    rest.push(res);
                    while (!rest.empty()) {
                      auto fnt = rest.front(); rest.pop();
                      if (!is_adjacent(fnt->key, curr->key)) {
                        Wset.insert(fnt);
                      } else {
                        if (fnt->is_leaf) {
                          Uset.insert(fnt);
                        } else { 
                          for (int i=0; i<8; i++) {
                            auto child = fnt->children[i];
                            if (child != nullptr) {
                              rest.push(child);
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    if(curr->is_leaf)
      Uset.insert(curr);
    for(typename set<Node<T>*>::iterator i=Uset.begin(); i!=Uset.end(); i++) {
      if ((*i) != nullptr) {
        curr->P2P_list.push_back(*i);
      }
    }
    for(typename set<Node<T>*>::iterator i=Xset.begin(); i!=Xset.end(); i++) {
      if ((*i) != nullptr) {
        curr->P2L_list.push_back(*i);
      }
    }
    for(typename set<Node<T>*>::iterator i=Wset.begin(); i!=Wset.end(); i++) {
      if ((*i) != nullptr) {
        curr->M2P_list.push_back(*i);
      }
    }
  }
  

  template <typename T>
  void build_V(Node<T>* node, Nodes<T>& nodes, const unordered_map<uint64_t, size_t>& key2id) {
    node->M2L_list.resize(REL_COORD[M2L_Type].size(), nullptr);
    Node<T>* curr = node;
    ivec3 min_iX = 0;
    ivec3 max_iX = 1 << curr->level;
    if (!node->is_leaf) {
      ivec3 curr_iX = get3DIndex(curr->key);
      ivec3 col_iX;
      ivec3 rel_coord;
      for (int i=-1; i<=1; i++) {
        rel_coord[0] = i;
        for (int j=-1; j<=1; j++) {
          rel_coord[1] = j;
          for (int k=-1; k<=1; k++) {
            rel_coord[2] = k;
            if (i || j || k) {  // exclude current node itself
              col_iX = curr_iX + rel_coord;
              if (col_iX >= min_iX && col_iX < max_iX) {
                uint64_t col_key = getKey(col_iX, curr->level, true); 
                if (key2id.find(col_key) != key2id.end()) {
                  Node<T>* col = &nodes[key2id.at(col_key)];
                  if (!col->is_leaf) {
                    int c_hash = hash(rel_coord);
                    int idx = HASH_LUT[M2L_Type][c_hash];
                    curr->M2L_list[idx] = col;
                  }
                }
              } 
            }
          }
        }
      }
    }
  }

  template <typename T>
  void build_list(Nodes<T>& nodes, const FmmBase<T>& fmm) {
    unordered_map<uint64_t, size_t> key2id = get_key2id(nodes);
    unordered_set<uint64_t> leaf_keys = get_leaf_keys(nodes);

    for(size_t i=0; i<nodes.size(); i++) {
      Node<T>* node = &nodes[i];
      build_V(node, nodes, key2id);
      build_list(node, leaf_keys, key2id, nodes);
    }
  }
}
#endif
