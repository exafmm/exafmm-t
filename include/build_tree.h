#ifndef build_tree_h
#define build_tree_h
#include <cassert>
#include <queue>
#include <unordered_map>
#include "exafmm_t.h"
#include "hilbert.h"
#include "fmm_base.h"

namespace exafmm_t {
  /**
   * @brief Get bounding box of sources and targets.
   *
   * @tparam T Target's value type (real or complex).
   * @param sources Vector of sources.
   * @param targets Vector of targets.
   * @param x0 Coordinates of the center of the bounding box.
   * @param r0 Radius of the bounding box.
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
    r0 = fmax(max(x0-Xmin), max(Xmax-x0));
    r0 *= 1.00001;
  } 

  /**
   * @brief Sort a chunk of bodies in a node according to their octants
   *
   * @tparam T Target's value type (real or complex)
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
      buffer[counter[octant]].ibody = bodies[i].ibody;
      counter[octant]++;
    }
  }

  //! Build nodes of tree adaptively using a top-down approach based on recursion
  template <typename T>
  void build_tree(Body<T>* sources, Body<T>* sources_buffer, int source_begin, int source_end, 
                  Body<T>* targets, Body<T>* targets_buffer, int target_begin, int target_end,
                  Node<T>* node, Nodes<T>& nodes, NodePtrs<T>& leafs, NodePtrs<T>& nonleafs,
                  FmmBase<T>& fmm, bool direction=false) {
    //! Create a tree node
    node->idx = int(node-&nodes[0]);  // current node's index in nodes
    node->nsrcs = source_end - source_begin;
    node->ntrgs = target_end - target_begin;
    node->up_equiv.resize(fmm.nsurf, (T)(0.));
    node->dn_equiv.resize(fmm.nsurf, (T)(0.));
    ivec3 iX = get3DIndex(node->x, node->level, fmm.x0, fmm.r0);
    node->key = getKey(iX, node->level);

    //! If node is a leaf
    if (node->nsrcs<=fmm.ncrit && node->ntrgs<=fmm.ncrit) {
      node->is_leaf = true;
      node->trg_value.resize(node->ntrgs*4, (T)(0.));   // initialize target result vector
      if (node->nsrcs || node->ntrgs)
        leafs.push_back(node);
      if (direction) {
        for (int i=source_begin; i<source_end; i++) {
          sources_buffer[i].X = sources[i].X;
          sources_buffer[i].q = sources[i].q;
          sources_buffer[i].ibody = sources[i].ibody;
        }
        for (int i=target_begin; i<target_end; i++) {
          targets_buffer[i].X = targets[i].X;
          targets_buffer[i].ibody = targets[i].ibody;
        }
      }
      // Copy sources and targets' coords and values to leaf
      Body<T>* first_source = (direction ? sources_buffer : sources) + source_begin;
      Body<T>* first_target = (direction ? targets_buffer : targets) + target_begin;
      for (Body<T>* B=first_source; B<first_source+node->nsrcs; ++B) {
        for (int d=0; d<3; ++d) {
          node->src_coord.push_back(B->X[d]);
        }
        node->isrcs.push_back(B->ibody);
        node->src_value.push_back(B->q);
      }
      for (Body<T>* B=first_target; B<first_target+node->ntrgs; ++B) {
        for (int d=0; d<3; ++d) {
          node->trg_coord.push_back(B->X[d]);
        }
        node->itrgs.push_back(B->ibody);
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
                 &child[c], nodes, leafs, nonleafs, fmm, !direction);
    }
  }

  /**
   * @brief Recursively build the tree and return the tree as a vector of nodes
   *
   * @tparam T Target's value type (real or complex)
   * @param sources Vector of sources
   * @param targets Vector of targets
   * @param leafs Vector of pointers of leaf nodes
   * @param nonleafs Vector of pointers of nonleaf nodes
   * @param fmm FMM instance
   * @param leafkeys Vector of leaf Hilbert keys of each level, only used during 2:1 tree balancing 
   *
   * @return Vector of nodes that represents the tree
   */
  template <typename T>
  Nodes<T> build_tree(Bodies<T>& sources, Bodies<T>& targets,
                      NodePtrs<T>& leafs, NodePtrs<T>& nonleafs,
                      FmmBase<T>& fmm, const Keys& leafkeys=Keys()) {
    Bodies<T> sources_buffer = sources;
    Bodies<T> targets_buffer = targets;
    Nodes<T> nodes(1);
    nodes[0].parent = nullptr;
    nodes[0].octant = 0;
    nodes[0].x = fmm.x0;
    nodes[0].r = fmm.r0;
    nodes[0].level = 0;
    nodes.reserve((sources.size()+targets.size()) * (32/fmm.ncrit+1));
    build_tree(&sources[0], &sources_buffer[0], 0, sources.size(), 
               &targets[0], &targets_buffer[0], 0, targets.size(),
               &nodes[0], nodes, leafs, nonleafs, fmm);
    return nodes;
  }
}
#endif
