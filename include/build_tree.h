#ifndef buildtree_h
#define buildtree_h
#include "exafmm_t.h"

namespace exafmm_t {
  //! Build nodes of tree adaptively using a top-down approach based on recursion
  void buildTree(Body * bodies, Body * buffer, int begin, int end, Node * node, Nodes & nodes,
                  const vec3 & X, real_t R, int level=0, bool direction=false) {
    node->depth = level;         // depth
    //! Create a tree node
    node->body = bodies + begin;
    if(direction) node->body = buffer + begin;
    node->numBodies = end - begin;
    node->numChilds = 0;
    node->X = X;
    node->R = R;
    node->upward_equiv.resize(NSURF, 0.0);
    node->dnward_equiv.resize(NSURF, 0.0);
    //! Count number of bodies in each octant
    int size[8] = {0,0,0,0,0,0,0,0};
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
    if (end - begin <= NCRIT) {
      node->numChilds = 0;
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

        buildTree(buffer, bodies, offsets[i], offsets[i] + size[i],
                   &child[c++], nodes, Xchild, Rchild, level+1, !direction);
      }
    }
  }

  Nodes buildTree(Bodies & bodies) {
    real_t R0 = 0.5;
    vec3 X0(0.5);
    Bodies buffer = bodies;
    Nodes nodes(1);
    nodes[0].parent = NULL;
    nodes[0].octant = 0;
    nodes.reserve(bodies.size()*(32/NCRIT+1));
    buildTree(&bodies[0], &buffer[0], 0, bodies.size(), &nodes[0], nodes, X0, R0);
    return nodes;
  }
}
#endif
