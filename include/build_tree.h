#ifndef buildtree_h
#define buildtree_h
#include <cassert>
#include "pvfmm.h"
namespace pvfmm {
  //! Build cells of tree adaptively using a top-down approach based on recursion
  void buildCells(Body * bodies, Body * buffer, int begin, int end, FMM_Node * cell, FMM_Nodes & cells,
                  const vec3 & X, real_t R, int level=0, bool direction=false) {
    cell->depth = level;         // depth
    //! Create a tree cell
    cell->body = bodies + begin;
    if(direction) cell->body = buffer + begin;
    cell->numBodies = end - begin;
    cell->numChilds = 0;
    cell->X = X;
    cell->R = R;
    //cell->M.resize(NTERM, 0.0);
    //cell->L.resize(NTERM, 0.0);
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
      if (size[i]) cell->numChilds++;
    }
    //! If cell is a leaf
    if (end - begin <= NCRIT) {
      cell->numChilds = 0;
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
    assert(cells.capacity() >= cells.size()+cell->numChilds);
    cells.resize(cells.size()+cell->numChilds);
    FMM_Node * child = &cells.back() - cell->numChilds + 1;
    cell->fchild = child;
    cell->child.resize(8, NULL);
    for (int i=0, c=0; i<8; i++) {
      Xchild = X;
      real_t Rchild = R / 2;
      for (int d=0; d<3; d++) {
        Xchild[d] += Rchild * (((i & 1 << d) >> d) * 2 - 1);
      }
      if (size[i]) {
        child[c].parent = cell;
        child[c].octant = i;
        cell->child[i] = &child[c];

        buildCells(buffer, bodies, offsets[i], offsets[i] + size[i],
                   &child[c++], cells, Xchild, Rchild, level+1, !direction);
      }
    }
  }

  FMM_Nodes buildTree(Bodies & bodies) {
    real_t R0 = 0.5;
    vec3 X0(0.5);
    Bodies buffer = bodies;
    FMM_Nodes cells(1);
    cells[0].parent = NULL;
    cells[0].octant = 0;
    cells.reserve(bodies.size()*(32/NCRIT+1));
    buildCells(&bodies[0], &buffer[0], 0, bodies.size(), &cells[0], cells, X0, R0);
    return cells;
  }
}
#endif
