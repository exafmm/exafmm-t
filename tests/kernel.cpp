#include <numeric>
#if HELMHOLTZ
#include "helmholtz.h"
#include "precompute_helmholtz.h"
#else
#include "laplace.h"
#include "precompute_laplace.h"
#endif
#include "exafmm_t.h"
#include "print.h"

namespace exafmm_t {
  int P;
  int NSURF;
  int MAXLEVEL;
  vec3 X0;
  real_t R0;
#if HELMHOLTZ
  real_t WAVEK;
#endif
}

using namespace exafmm_t;

// Helper function to build the tree needed in kernel test
void set_children(Node* parent, Node* first_child) {
  parent->is_leaf = false;
  for(int octant=0; octant<8; ++octant) {
    Node* child = first_child + octant;
    child->octant = octant;
    child->parent = parent;
    child->level = parent->level + 1;
    child->x = parent->x;
    child->r = parent->r / 2;
    for(int d=0; d<3; d++) {
      child->x[d] += child->r * (((octant & 1 << d) >> d) * 2 - 1);
    }
    parent->children.push_back(child);
  }
}

int main() {
  // set global variables
  P = 8;
  NSURF = 6*(P-1)*(P-1) + 2;
  MAXLEVEL = 3;
  X0 = 4.;
  R0 = 4.;
#if HELMHOLTZ
  WAVEK = 10;
#endif
  // precomputation
  init_rel_coord();
  precompute();

  // create tree
  std::vector<int> nnodes = {1, 8, 8*2, 8*2};  // number of nodes at each level
  Nodes nodes(std::accumulate(nnodes.begin(), nnodes.end(), 0));
  // initialize nodes
  for(size_t i=0; i<nodes.size(); ++i) {
    Node& node = nodes[i];
    node.is_leaf = true;
    node.idx = i;
#if COMPLEX
    node.up_equiv.resize(NSURF, complex_t(0.,0.));
    node.dn_equiv.resize(NSURF, complex_t(0.,0.));
#else
    node.up_equiv.resize(NSURF, 0.);
    node.dn_equiv.resize(NSURF, 0.);
#endif
  }
  // set root node
  Node* root = &nodes[0];
  root->parent = nullptr;
  root->x = X0;
  root->r = R0;
  root->level = 0;

  // create descendants
  set_children(root, &nodes[1]);      // lvl 1 nodes
  set_children(&nodes[1], &nodes[9]);  // lvl 2 nodes left corner
  set_children(&nodes[8], &nodes[17]);  // lvl 2 nodes upper corner
  set_children(&nodes[9], &nodes[25]);  // lvl 3 nodes
  set_children(&nodes[24], &nodes[33]); // lvl 3 nodes

  // add source and target
  Node* source = &nodes[25];   // lvl 3 source node
  Node* target = &nodes[40];   // lvl 3 target node
  source->src_coord.resize(3, 0.5);
  source->src_value.push_back(1.0);
  target->trg_coord.resize(3, 7.5);
  target->trg_value.resize(4, 0.);
  
#if DEBUG
  cout << "index level is_leaf nsrcs ntrgs" << endl;
  for(auto& node : nodes) {
    cout << node.idx << " " << node.level << " " << node.is_leaf << " "
         << node.src_coord.size()/3 << " " << node.trg_coord.size()/3 <<endl;
  }
  return 0;
#endif

  // P2M
  NodePtrs leafs;
  leafs.push_back(source);
  P2M(leafs);

  // M2M
  M2M(root);
#if DEBUG
  cout << "lvl 2 source node's upward equivalent charges" << endl;
  for(int i=0; i<NSURF; ++i) {
    cout << i << " " << source->parent->up_equiv[i] << endl;
  }
#endif

  // set up M2L_list
  target->parent->parent->M2L_list.resize(REL_COORD[M2L_Type].size(), nullptr);
  target->parent->parent->M2L_list[0] = source->parent->parent;
  // M2L
  NodePtrs nonleafs;
  nonleafs.push_back(target->parent->parent);
  M2L_setup(nonleafs);
  M2L(nodes);

  // L2L
  L2L(root);

  // L2P
  leafs.clear();
  leafs.push_back(target);
  L2P(leafs);

  // direct summation
#if HELMHOLTZ
  ComplexVec trg_value_direct(4, 0.);
#else
  RealVec trg_value_direct(4, 0.);
#endif
  gradient_P2P(source->src_coord, source->src_value, target->trg_coord, trg_value_direct);

  // calculate error
#if HELMHOLTZ
  ComplexVec& trg_value = target->trg_value;
#else
  RealVec& trg_value = target->trg_value;
#endif
  real_t p_diff = 0, p_norm = 0, p_error = 0;
  p_diff = std::norm(trg_value[0]-trg_value_direct[0]);
  p_norm = std::norm(trg_value_direct[0]);
  p_error = sqrt(p_diff/p_norm);
  real_t F_diff = 0, F_norm = 0, F_error = 0;
  for(int d=1; d<4; ++d) {
    F_diff += std::norm(trg_value[d]-trg_value_direct[d]);
    F_norm += std::norm(trg_value_direct[d]);
  }
  F_error = sqrt(F_diff/F_norm);
  print("Potential Error", p_error);
  print("Gradient Error", F_error);

  return 0;
}
