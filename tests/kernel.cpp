#include <numeric>
#if HELMHOLTZ
#include "helmholtz.h"
#include "precompute_helmholtz.h"
#else
#include "laplace.h"
#include "precompute_laplace.h"
#endif
#include "exafmm_t.h"

namespace exafmm_t {
  int P;
  int NSURF;
  int MAXLEVEL;
  vec3 XMIN0;
  real_t R0;
#if HELMHOLTZ
  real_t MU;
#endif
}

using namespace exafmm_t;
using namespace std;

void set_children(Node* parent, Node* first_child) {
  for(int octant=0; octant<8; ++octant) {
    Node* child = first_child + octant;
    child->octant = octant;
    child->parent = parent;
    child->level = parent->level + 1;
    child->r = parent->r / 2;
    child->xmin = parent->xmin;
    for(int d=0; d<3; d++) {
      child->xmin[d] += parent->r * ((octant & 1 << d) >> d);
    }
    parent->children.push_back(child);
    parent->is_leaf = false;
  }
}

int main() {
  // set global variables
  P = 6;
  NSURF = 6*(P-1)*(P-1) + 2;
  MAXLEVEL = 3;
  XMIN0 = 0.;
  R0 = 4.;
#if HELMHOLTZ
  MU = 20;
#endif
  // precomputation
  init_rel_coord();
  precompute();
  // create a source and a target
  // create tree (nodes)
  vector<int> nnodes = {1, 8, 8*2, 8*2};  // number of nodes at each level
  Nodes nodes(accumulate(nnodes.begin(), nnodes.end(), 0));
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
  Node& root = nodes[0];
  root.parent = nullptr;
  root.octant = 0;
  root.xmin = XMIN0;
  root.r = R0;
  root.level = 0;
  // create descendants
  set_children(&root, &nodes[1]);      // lvl 1 nodes
  set_children(&nodes[1], &nodes[9]);  // lvl 2 nodes left corner
  set_children(&nodes[8], &nodes[17]);  // lvl 2 nodes upper corner
  set_children(&nodes[9], &nodes[25]);  // lvl 3 nodes
  set_children(&nodes[24], &nodes[33]); // lvl 3 nodes
  // add source and target
  nodes[25].src_coord.resize(3, 0.5);
  nodes[25].src_value.push_back(1.0);
  nodes[40].trg_coord.resize(3, 7.5);
  nodes[40].trg_value.resize(4, 0.);
  
  // P2M
  NodePtrs leafs;
  Node* source = &nodes[25];
  leafs.push_back(source);
  P2M(leafs);
  // M2M
  // M2M(source->parent);
#if 0
  for(int i=0; i<NSURF; ++i)
    cout << i << " " << source->parent->up_equiv[i] << endl;
#endif
  // set M2Llist
  Node* target = &nodes[40];
  target->parent->M2L_list.resize(REL_COORD[M2L_Type].size(), nullptr);
  target->parent->M2L_list[0] = source->parent;
  // M2L
  NodePtrs nonleafs;
  nonleafs.push_back(target->parent);
  M2L_setup(nonleafs);
  M2L(nodes);
  // L2L
  L2L(target->parent);  // this won't do anything
  // L2P
  leafs.clear();
  leafs.push_back(target);
  L2P(leafs);
  // calculate error
  for(int i=0; i<4; ++i)
    cout << target->trg_value[i] << " ";
  cout << endl;


  RealVec ans(4, 0.);
  gradient_P2P(source->src_coord, source->src_value, target->trg_coord, ans);
  for(int i=0; i<4; ++i)
    cout << ans[i] << " ";
  cout << endl;

#if 0
  RealVec trg_value_fmm = target->trg_value;
  fill(target->trg_value.begin(), target->trg_value.end(), 0.);
  gradient_P2P(source->src_coord, source->src_value, target->trg_coord, target->trg_value);
  RealVec trg_value_p2p = target->trg_value;

  real_t p_diff = (trg_value_fmm[0] - trg_value_p2p[0]) * (trg_value_fmm[0] - trg_value_p2p[0]);
  real_t p_norm = trg_value_p2p[0] * trg_value_p2p[0];
  real_t g_diff = 0, g_norm = 0;
  for(int d=1; d<4; ++d) {
    g_diff += (trg_value_fmm[d] - trg_value_p2p[d]) * (trg_value_fmm[d] - trg_value_p2p[d]);
    g_norm += trg_value_p2p[d] * trg_value_p2p[d];
  }
  cout << sqrt(p_diff/p_norm) << endl;
#endif
  return 0;
}
