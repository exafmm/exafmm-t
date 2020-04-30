#include <numeric>    // std::accumulate
#include "exafmm_t.h"
#include "helmholtz.h"
#include "test.h"     // exafmm_t::set_children
#include "timer.h"

using namespace exafmm_t;

int main(int argc, char **argv) {
  Args args(argc, argv);
#if HAVE_OPENMP
  omp_set_num_threads(args.threads);
#endif

  // create fmm instance
  args.k = 0.1;
  HelmholtzFmm fmm(args.P, args.ncrit, args.maxlevel, args.k, "helmholtz_kernel_test.dat");
  fmm.depth = 3;
  fmm.x0 = 4.;
  fmm.r0 = 4.;

  // precomputation
  init_rel_coord();
  fmm.precompute();

  // create tree
  std::vector<int> nnodes = {1, 8, 8*2, 8*2};  // number of nodes at each level
  Nodes<complex_t> nodes(std::accumulate(nnodes.begin(), nnodes.end(), 0));

  // initialize nodes
  for (size_t i=0; i<nodes.size(); ++i) {
    Node<complex_t>& node = nodes[i];
    node.is_leaf = true;
    node.idx = i;
    node.up_equiv.resize(fmm.nsurf, 0.);
    node.dn_equiv.resize(fmm.nsurf, 0.);
  }

  // set root node
  Node<complex_t>* root = &nodes[0];
  root->parent = nullptr;
  root->x = fmm.x0;
  root->r = fmm.r0;
  root->level = 0;

  // create descendants
  set_children(root, &nodes[1]);      // lvl 1 nodes
  set_children(&nodes[1], &nodes[9]);  // lvl 2 nodes left corner
  set_children(&nodes[8], &nodes[17]);  // lvl 2 nodes upper corner
  set_children(&nodes[9], &nodes[25]);  // lvl 3 nodes
  set_children(&nodes[24], &nodes[33]); // lvl 3 nodes

  // add source and target
  Node<complex_t>* source = &nodes[25];   // lvl 3 source node
  Node<complex_t>* target = &nodes[40];   // lvl 3 target node
  source->src_coord.resize(3, 0.5);
  source->src_value.push_back(1.0);
  target->trg_coord.resize(3, 7.5);
  target->trg_value.resize(4, 0.);
  
#if DEBUG
  std::cout << "index level is_leaf nsrcs ntrgs" << std::endl;
  for (auto& node : nodes) {
    std::cout << node.idx << " " << node.level << " " << node.is_leaf << " "
              << node.src_coord.size()/3 << " " << node.trg_coord.size()/3 << std::endl;
  }
#endif

  // P2M
  NodePtrs<complex_t> leafs;
  leafs.push_back(source);
  fmm.P2M(leafs);
#if DEBUG
  std::cout << "lvl 3 source node's upward equivalent charges" << std::endl;
  for (int i=0; i<fmm.nsurf; ++i)
    std::cout << i << " " << source->up_equiv[i] << std::endl;
#endif

  // M2M
  fmm.M2M(root);
#if DEBUG
  std::cout << "lvl 2 source node's upward equivalent charges" << std::endl;
  for (int i=0; i<fmm.nsurf; ++i) {
    std::cout << i << " " << source->parent->up_equiv[i] << std::endl;
  }
#endif

  // set up M2L_list
  target->parent->parent->M2L_list.resize(REL_COORD[M2L_Type].size(), nullptr);
  target->parent->parent->M2L_list[0] = source->parent->parent;

  // M2L
  NodePtrs<complex_t> nonleafs;
  nonleafs.push_back(target->parent->parent);
  fmm.M2L_setup(nonleafs);
  fmm.M2L(nodes);

  // L2L
  fmm.L2L(root);

  // L2P
  leafs.clear();
  leafs.push_back(target);
  fmm.L2P(leafs);

  // direct summation
  ComplexVec trg_value_direct(4, 0.);
  fmm.gradient_P2P(source->src_coord, source->src_value, target->trg_coord, trg_value_direct);

  // calculate error
  ComplexVec& trg_value = target->trg_value;
  real_t p_diff = 0, p_norm = 0, p_error = 0;
  p_diff = std::norm(trg_value[0]-trg_value_direct[0]);
  p_norm = std::norm(trg_value_direct[0]);
  p_error = sqrt(p_diff/p_norm);
  real_t F_diff = 0, F_norm = 0, F_error = 0;
  for (int d=1; d<4; ++d) {
    F_diff += std::norm(trg_value[d]-trg_value_direct[d]);
    F_norm += std::norm(trg_value_direct[d]);
  }
  F_error = sqrt(F_diff/F_norm);
  print("Potential Error", p_error);
  print("Gradient Error", F_error);

  return 0;
}
