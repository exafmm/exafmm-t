/* An example to solve a 3d Helmholtz N-body problem.*/

#include <random>
#include "build_tree.h"
#include "build_list.h"
#include "helmholtz.h"

int main() {

  using exafmm_t::start;
  using exafmm_t::stop;
  using exafmm_t::print;
  using exafmm_t::print_divider;
  using exafmm_t::complex_t;
  
  print_divider("Time");

  /* step 1: Generate 100,000 sources and targets that are randomly distributed
   *         in a cube from -1 to 1. The charge of each source is also from -1
   *         to 1.
   */
  std::random_device rd;
  std::mt19937 gen(rd());  // random number generator
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  int ntargets = 100000;
  int nsources = 100000;

  exafmm_t::Bodies<complex_t> sources(nsources);
  for (int i=0; i<nsources; i++) {
    sources[i].ibody = i;
    sources[i].q = complex_t(dist(gen), dist(gen));
    for (int d=0; d<3; d++)
      sources[i].X[d] = dist(gen);
  }

  exafmm_t::Bodies<complex_t> targets(ntargets);
  for (int i=0; i<ntargets; i++) {
    targets[i].ibody = i;
    for (int d=0; d<3; d++)
      targets[i].X[d] = dist(gen);
  }


  /* step 2: Create an Fmm instance for Laplace kernel.
   */
  int P = 8;         // expansion order
  int ncrit = 400;   // max number of bodies per leaf
  complex_t wavek(2, 4);   // wavenumber
  exafmm_t::HelmholtzFmm fmm(P, ncrit, wavek);

  /* step 3: Build and balance the octree.
   */
  start("Build Tree");
  exafmm_t::NodePtrs<complex_t> leafs, nonleafs;
  exafmm_t::Nodes<complex_t> nodes;

  exafmm_t::get_bounds(sources, targets, fmm.x0, fmm.r0);
  nodes = exafmm_t::build_tree(sources, targets, leafs, nonleafs, fmm);
  stop("Build Tree");

  /* step 4: Build lists and pre-compute invariant matrices.
   */
  start("Build Lists");
  exafmm_t::init_rel_coord();
  exafmm_t::build_list(nodes, fmm);
  fmm.M2L_setup(nonleafs);
  stop("Build Lists");

  start("Precomputation");
  fmm.precompute();
  stop("Precomputation");

  /* step 5: Use FMM to evaluate potential
   */
  start("Evaluation");
  fmm.upward_pass(nodes, leafs);
  fmm.downward_pass(nodes, leafs);
  stop("Evaluation");


  auto err = fmm.verify(leafs);
  print_divider("Error");

  print("Potential Error L2", err[0]);
  print("Gradient Error L2", err[1]);

  /* step 6: Optional.
   *         After evaluation, the computed potentials and gradients of targets
   *         are stored in trg_value of each leaf. During tree construction, the
   *         particles are sorted based on their Hibert indices. To sort these
   *         values back to the original order of targets, see the example below. 
   */
  std::vector<complex_t> potential(ntargets);
  std::vector<complex_t> gradient_x(ntargets);
  std::vector<complex_t> gradient_y(ntargets);
  std::vector<complex_t> gradient_z(ntargets);

  for (int i=0; i<leafs.size(); ++i) {
    exafmm_t::Node<complex_t>* leaf = leafs[i];
    std::vector<int>& itrgs = leaf->itrgs;
    for (int j=0; j<itrgs.size(); ++j) {
      potential[itrgs[j]]  = leaf->trg_value[4*j+0];
      gradient_x[itrgs[j]] = leaf->trg_value[4*j+1];
      gradient_y[itrgs[j]] = leaf->trg_value[4*j+2];
      gradient_z[itrgs[j]] = leaf->trg_value[4*j+3];
    }
  }

  print_divider("Tree");
  print("Root Center x", fmm.x0[0]);
  print("Root Center y", fmm.x0[1]);
  print("Root Center z", fmm.x0[2]);
  print("Root Radius R", fmm.r0);
  print("Tree Depth", fmm.depth);
  print("Leaf Nodes", leafs.size());

  return 0;
}
