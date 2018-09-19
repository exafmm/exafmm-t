#include "build_tree.h"
#include "dataset.h"
#include "interaction_list.h"
#include "laplace_c.h"
#include "precompute_c.h"
#include "profile.h"

using namespace exafmm_t;

int main(int argc, char **argv) {
  Args args(argc, argv);
  omp_set_num_threads(args.threads);
  size_t N = args.numBodies;
  MULTIPOLE_ORDER = args.P;
  NSURF = 6*(MULTIPOLE_ORDER-1)*(MULTIPOLE_ORDER-1) + 2;
  Profile::Enable(true); 

  Bodies bodies = initBodies(args.numBodies, args.distribution, 0);
  std::vector<Node*> leafs, nonleafs;
  Nodes nodes = buildTree(bodies, leafs, nonleafs, args);

  // balanced tree
  std::unordered_map<uint64_t, size_t> key2id;
  Keys keys = breadthFirstTraversal(&nodes[0], key2id);
  Keys bkeys = balanceTree(keys, key2id, nodes);
  Keys leafkeys = findLeafKeys(bkeys);
  nodes.clear();
  leafs.clear();
  nonleafs.clear();
  nodes = buildTree(bodies, leafs, nonleafs, args, leafkeys);  // rebuild 2:1 balanced tree
  MAXLEVEL = keys.size() - 1;

  // fill in pt_coord, pt_src, correct coord for compatibility
  // remove this later
  for(int i=0; i<nodes.size(); i++) {
    for(int d=0; d<3; d++) {
      nodes[i].coord[d] = nodes[i].X[d] - nodes[i].R;
    }
    if(nodes[i].IsLeaf()) {
      for(Body* B=nodes[i].body; B<nodes[i].body+nodes[i].numBodies; B++) {
        nodes[i].pt_coord.push_back(B->X[0]);
        nodes[i].pt_coord.push_back(B->X[1]);
        nodes[i].pt_coord.push_back(B->X[2]);
        nodes[i].pt_src.push_back(B->q);
      }
    }
  }

#if TEST_P2P
  int n = 20;
  RealVec src_coord(3*n), trg_coord(3*n);
  ComplexVec src_value(n), trg_value(n, complex_t(0.,0.));
  ComplexVec trg_F(4*n, complex_t(0.,0.));
  srand48(10);

  for(int i=0; i<n; ++i) {
    for(int d=0; d<3; ++d) {
      src_coord[3*i+d] = drand48();
      trg_coord[3*i+d] = drand48();
    } 
    src_value[i] = complex_t(drand48(), drand48());
  }

  //potentialP2P(src_coord, src_value, trg_coord, trg_value);
  gradientP2P(src_coord, src_value, trg_coord, trg_F);
  for(int i=0; i<n; ++i) {
    //std::cout << trg_value[i] << std::endl;
    for(int d=0; d<4; ++d) std::cout << trg_F[4*i+d] << " ";
    std::cout << std::endl;
  }
#endif

  initRelCoord();    // initialize relative coords
  Precompute();
  setColleagues(nodes);
  buildList(nodes);

  P2M(leafs);
  M2M(&nodes[0]);
  P2L(nodes);
  M2P(leafs);
  P2P(leafs);

#if TEST_PRECOMP
  std::cout << mat_M2L.size() << std::endl;
  for(int i=0; i<mat_M2L.size(); ++i) {
    int len = mat_M2L[i].size();
    for(int j=0; j<len; j+=3) {
      std::cout << mat_M2L[i][j] << std::endl;
    }
  }
#endif

#if TEST_SCGEMM
  std::cout << M2M_U.size() << std::endl;
  ComplexVec q(NSURF), result(NSURF);
  for(int i=0; i<NSURF; ++i) {
    q[i] = complex_t(drand48(), drand48()); 
  }
  gemm(1, NSURF, NSURF, &q[0], &M2M_U[0], &result[0]);
  for(int i=0; i<NSURF; ++i) {
    std::cout << result[i] << std::endl;
  }
#endif
  return 0;
}
