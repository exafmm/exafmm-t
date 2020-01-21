#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <iostream>
#include "exafmm_t.h"
#include "dataset.h"
#if NON_ADAPTIVE
#include "build_non_adaptive_tree.h"
#else
#include "build_tree.h"
#endif
#include "build_list.h"
#include "laplace.h"
#include "helmholtz.h"
#include "modified_helmholtz.h"

namespace py = pybind11;
using real_t = exafmm_t::real_t;
using complex_t = exafmm_t::complex_t;
template <typename T> using Body = exafmm_t::Body<T>;
template <typename T> using Bodies = exafmm_t::Bodies<T>;
template <typename T> using Node = exafmm_t::Node<T>;
template <typename T> using Nodes = exafmm_t::Nodes<T>;
template <typename T> using NodePtrs = exafmm_t::NodePtrs<T>;

/**
 * @brief Initialize sources with real-type charges.
 *
 * @param coords Coordinates of sources, an n-by-3 numpy array.
 * @param charges Charges of sources, an n-element numpy array.
 * @return Bodies Vector of sources.
 */
Bodies<real_t> init_sources(py::array_t<real_t> coords, py::array_t<real_t> charges) {
  // checking dimensions
  if (coords.ndim() != 2 || charges.ndim() != 1 || coords.shape(1) != 3)
    throw std::runtime_error("coords should have a shape of (n, 3), charges should have a shape of (n,)");
  if (coords.shape(0) != charges.shape(0))
    throw std::runtime_error("Number of coords and charges must match");
  const ssize_t nsrcs = charges.size();
  Bodies<real_t> sources(nsrcs);
  auto coords_ = coords.unchecked<2>();
  auto charges_ = charges.unchecked<1>();
#pragma omp parallel for
  for(ssize_t i=0; i<nsrcs; ++i) {
    sources[i].X[0] = coords_(i, 0);
    sources[i].X[1] = coords_(i, 1);
    sources[i].X[2] = coords_(i, 2);
    sources[i].q = charges_(i);
    sources[i].ibody = i;
  }
  return sources;
}

/**
 * @brief Initialize sources with complex-type charges.
 *
 * @param coords Coordinates of sources, an n-by-3 numpy array.
 * @param charges Charges of sources, an n-element numpy array.
 * @return Bodies Vector of sources.
 */
Bodies<complex_t> init_sources(py::array_t<real_t> coords, py::array_t<complex_t> charges) {
  // checking dimensions
  if (coords.ndim() != 2 || charges.ndim() != 1 || coords.shape(1) != 3)
    throw std::runtime_error("coords should have a shape of (n, 3), charges should have a shape of (n,)");
  if (coords.shape(0) != charges.shape(0))
    throw std::runtime_error("Number of coords and charges must match");
  const ssize_t nsrcs = charges.size();
  Bodies<complex_t> sources(nsrcs);
  auto coords_ = coords.unchecked<2>();
  auto charges_ = charges.unchecked<1>();
#pragma omp parallel for
  for(ssize_t i=0; i<nsrcs; ++i) {
    sources[i].X[0] = coords_(i, 0);
    sources[i].X[1] = coords_(i, 1);
    sources[i].X[2] = coords_(i, 2);
    sources[i].q = charges_(i);
    sources[i].ibody = i;
  }
  return sources;
}

/**
 * @brief Initialize targets.
 * 
 * @param coords Coordinates of targets, an n-by-3 numpy array.
 * @return Bodies Vector of targets.
 */
template <typename T>
Bodies<T> init_targets(py::array_t<real_t> coords) {
  // checking dimensions
  if (coords.ndim() != 2 || coords.shape(1) != 3)
    throw std::runtime_error("coords should have a shape of (n, 3)");
  const ssize_t ntrgs = coords.shape(0);
  Bodies<T> targets(ntrgs);
  auto coords_ = coords.unchecked<2>();
#pragma omp parallel for
  for(ssize_t i=0; i<ntrgs; ++i) {
    targets[i].X[0] = coords_(i, 0);
    targets[i].X[1] = coords_(i, 1);
    targets[i].X[2] = coords_(i, 2);
    targets[i].ibody = i;
  }
  return targets;
}

/**
 * @brief Structure of tree.
 *
 * @tparam T Value type of sources and targets.
 */
template <typename T>
struct Tree {
public:
  Nodes<T> nodes;          //!< Vector of all nodes in the tree
  NodePtrs<T> leafs;       //!< Vector of leaf pointers
  NodePtrs<T> nonleafs;    //!< Vector of nonleaf pointers
};

/**
 * @brief Construct a tree using sources and targets.
 *
 * @tparam T Value type of sources and targets.
 * @param sources Array of sources.
 * @param targets Array of targets.
 * @param fmm The FMM instance.
 * @return The tree.
 */
template <typename T>
Tree<T> build_tree(Bodies<T>& sources, Bodies<T>& targets, exafmm_t::FMM& fmm) {
  exafmm_t::get_bounds<T>(sources, targets, fmm.x0, fmm.r0);
  Tree<T> tree;
#if NON_ADAPTIVE
  tree.nodes = exafmm_t::build_tree<T>(sources, targets, tree.leafs, tree.nonleafs, fmm);
#else
  tree.nodes = exafmm_t::build_tree<T>(sources, targets, tree.leafs, tree.nonleafs, fmm);
  exafmm_t::balance_tree(nodes, sources, targets, leafs, nonleafs, fmm);
#endif
  return tree;
}

/**
 * @brief Create colleagues list, interaction lists and setup M2L kernel.
 * 
 * @tparam T Value type of sources and targets.
 * @param tree The octree.
 * @param fmm The FMM instance.
 * @param skip_P2P A flag to switch off P2P (near-field interaction).
 */
template <typename T>
void build_list(Tree<T>& tree, exafmm_t::FMM& fmm, bool skip_P2P) {
  exafmm_t::set_colleagues<T>(tree.nodes);
  exafmm_t::build_list<T>(tree.nodes, fmm, skip_P2P);
}

Tree<real_t> setup_laplace(Bodies<real_t>& sources, Bodies<real_t>& targets, exafmm_t::LaplaceFMM& fmm, bool skip_P2P) {
  auto tree = build_tree<real_t>(sources, targets, fmm);
  exafmm_t::init_rel_coord();
  build_list<real_t>(tree, fmm, skip_P2P);
  fmm.M2L_setup(tree.nonleafs);
  fmm.precompute();
  return tree;
}

Tree<complex_t> setup_helmholtz(Bodies<complex_t>& sources, Bodies<complex_t>& targets, exafmm_t::HelmholtzFMM& fmm, bool skip_P2P) {
  auto tree = build_tree<complex_t>(sources, targets, fmm);
  exafmm_t::init_rel_coord();
  build_list<complex_t>(tree, fmm, skip_P2P);
  fmm.M2L_setup(tree.nonleafs);
  fmm.precompute();
  return tree;
}

Tree<real_t> setup_modified_helmholtz(Bodies<real_t>& sources, Bodies<real_t>& targets, exafmm_t::ModifiedHelmholtzFMM& fmm, bool skip_P2P) {
  auto tree = build_tree<real_t>(sources, targets, fmm);
  exafmm_t::init_rel_coord();
  build_list<real_t>(tree, fmm, skip_P2P);
  fmm.M2L_setup(tree.nonleafs);
  fmm.precompute();
  return tree;
}

/**
 * @brief Evaluate Laplace potential and gradient at targets.
 *
 * @param tree The octree.
 * @param fmm Laplace FMM instance.
 * @return trg_value Potential and gradient of targets, an n_trg-by-4 numpy array.
 */
py::array_t<real_t> evaluate(Tree<real_t>& tree, exafmm_t::LaplaceFMM& fmm) {
  // redirect ostream to python ouptut
  py::scoped_ostream_redirect stream(
      std::cout,                                 // std::ostream&
      py::module::import("sys").attr("stdout")   // Python output
  );

  fmm.upward_pass(tree.nodes, tree.leafs);
  fmm.downward_pass(tree.nodes, tree.leafs);

  auto trg_value = py::array_t<real_t>({tree.nodes[0].ntrgs, 4});
  auto r = trg_value.mutable_unchecked<2>();  // access function

#pragma omp parallel for
  for (int i=0; i<tree.leafs.size(); ++i) {
    Node<real_t>* leaf = tree.leafs[i];
    std::vector<int> & itrgs = leaf->itrgs;
    for (int j=0; j<itrgs.size(); ++j) {
      r(itrgs[j], 0) = leaf->trg_value[4*j+0];
      r(itrgs[j], 1) = leaf->trg_value[4*j+1];
      r(itrgs[j], 2) = leaf->trg_value[4*j+2];
      r(itrgs[j], 3) = leaf->trg_value[4*j+3];
    }
  }
  return trg_value;
}

/**
 * @brief Evaluate Helmholtz potential and gradient at targets.
 *
 * @param tree The octree.
 * @param fmm Helmholtz FMM instance.
 * @return trg_value Potential and gradient of targets, an n_trg-by-4 numpy array.
 */
py::array_t<complex_t> evaluate_h(Tree<complex_t>& tree, exafmm_t::HelmholtzFMM& fmm) {
  // redirect ostream to python ouptut
  py::scoped_ostream_redirect stream(
      std::cout,                                 // std::ostream&
      py::module::import("sys").attr("stdout")   // Python output
  );

  fmm.upward_pass(tree.nodes, tree.leafs);
  fmm.downward_pass(tree.nodes, tree.leafs);
  
  auto trg_value = py::array_t<complex_t>({tree.nodes[0].ntrgs, 4});
  auto r = trg_value.mutable_unchecked<2>();  // access function

#pragma omp parallel for
  for (int i=0; i<tree.leafs.size(); ++i) {
    Node<complex_t>* leaf = tree.leafs[i];
    std::vector<int> & itrgs = leaf->itrgs;
    for (int j=0; j<itrgs.size(); ++j) {
      r(itrgs[j], 0) = leaf->trg_value[4*j+0];
      r(itrgs[j], 1) = leaf->trg_value[4*j+1];
      r(itrgs[j], 2) = leaf->trg_value[4*j+2];
      r(itrgs[j], 3) = leaf->trg_value[4*j+3];
    }
  }
  return trg_value;
}

/**
 * @brief Evaluate modified Helmholtz potential and gradient at targets.
 *
 * @param tree The octree.
 * @param fmm The modified Helmholtz FMM instance.
 * @return trg_value Potential and gradient of targets, an n_trg-by-4 numpy array.
 */
py::array_t<real_t> evaluate_m(Tree<real_t>& tree, exafmm_t::ModifiedHelmholtzFMM& fmm) {
  // redirect ostream to python ouptut
  py::scoped_ostream_redirect stream(
      std::cout,                                 // std::ostream&
      py::module::import("sys").attr("stdout")   // Python output
  );

  fmm.upward_pass(tree.nodes, tree.leafs);
  fmm.downward_pass(tree.nodes, tree.leafs);

  auto trg_value = py::array_t<real_t>({tree.nodes[0].ntrgs, 4});
  auto r = trg_value.mutable_unchecked<2>();  // access function

#pragma omp parallel for
  for (int i=0; i<tree.leafs.size(); ++i) {
    Node<real_t>* leaf = tree.leafs[i];
    std::vector<int> & itrgs = leaf->itrgs;
    for (int j=0; j<itrgs.size(); ++j) {
      r(itrgs[j], 0) = leaf->trg_value[4*j+0];
      r(itrgs[j], 1) = leaf->trg_value[4*j+1];
      r(itrgs[j], 2) = leaf->trg_value[4*j+2];
      r(itrgs[j], 3) = leaf->trg_value[4*j+3];
    }
  }
  return trg_value;
}

/**
 * @brief Update the charges of sources (real type).
 * 
 * @param tree The octree.
 * @param charges Charges of sources, a numpy array.
 */
void update_charges_real(Tree<real_t>& tree, py::array_t<real_t>& charges) {
  // update charges of sources
  auto charges_ = charges.unchecked<1>();
#pragma omp parallel for
  for (int i=0; i<tree.leafs.size(); ++i) {
    auto leaf = tree.leafs[i];
    std::vector<int>& isrcs = leaf->isrcs;
    for (int j=0; j<isrcs.size(); ++j) {
      leaf->src_value[j] = charges_[isrcs[j]];
    }
  }
}

/**
 * @brief Update the charges of sources (complex type).
 * 
 * @param tree The octree.
 * @param charges Charges of sources, a numpy array.
 */
void update_charges_cplx(Tree<complex_t>& tree, py::array_t<complex_t>& charges) {
  // update charges of sources
  auto charges_ = charges.unchecked<1>();
#pragma omp parallel for
  for (int i=0; i<tree.leafs.size(); ++i) {
    auto leaf = tree.leafs[i];
    std::vector<int>& isrcs = leaf->isrcs;
    for (int j=0; j<isrcs.size(); ++j) {
      leaf->src_value[j] = charges_[isrcs[j]];
    }
  }
}

/**
 * @brief Reset target values, equivalent charges and check potentials to 0.
 * 
 * @tparam T Value type of sources and targets.
 * @param The Octree.
 */
template <typename T>
void clear_values(Tree<T>& tree) {
#pragma omp parallel for
  for (int i=0; i<tree.nodes.size(); ++i) {
    auto& node = tree.nodes[i];
    std::fill(node.up_equiv.begin(), node.up_equiv.end(), 0.);
    std::fill(node.dn_equiv.begin(), node.dn_equiv.end(), 0.);
    if (node.is_leaf)
      std::fill(node.trg_value.begin(), node.trg_value.end(), 0.);
  }
}


PYBIND11_MODULE(exafmm, m) {
  /**
   * m:  exafmm-t's module.
   *     - class: vec3
   *     - class: FMM
   *     - function: init_rel_coord()
   * m0: Laplace submodule (real type).
   *     - class: Body
   *     - class: Node
   *     - class: Tree
   *     - class: LaplaceFMM
   *     - function: init_sources()
   *     - function: init_targets()
   *     - function: build_tree()
   *     - function: build_list()
   *     - function: evaluate()
   * m1: Helmholtz submodule (complex type).
   *     - class: Body
   *     - class: Node
   *     - class: Tree
   *     - class: HelmholtzFMM
   *     - function: init_sources()
   *     - function: init_targets()
   *     - function: build_tree()
   *     - function: build_list()
   *     - function: evaluate()
   * m2: Modified Helmholtz submodule (real type).
   *     - class: Body, Node, Tree are aliases from Laplace submodule.
   *     - class: ModifiedHelmholtzFMM
   *     - function: init_sources(), init_targets(), build_tree(), build_list() are aliases from Laplace submodule.  
   *     - function: evaluate()
   */
  m.doc() = "exafmm's pybind11 module";

  py::module m0 = m.def_submodule("laplace", "A submodule of exafmm's Laplace kernel");
  m0.doc() = "exafmm's submodule for Laplace kernel";

  py::module m1 = m.def_submodule("helmholtz", "A submodule of exafmm's Helmholtz kernel");
  m1.doc() = "exafmm's submodule for Helmholtz kernel";
  
  py::module m2 = m.def_submodule("modified_helmholtz", "A submodule of exafmm's Modified Helmholtz kernel");
  m2.doc() = "exafmm's submodule for Modified Helmholtz kernel";

  py::class_<exafmm_t::vec3>(m, "vec3")
     .def("__str__", [](const exafmm_t::vec3 &x) {
         return "[" + std::to_string(x[0]) + ", "
                    + std::to_string(x[1]) + ", "
                    + std::to_string(x[2]) + "]";
     })
     .def("__getitem__", [](const exafmm_t::vec3 &x, int i) {  // should check i and throw out of bound error
         return x[i];
     }, py::is_operator())
     .def("__setitem__", [](exafmm_t::vec3 &x, int i, real_t value) {
         x[i] = value;
     }, py::is_operator())
     .def(py::init<>());

  py::class_<exafmm_t::FMM>(m, "FMM")
     .def_readwrite("p", &exafmm_t::FMM::p)
     .def_readwrite("nsurf", &exafmm_t::FMM::nsurf)
     .def_readwrite("depth", &exafmm_t::FMM::depth)
     .def_readwrite("ncrit", &exafmm_t::FMM::ncrit)
     .def_readwrite("r0", &exafmm_t::FMM::r0)
     .def_readwrite("x0", &exafmm_t::FMM::x0)
     .def(py::init<>());

  m.def("init_rel_coord", py::overload_cast<>(&exafmm_t::init_rel_coord), "initialize relative coordinates matrix");

  py::class_<Body<real_t>>(m0, "Body")
     .def_readwrite("q", &Body<real_t>::q)
     .def_readwrite("p", &Body<real_t>::p)
     .def_readwrite("X", &Body<real_t>::X)
     .def_readwrite("F", &Body<real_t>::F)
     .def_readwrite("ibody", &Body<real_t>::ibody)
     .def(py::init<>());

  py::class_<Node<real_t>>(m0, "Node")
     .def_readwrite("isrcs", &Node<real_t>::isrcs)
     .def_readwrite("itrgs", &Node<real_t>::itrgs)
     .def_readwrite("trg_value", &Node<real_t>::trg_value)
     .def_readwrite("key", &Node<real_t>::key)
     .def_readwrite("parent", &Node<real_t>::parent)
     .def_readwrite("colleagues", &Node<real_t>::colleagues)
     .def_readwrite("x", &Node<real_t>::x)
     .def_readwrite("r", &Node<real_t>::r)
     .def_readwrite("nsrcs", &Node<real_t>::nsrcs)
     .def_readwrite("ntrgs", &Node<real_t>::ntrgs)
     .def_readwrite("level", &Node<real_t>::level)
     .def_readwrite("is_leaf", &Node<real_t>::is_leaf)
     .def(py::init<>());

  py::class_<Tree<real_t>>(m0, "Tree")
     .def_readwrite("nodes", &Tree<real_t>::nodes)
     .def_readwrite("leafs", &Tree<real_t>::leafs)
     .def_readwrite("nonleafs", &Tree<real_t>::nonleafs)
     .def(py::init<>());

  py::class_<exafmm_t::LaplaceFMM, exafmm_t::FMM>(m0, "LaplaceFMM")
     .def("precompute", &exafmm_t::LaplaceFMM::precompute)
     .def("M2L_setup", &exafmm_t::LaplaceFMM::M2L_setup)
     .def("verify", &exafmm_t::LaplaceFMM::verify)
     .def(py::init<>());

  py::class_<Body<complex_t>>(m1, "Body")
     .def_readwrite("q", &Body<complex_t>::q)
     .def_readwrite("p", &Body<complex_t>::p)
     .def_readwrite("X", &Body<complex_t>::X)
     .def_readwrite("F", &Body<complex_t>::F)
     .def_readwrite("ibody", &Body<complex_t>::ibody)
     .def(py::init<>());

  py::class_<Node<complex_t>>(m1, "Node")
     .def_readwrite("isrcs", &Node<complex_t>::isrcs)
     .def_readwrite("itrgs", &Node<complex_t>::itrgs)
     .def_readwrite("trg_value", &Node<complex_t>::trg_value)
     .def_readwrite("key", &Node<complex_t>::key)
     .def_readwrite("parent", &Node<complex_t>::parent)
     .def_readwrite("colleagues", &Node<complex_t>::colleagues)
     .def_readwrite("x", &Node<complex_t>::x)
     .def_readwrite("r", &Node<complex_t>::r)
     .def_readwrite("nsrcs", &Node<complex_t>::nsrcs)
     .def_readwrite("ntrgs", &Node<complex_t>::ntrgs)
     .def_readwrite("level", &Node<complex_t>::level)
     .def_readwrite("is_leaf", &Node<complex_t>::is_leaf)
     .def(py::init<>());

  py::class_<Tree<complex_t>>(m1, "Tree")
     .def_readwrite("nodes", &Tree<complex_t>::nodes)
     .def_readwrite("leafs", &Tree<complex_t>::leafs)
     .def_readwrite("nonleafs", &Tree<complex_t>::nonleafs)
     .def(py::init<>());

  py::class_<exafmm_t::HelmholtzFMM, exafmm_t::FMM>(m1, "HelmholtzFMM")
     .def("precompute", &exafmm_t::HelmholtzFMM::precompute)
     .def("M2L_setup", &exafmm_t::HelmholtzFMM::M2L_setup)
     .def("verify", &exafmm_t::HelmholtzFMM::verify)
     .def_readwrite("wavek", &exafmm_t::HelmholtzFMM::wavek)
     .def(py::init<>());

  m2.attr("Body") = m0.attr("Body");
  m2.attr("Node") = m0.attr("Node");
  m2.attr("Tree") = m0.attr("Tree");

  py::class_<exafmm_t::ModifiedHelmholtzFMM, exafmm_t::FMM>(m2, "ModifiedHelmholtzFMM")
     .def("precompute", &exafmm_t::ModifiedHelmholtzFMM::precompute)
     .def("M2L_setup", &exafmm_t::ModifiedHelmholtzFMM::M2L_setup)
     .def("verify", &exafmm_t::ModifiedHelmholtzFMM::verify)
     .def_readwrite("wavek", &exafmm_t::ModifiedHelmholtzFMM::wavek)
     .def(py::init<>());

  /* Cannot register the same C++ class/function two times in Python interface.
     The workaround is to create alias on Python side, see https://github.com/pybind/pybind11/issues/439.
     Laplace and Modified Helmholtz kernels share a lot of functions in common (real-type version),
     so we create definitions of Python bindings in Laplace module and use aliases in Modified Helmholtz module. */
  m0.def("init_sources", py::overload_cast<py::array_t<real_t>, py::array_t<real_t>>(&init_sources), "initialize sources");
  m1.def("init_sources", py::overload_cast<py::array_t<real_t>, py::array_t<complex_t>>(&init_sources), "initialize sources");
  m2.attr("init_sources") = m0.attr("init_sources");

  m0.def("init_targets", &init_targets<real_t>, "initialize targets");
  m1.def("init_targets", &init_targets<complex_t>, "initialize targets");
  m2.attr("init_targets") = m0.attr("init_targets");

  m0.def("build_tree", &build_tree<real_t>, py::return_value_policy::reference, "build tree");
  m1.def("build_tree", &build_tree<complex_t>, py::return_value_policy::reference, "build tree");
  m2.attr("build_tree") = m0.attr("build_tree");

  m0.def("build_list", &build_list<real_t>, "build list");
  m1.def("build_list", &build_list<complex_t>, "build list");
  m2.attr("build_list") = m0.attr("build_list");

  m0.def("setup", &setup_laplace, "setup FMM, including tree construction, list construction, M2L setup and pre-computation.");
  m1.def("setup", &setup_helmholtz, "setup FMM, including tree construction, list construction, M2L setup and pre-computation.");
  m2.def("setup", &setup_modified_helmholtz, "setup FMM, including tree construction, list construction, M2L setup and pre-computation.");

  m0.def("evaluate", &evaluate, "evaluate");
  m1.def("evaluate", &evaluate_h, "evaluate");
  m2.def("evaluate", &evaluate_m, "evaluate");

  m0.def("update_charges", &update_charges_real, "update charges of sources");
  m1.def("update_charges", &update_charges_cplx, "update charges of sources");
  m2.attr("update_charges") = m0.attr("update_charges");

  m0.def("clear_values", &clear_values<real_t>, "clear target potentials, equivalent charges and check potentials");
  m1.def("clear_values", &clear_values<complex_t>, "clear target potentials, equivalent charges and check potentials");
  m2.attr("clear_values") = m0.attr("clear_values");
}