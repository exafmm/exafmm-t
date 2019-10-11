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
#include "traverse.h"

namespace py = pybind11;

namespace exafmm_t {
  int P;
  int NSURF;
  int MAXLEVEL;
  int NSRCS;
  int NTRGS;
  vec3 X0;
  real_t R0;
  real_t WAVEK;
  Nodes nodes;
  NodePtrs leafs;
  NodePtrs nonleafs;
  Args args;
  
  /**
   * @brief Initialize sources.
   *
   * @param coords Coordinates of sources, an n-by-3 numpy array.
   * @param charges Charges of sources, an n-element numpy array.
   * @return Bodies Vector of sources.
   */
  Bodies init_sources(py::array_t<real_t> coords, py::array_t<real_t> charges) {
    // checking dimensions
    if (coords.ndim() != 2 || charges.ndim() != 1 || coords.shape(1) != 3)
      throw std::runtime_error("coords should have a shape of (n, 3), charges should have a shape of (n,)");
    if (coords.shape(0) != charges.shape(0))
      throw std::runtime_error("Number of coords and charges must match");
    
    const ssize_t nsrcs = charges.size();
    NSRCS = nsrcs;
    Bodies sources(nsrcs);

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
  Bodies init_targets(py::array_t<real_t> coords) {
    // checking dimensions
    if (coords.ndim() != 2 || coords.shape(1) != 3)
      throw std::runtime_error("coords should have a shape of (n, 3)");
    
    const ssize_t ntrgs = coords.shape(0);
    NTRGS = ntrgs;
    Bodies targets(ntrgs);

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
   * @brief Set FMM parameters.
   * 
   * @param p Order of expansion.
   * @param ncrit Max number of bodies allowed per leaf box.
   * @param max_level Max level of the non-adaptive tree.
   */
  void configure(int p, int ncrit, int max_level) {
    P = p;
    MAXLEVEL = max_level;
    args.P = p;
    args.ncrit = ncrit;
    omp_set_num_threads(args.threads);
    NSURF = 6*(P-1)*(P-1) + 2;
    init_rel_coord();
  }

  /**
   * @brief Given sources and targets, return a 2:1 balanced tree as a vector of nodes.
   * 
   * @param sources Sources.
   * @param targets Targets.
   * @return Nodes Vector of nodes.
   */
  Nodes build_tree(Bodies& sources, Bodies& targets) {
    get_bounds(sources, targets, X0, R0);
#if NON_ADAPTIVE
    nodes = build_tree(sources, targets, X0, R0, leafs, nonleafs);
#else
    nodes = build_tree(sources, targets, X0, R0, leafs, nonleafs, args);
    balance_tree(nodes, sources, targets, X0, R0, leafs, nonleafs, args);
#endif
    return nodes;
  }

  /**
   * @brief Create colleagues list, interaction lists and setup M2L kernel.
   * 
   * @param skip_P2P A flag to switch off P2P (near-field interaction).
   * @return Nodes Vector of nodes.
   */
  Nodes build_list(bool skip_P2P) {
    set_colleagues(nodes);
    build_list(nodes, skip_P2P);
    M2L_setup(nonleafs);
    return nodes;
  }

  /**
   * @brief Evaluate the potentials and the gradients of target points.
   * 
   * @return A numpy array of the potentials of targets.
   */
  py::array_t<real_t> evaluate() {
    // redirect ostream to python ouptut
    py::scoped_ostream_redirect stream(
        std::cout,                                 // std::ostream&
        py::module::import("sys").attr("stdout")   // Python output
    );
    upward_pass(nodes, leafs);
    downward_pass(nodes, leafs);
    
    auto potentials = py::array_t<real_t>(NTRGS);
    py::buffer_info buf = potentials.request();
    real_t * potential_ = (real_t *) buf.ptr;

#pragma omp parallel for
    for (int i=0; i<leafs.size(); ++i) {
      Node * leaf = leafs[i];
      std::vector<int> & itrgs = leaf->itrgs;
      for (int j=0; j<itrgs.size(); ++j) {
        potential_[itrgs[j]] = leaf->trg_value[4*j+0];
      }
    }
    return potentials;
  }

  /**
   * @brief Update the charges of sources. (Coordinates do not change)
   * 
   * @param charges Charges of sources, a numpy array.
   */
  void update(py::array_t<real_t> charges) {
    // update charges of sources
    auto charges_ = charges.unchecked<1>();
#pragma omp parallel for
    for (int i=0; i<leafs.size(); ++i) {
      Node * leaf = leafs[i];
      std::vector<int> & isrcs = leaf->isrcs;
      for (int j=0; j<isrcs.size(); ++j) {
        leaf->src_value[j] = charges_[isrcs[j]];
      }
    }
  }

  /**
   * @brief Reset target values, equivalent charges and check potentials to 0.
   * 
   */
  void clear() {
#pragma omp parallel for
    for (int i=0; i<nodes.size(); ++i) {
      Node & node = nodes[i];
      std::fill(node.up_equiv.begin(), node.up_equiv.end(), 0.);
      std::fill(node.dn_equiv.begin(), node.dn_equiv.end(), 0.);
      if (node.is_leaf)
        std::fill(node.trg_value.begin(), node.trg_value.end(), 0.);
    }
  }

  /**
   * @brief Check the accuracy of FMM against direct evaluation.
   * 
   */
  void check_accuracy() {
    // redirect ostream to python ouptut
    py::scoped_ostream_redirect stream(
        std::cout,                                 // std::ostream&
        py::module::import("sys").attr("stdout")   // Python output
    );
    RealVec error = verify(leafs);
    print_divider("Error");
    print("Potential Error", error[0]);
    print("Gradient Error", error[1]);
  }
 
  void exafmm_main() {
    // redirect ostream to python ouptut
    py::scoped_ostream_redirect stream(
        std::cout,                                 // std::ostream&
        py::module::import("sys").attr("stdout")); // Python output

    omp_set_num_threads(args.threads);
    WAVEK = args.k;
    size_t N = args.numBodies;
    P = args.P;
    NSURF = 6*(P-1)*(P-1) + 2;
    print_divider("Parameters");
    args.print();

    print_divider("Time");
    start("Total");
    Bodies sources = init_bodies(args.numBodies, args.distribution, 0, true);
    Bodies targets = init_bodies(args.numBodies, args.distribution, 5, false);

    start("Build Tree");
    get_bounds(sources, targets, X0, R0);
#if NON_ADAPTIVE
    nodes = build_tree(sources, targets, X0, R0, leafs, nonleafs);
#else
    Nodes nodes = build_tree(sources, targets, X0, R0, leafs, nonleafs, args);
    balance_tree(nodes, sources, targets, X0, R0, leafs, nonleafs, args);
#endif
    stop("Build Tree");

    init_rel_coord();
    start("Precomputation");
    precompute();
    stop("Precomputation");
    start("Build Lists");
    set_colleagues(nodes);
    build_list(nodes);
    stop("Build Lists");
    M2L_setup(nonleafs);
    upward_pass(nodes, leafs);
    downward_pass(nodes, leafs);
    stop("Total");

    RealVec error = verify(leafs);
    
    print_divider("Error");
    print("Potential Error", error[0]);
    print("Gradient Error", error[1]);
    
    print_divider("Tree");
    print("Root Center x", X0[0]);
    print("Root Center y", X0[1]);
    print("Root Center z", X0[2]);
    print("Root Radius R", R0);
    print("Tree Depth", MAXLEVEL);
    print("Leaf Nodes", leafs.size());

    return;
  }
}

PYBIND11_MODULE(exafmm, m) {
  m.doc() = "exafmm's pybind11 module";

  py::module m0 = m.def_submodule("laplace", "A submodule of exafmm's Laplace kernel");
  m0.doc() = "exafmm's submodule for Laplace kernel.";

  py::class_<exafmm_t::vec3>(m0, "vec3")
     .def("__str__", [](const exafmm_t::vec3 &x) {
         return "[" + std::to_string(x[0]) + ", "
                    + std::to_string(x[1]) + ", "
                    + std::to_string(x[2]) + "]";
     })
     .def("__getitem__", [](const exafmm_t::vec3 &x, int i) {  // should check i and throw out of bound error
         return x[i];
     }, py::is_operator())
     .def("__setitem__", [](exafmm_t::vec3 &x, int i, exafmm_t::real_t value) {
         x[i] = value;
     }, py::is_operator())
     .def(py::init<>());

  py::class_<exafmm_t::Body>(m0, "Body")
     .def_readwrite("q", &exafmm_t::Body::q)
     .def_readwrite("p", &exafmm_t::Body::p)
     .def_readwrite("X", &exafmm_t::Body::X)
     .def_readwrite("F", &exafmm_t::Body::F)
     .def_readwrite("ibody", &exafmm_t::Body::ibody)
     .def(py::init<>());

  py::class_<exafmm_t::Node>(m0, "Node")
     .def_readwrite("isrcs", &exafmm_t::Node::isrcs)
     .def_readwrite("itrgs", &exafmm_t::Node::itrgs)
     .def_readwrite("trg_value", &exafmm_t::Node::trg_value)
     .def_readwrite("key", &exafmm_t::Node::key)
     .def_readwrite("parent", &exafmm_t::Node::parent)
     .def_readwrite("colleagues", &exafmm_t::Node::colleagues)
     .def_readwrite("x", &exafmm_t::Node::x)
     .def_readwrite("r", &exafmm_t::Node::r)
     .def_readwrite("nsrcs", &exafmm_t::Node::nsrcs)
     .def_readwrite("ntrgs", &exafmm_t::Node::ntrgs)
     .def_readwrite("level", &exafmm_t::Node::level)
     .def_readwrite("is_leaf", &exafmm_t::Node::is_leaf)
     .def(py::init<>());

  m0.def("init_sources", &exafmm_t::init_sources, "initialize sources");

  m0.def("init_targets", &exafmm_t::init_targets, "initialize targets");

  m0.def("configure", &exafmm_t::configure, "set fmm parameters: p and ncrit");

  m0.def("build_tree", py::overload_cast<exafmm_t::Bodies&, exafmm_t::Bodies&>(&exafmm_t::build_tree), 
        py::return_value_policy::reference, "build tree");

  m0.def("build_list", py::overload_cast<bool>(&exafmm_t::build_list), "build list");

  m0.def("precompute", &exafmm_t::precompute, "precompute translation matrices");

  m0.def("evaluate", &exafmm_t::evaluate,
        py::return_value_policy::reference, "evaluate potential and force at targets");

  m0.def("update", &exafmm_t::update, "update charges of sources");
  
  m0.def("clear", &exafmm_t::clear, "clear target potentials, equivalent charges and check potentials");

  m0.def("check_accuracy", &exafmm_t::check_accuracy, "check accuracy");

  m0.def("exafmm_main", &exafmm_t::exafmm_main, "exafmm's main function");
}
