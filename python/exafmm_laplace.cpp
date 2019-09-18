#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <iostream>
#include "exafmm_t.h"
#include "dataset.h"
#include "build_tree.h"
#include "build_list.h"
#include "laplace.h"
#include "traverse.h"

namespace py = pybind11;

namespace exafmm_t {
  int P;
  int NSURF;
  int MAXLEVEL;
  vec3 X0;
  real_t R0;
  real_t WAVEK;
  Nodes nodes;
  NodePtrs leafs;
  NodePtrs nonleafs;
  Args args;

  Bodies init_sources(int nsrcs) {
    return init_bodies(nsrcs, "cube", 0, true);
  }

  Bodies init_targets(int ntrgs) {
    return init_bodies(ntrgs, "cube", 5, false);
  }
 

  void configure(int p, int ncrit) {
    P = p;
    args.P = p;
    args.ncrit = ncrit;
    omp_set_num_threads(args.threads);
    NSURF = 6*(P-1)*(P-1) + 2;
    init_rel_coord();
  }

  Nodes build_tree(Bodies& sources, Bodies& targets) {
    get_bounds(sources, targets, X0, R0);
    nodes = build_tree(sources, targets, X0, R0, leafs, nonleafs, args);
    balance_tree(nodes, sources, targets, X0, R0, leafs, nonleafs, args);
    return nodes;
  }

  void build_list() {
    set_colleagues(nodes);
    build_list(nodes);
    M2L_setup(nonleafs);
  }

  Nodes evaluate() {
    // redirect ostream to python ouptut
    py::scoped_ostream_redirect stream(
        std::cout,                                 // std::ostream&
        py::module::import("sys").attr("stdout")   // Python output
    );
    upward_pass(nodes, leafs);
    downward_pass(nodes, leafs);
    return nodes;
  }

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
    Nodes nodes = build_tree(sources, targets, X0, R0, leafs, nonleafs, args);
    balance_tree(nodes, sources, targets, X0, R0, leafs, nonleafs, args);
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


PYBIND11_MODULE(exafmm_laplace, m) {
  m.doc() = "exafmm's pybind11 module for Laplace kernel";

  py::class_<exafmm_t::vec3>(m, "vec3")
     .def("__getitem__", [](const exafmm_t::vec3 &x, int i) {  // should check i and throw out of bound error
         return x[i];
     }, py::is_operator())
     .def("__setitem__", [](exafmm_t::vec3 &x, int i, exafmm_t::real_t value) {
         x[i] = value;
     }, py::is_operator())
     .def(py::init<>());

  py::class_<exafmm_t::Body>(m, "Body")
     .def_readwrite("q", &exafmm_t::Body::q)
     .def_readwrite("p", &exafmm_t::Body::p)
     .def_readwrite("X", &exafmm_t::Body::X)
     .def_readwrite("F", &exafmm_t::Body::F)
     .def(py::init<>());

  py::class_<exafmm_t::Node>(m, "Node")
     .def_readwrite("key", &exafmm_t::Node::key)
     .def_readwrite("parent", &exafmm_t::Node::parent)
     .def_readwrite("r", &exafmm_t::Node::r)
     .def_readwrite("nsrcs", &exafmm_t::Node::nsrcs)
     .def_readwrite("ntrgs", &exafmm_t::Node::ntrgs)
     .def_readwrite("level", &exafmm_t::Node::level)
     .def_readwrite("is_leaf", &exafmm_t::Node::is_leaf)
     .def(py::init<>());

  m.def("init_sources", &exafmm_t::init_sources, "initialize sources");
  m.def("init_targets", &exafmm_t::init_targets, "initialize targets");
  m.def("configure", &exafmm_t::configure, "set fmm parameters: p and ncrit");
  m.def("build_tree", py::overload_cast<exafmm_t::Bodies&, exafmm_t::Bodies&>(&exafmm_t::build_tree), 
        py::return_value_policy::reference, "build tree");
  m.def("build_list", py::overload_cast<>(&exafmm_t::build_list), "build list");
  m.def("precompute", &exafmm_t::precompute, "precompute translation matrices");
  m.def("evaluate", &exafmm_t::evaluate,
        py::return_value_policy::reference, "evaluate potential and force at targets");
  m.def("check_accuracy", &exafmm_t::check_accuracy, "check accuracy");

  m.def("exafmm_main", &exafmm_t::exafmm_main, "exafmm's main function");
}
