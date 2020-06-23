#ifndef fmm_base_h
#define fmm_base_h
#include <algorithm>    // std::fill
#include "exafmm_t.h"
#include "geometry.h"
#include "timer.h"

namespace exafmm_t {
  //! Base FMM class
  template <typename T>
  class FmmBase {
  public:
    int p;                 //!< Order of expansion
    int nsurf;             //!< Number of points on equivalent / check surface
    int nconv;             //!< Number of points on convolution grid
    int nfreq;             //!< Number of coefficients in DFT (depending on whether T is real_t)
    int ncrit;             //!< Max number of bodies per leaf
    int depth;             //!< Depth of the tree
    real_t r0;             //!< Half of the side length of the bounding box
    vec3 x0;               //!< Coordinates of the center of root box
    bool is_precomputed;   //!< Whether the matrix file is found
    bool is_real;          //!< Whether template parameter T is real_t
    std::string filename;  //!< File name of the precomputation matrices

    FmmBase() {}
    FmmBase(int p_, int ncrit_, int depth_, std::string filename_=std::string()) :
      p(p_), ncrit(ncrit_), depth(depth_), filename(filename_)
    {
      nsurf = 6*(p_-1)*(p_-1) + 2;
      int n1 = 2 * p_;
      nconv = n1 * n1 * n1;
      is_real = std::is_same<T, real_t>::value;
      nfreq = is_real ? n1*n1*(n1/2+1) : nconv;
      is_precomputed = false;
    }

    virtual void potential_P2P(RealVec& src_coord, std::vector<T>& src_value,
                               RealVec& trg_coord, std::vector<T>& trg_value) = 0;

    virtual void gradient_P2P(RealVec& src_coord, std::vector<T>& src_value,
                              RealVec& trg_coord, std::vector<T>& trg_value) = 0;
    //! M2L operator.
    virtual void M2L(Nodes<T>& nodes) = 0;

    //! M2M operator.
    virtual void M2M(Node<T>* node) = 0;

    //! L2L operator.
    virtual void L2L(Node<T>* node) = 0;
    
    //! P2M operator.
    virtual void P2M(NodePtrs<T>& leafs) = 0;
    
    //! L2P operator.
    virtual void L2P(NodePtrs<T>& leafs) = 0;

    /**
     * @brief Compute the kernel matrix of a given kernel.
     * 
     * @param src_coord Vector of source coordinates.
     * @param trg_coord Vector of target coordinates.
     * @param matrix Kernel matrix.
     */
    void kernel_matrix(RealVec& src_coord, RealVec& trg_coord, std::vector<T>& matrix) {
      std::vector<T> src_value(1, 1.);  // use unit weight to generate kernel matrix
      int nsrcs = src_coord.size() / 3;
      int ntrgs = trg_coord.size() / 3;
#pragma omp parallel for
      for (int i=0; i<nsrcs; i++) {
        RealVec src_coord_(src_coord.data()+3*i, src_coord.data()+3*(i+1));
        std::vector<T> trg_value_(ntrgs, 0.);
        potential_P2P(src_coord_, src_value, trg_coord, trg_value_);
        std::copy(trg_value_.begin(), trg_value_.end(), matrix.data()+i*ntrgs);
      }
    }

    /* the following kernels do not use precomputation matrices
     * thus can be defined in the base class */

    //! P2P operator.
    void P2P(NodePtrs<T>& leafs) {
      NodePtrs<T>& targets = leafs;
#pragma omp parallel for
      for (size_t i=0; i<targets.size(); i++) {
        Node<T>* target = targets[i];
        NodePtrs<T>& sources = target->P2P_list;
        for (size_t j=0; j<sources.size(); j++) {
          Node<T>* source = sources[j];
          gradient_P2P(source->src_coord, source->src_value,
                       target->trg_coord, target->trg_value);
        }
      }
    }

    //! M2P operator.
    void M2P(NodePtrs<T>& leafs) {
      NodePtrs<T>& targets = leafs;
      real_t c[3] = {0.0};
      std::vector<RealVec> up_equiv_surf;
      up_equiv_surf.resize(depth+1);
      for (int level=0; level<=depth; level++) {
        up_equiv_surf[level].resize(nsurf*3);
        up_equiv_surf[level] = surface(p, r0, level, c, 1.05);
      }
#pragma omp parallel for
      for (size_t i=0; i<targets.size(); i++) {
        Node<T>* target = targets[i];
        NodePtrs<T>& sources = target->M2P_list;
        for (size_t j=0; j<sources.size(); j++) {
          Node<T>* source = sources[j];
          RealVec src_equiv_coord(nsurf*3);
          int level = source->level;
          // source node's equiv coord = relative equiv coord + node's center
          for (int k=0; k<nsurf; k++) {
            src_equiv_coord[3*k+0] = up_equiv_surf[level][3*k+0] + source->x[0];
            src_equiv_coord[3*k+1] = up_equiv_surf[level][3*k+1] + source->x[1];
            src_equiv_coord[3*k+2] = up_equiv_surf[level][3*k+2] + source->x[2];
          }
          gradient_P2P(src_equiv_coord, source->up_equiv,
                       target->trg_coord, target->trg_value);
        }
      }
    }

    //! P2L operator.
    void P2L(Nodes<T>& nodes) {
      Nodes<T>& targets = nodes;
      real_t c[3] = {0.0};
      std::vector<RealVec> dn_check_surf;
      dn_check_surf.resize(depth+1);
      for (int level=0; level<=depth; level++) {
        dn_check_surf[level].resize(nsurf*3);
        dn_check_surf[level] = surface(p, r0, level, c, 1.05);
      }
#pragma omp parallel for
      for (size_t i=0; i<targets.size(); i++) {
        Node<T>* target = &targets[i];
        NodePtrs<T>& sources = target->P2L_list;
        for (size_t j=0; j<sources.size(); j++) {
          Node<T>* source = sources[j];
          RealVec trg_check_coord(nsurf*3);
          int level = target->level;
          // target node's check coord = relative check coord + node's center
          for (int k=0; k<nsurf; k++) {
            trg_check_coord[3*k+0] = dn_check_surf[level][3*k+0] + target->x[0];
            trg_check_coord[3*k+1] = dn_check_surf[level][3*k+1] + target->x[1];
            trg_check_coord[3*k+2] = dn_check_surf[level][3*k+2] + target->x[2];
          }
          potential_P2P(source->src_coord, source->src_value,
                        trg_check_coord, target->dn_equiv);
        }
      }
    }
    
    /**
     * @brief Evaluate upward equivalent charges for all nodes in a post-order traversal.
     * 
     * @param nodes Vector of all nodes.
     * @param leafs Vector of pointers to leaf nodes.
     */   
    void upward_pass(Nodes<T>& nodes, NodePtrs<T>& leafs, bool verbose=true) {
      start("P2M");
      P2M(leafs);
      stop("P2M", verbose);
      start("M2M");
#pragma omp parallel
#pragma omp single nowait
      M2M(&nodes[0]);
      stop("M2M", verbose);
    }

    /**
     * @brief Evaluate potentials and gradients for all targets in a pre-order traversal.
     * 
     * @param nodes Vector of all nodes.
     * @param leafs Vector of pointers to leaf nodes.
     */   
    void downward_pass(Nodes<T>& nodes, NodePtrs<T>& leafs, bool verbose=true) {
      start("P2L");
      P2L(nodes);
      stop("P2L", verbose);
      start("M2P");
      M2P(leafs);
      stop("M2P", verbose);
      start("P2P");
      P2P(leafs);
      stop("P2P", verbose);
      start("M2L");
      M2L(nodes);
      stop("M2L", verbose);
      start("L2L");
#pragma omp parallel
#pragma omp single nowait
      L2L(&nodes[0]);
      stop("L2L", verbose);
      start("L2P");
      L2P(leafs);
      stop("L2P", verbose);
    }

    /**
     * @brief Check FMM accuracy.
     *
     * @param leafs Vector of leaves.
     * @return The relative error of potential and gradient in L2 norm.
     */
    RealVec verify(NodePtrs<T>& leafs, bool sample=false) {
      Nodes<T> targets;  // vector of target nodes
      if (sample) {
        int nsamples = 10;
        int stride = leafs.size() / nsamples;
        for (int i=0; i<nsamples; i++) {
          targets.push_back(*(leafs[i*stride]));
        }
      } else { // compute all values directly without sampling
        for (size_t i=0; i<leafs.size(); i++) {
          targets.push_back(*leafs[i]);
        }
      }

      Nodes<T> targets2 = targets;    // target2 is used for direct summation
#pragma omp parallel for
      for (size_t i=0; i<targets2.size(); i++) {
        Node<T>* target = &targets2[i];
        std::fill(target->trg_value.begin(), target->trg_value.end(), 0.);
        for (size_t j=0; j<leafs.size(); j++) {
          gradient_P2P(leafs[j]->src_coord, leafs[j]->src_value, target->trg_coord, target->trg_value);
        }
      }

      // relative error in L-infinity norm
      /*
      double potential_max_err = -1;
      double gradient_max_err = -1;
      for (size_t i=0; i<targets.size(); i++) {
        for (int j=0; j<targets[i].ntrgs; j++) {
          auto potential_diff = std::abs(targets[i].trg_value[4*j+0] - targets2[i].trg_value[4*j+0]);
          auto potential_norm = std::abs(targets2[i].trg_value[4*j+0]);
          potential_max_err = std::max(potential_max_err, potential_diff/potential_norm);
          for (int d=1; d<4; d++) {
            auto gradient_diff = std::abs(targets[i].trg_value[4*j+d] - targets2[i].trg_value[4*j+d]);
            auto gradient_norm = std::abs(targets2[i].trg_value[4*j+d]);
            gradient_max_err = std::max(gradient_max_err, gradient_diff/gradient_norm);
          }
        }
      }
      */

      // relative error in L2 norm
      double p_diff = 0, p_norm = 0, g_diff = 0, g_norm = 0;
      for (size_t i=0; i<targets.size(); i++) {
        for (int j=0; j<targets[i].ntrgs; j++) {
          p_norm += std::norm(targets2[i].trg_value[4*j+0]);
          p_diff += std::norm(targets2[i].trg_value[4*j+0] - targets[i].trg_value[4*j+0]);
          for (int d=1; d<4; d++) {
            g_diff += std::norm(targets2[i].trg_value[4*j+d] - targets[i].trg_value[4*j+d]);
            g_norm += std::norm(targets2[i].trg_value[4*j+d]);
          }
        }
      }
      RealVec err(2);
      err[0] = sqrt(p_diff/p_norm);   // potential error in L2 norm
      err[1] = sqrt(g_diff/g_norm);   // gradient error in L2 norm

      return err;
    }
  };
}  // end namespace

#endif
