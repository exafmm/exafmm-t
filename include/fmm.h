#ifndef fmm_h
#define fmm_h
#include <cstring>      // std::memset
#include <fstream>      // std::ofstream
#include <type_traits>  // std::is_same
#include "fmm_base.h"
#include "intrinsics.h"
#include "math_wrapper.h"

namespace exafmm_t {
  template <typename T>
  class Fmm : public FmmBase<T> {
  public:
    /* precomputation matrices */
    std::vector<std::vector<T>> matrix_UC2E_U;
    std::vector<std::vector<T>> matrix_UC2E_V;
    std::vector<std::vector<T>> matrix_DC2E_U;
    std::vector<std::vector<T>> matrix_DC2E_V;
    std::vector<std::vector<std::vector<T>>> matrix_M2M;
    std::vector<std::vector<std::vector<T>>> matrix_L2L;

    std::vector<M2LData> m2ldata;

    /* constructors */
    Fmm() {}
    Fmm(int p_, int ncrit_, int depth_) : FmmBase<T>(p_, ncrit_, depth_) {}

    /* precomputation */
    //! Setup the sizes of precomputation matrices
    void initialize_matrix() {

    }

    //! P2M operator
    void P2M(NodePtrs<T>& leafs) {
      int nsurf = this->nsurf;
      real_t c[3] = {0,0,0};
      std::vector<RealVec> up_check_surf;
      up_check_surf.resize(this->depth+1);
      for (int level=0; level<=this->depth; level++) {
        up_check_surf[level].resize(nsurf*3);
        up_check_surf[level] = surface(this->p, this->r0, level, c, 2.95);
      }
#pragma omp parallel for
      for (size_t i=0; i<leafs.size(); i++) {
        Node<T>* leaf = leafs[i];
        int level = leaf->level;
        // calculate upward check potential induced by sources' charges
        RealVec check_coord(nsurf*3);
        for (int k=0; k<nsurf; k++) {
          check_coord[3*k+0] = up_check_surf[level][3*k+0] + leaf->x[0];
          check_coord[3*k+1] = up_check_surf[level][3*k+1] + leaf->x[1];
          check_coord[3*k+2] = up_check_surf[level][3*k+2] + leaf->x[2];
        }
        this->potential_P2P(leaf->src_coord, leaf->src_value,
                            check_coord, leaf->up_equiv);
        std::vector<T> buffer(nsurf);
        std::vector<T> equiv(nsurf);
        gemv(nsurf, nsurf, &(matrix_UC2E_U[level][0]), &(leaf->up_equiv[0]), &buffer[0]);
        gemv(nsurf, nsurf, &(matrix_UC2E_V[level][0]), &buffer[0], &equiv[0]);
        for (int k=0; k<nsurf; k++)
          leaf->up_equiv[k] = equiv[k];
      }
    }

    //! L2P operator
    void L2P(NodePtrs<T>& leafs) {
      int nsurf = this->nsurf;
      real_t c[3] = {0,0,0};
      std::vector<RealVec> dn_equiv_surf;
      dn_equiv_surf.resize(this->depth+1);
      for (int level=0; level<=this->depth; level++) {
        dn_equiv_surf[level].resize(nsurf*3);
        dn_equiv_surf[level] = surface(this->p, this->r0, level, c, 2.95);
      }
#pragma omp parallel for
      for (size_t i=0; i<leafs.size(); i++) {
        Node<T>* leaf = leafs[i];
        int level = leaf->level;
        // down check surface potential -> equivalent surface charge
        std::vector<T> buffer(nsurf);
        std::vector<T> equiv(nsurf);
        gemv(nsurf, nsurf, &(matrix_DC2E_U[level][0]), &(leaf->dn_equiv[0]), &buffer[0]);
        gemv(nsurf, nsurf, &(matrix_DC2E_V[level][0]), &buffer[0], &equiv[0]);
        for (int k=0; k<nsurf; k++)
          leaf->dn_equiv[k] = equiv[k];
        // equivalent surface charge -> target potential
        RealVec equiv_coord(nsurf*3);
        for (int k=0; k<nsurf; k++) {
          equiv_coord[3*k+0] = dn_equiv_surf[level][3*k+0] + leaf->x[0];
          equiv_coord[3*k+1] = dn_equiv_surf[level][3*k+1] + leaf->x[1];
          equiv_coord[3*k+2] = dn_equiv_surf[level][3*k+2] + leaf->x[2];
        }
        this->gradient_P2P(equiv_coord, leaf->dn_equiv,
                           leaf->trg_coord, leaf->trg_value);
      }
    }

    //! M2M operator
    void M2M(Node<T>* node) {
      int nsurf = this->nsurf;
      if (node->is_leaf) return;
      for (int octant=0; octant<8; octant++) {
        if (node->children[octant])
#pragma omp task untied
          M2M(node->children[octant]);
      }
#pragma omp taskwait
      for (int octant=0; octant<8; octant++) {
        if (node->children[octant]) {
          Node<T>* child = node->children[octant];
          std::vector<T> buffer(nsurf);
          int level = node->level;
          gemv(nsurf, nsurf, &(matrix_M2M[level][octant][0]), &child->up_equiv[0], &buffer[0]);
          for (int k=0; k<nsurf; k++) {
            node->up_equiv[k] += buffer[k];
          }
        }
      }
    }
  
    //! L2L operator
    void L2L(Node<T>* node) {
      int nsurf = this->nsurf;
      if (node->is_leaf) return;
      for (int octant=0; octant<8; octant++) {
        if (node->children[octant]) {
          Node<T>* child = node->children[octant];
          std::vector<T> buffer(nsurf);
          int level = node->level;
          gemv(nsurf, nsurf, &(matrix_L2L[level][octant][0]), &node->dn_equiv[0], &buffer[0]);
          for (int k=0; k<nsurf; k++)
            child->dn_equiv[k] += buffer[k];
        }
      }
      for (int octant=0; octant<8; octant++) {
        if (node->children[octant])
#pragma omp task untied
          L2L(node->children[octant]);
      }
#pragma omp taskwait
    }

    void M2L(Nodes<T>& nodes) {}
  };
}  // end namespace
#endif
