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

    void P2M(NodePtrs<T>& leafs) {}

    void L2P(NodePtrs<T>& leafs) {}

    void M2M(Node<T>* node) {}
    void L2L(Node<T>* node) {}
    void M2L(Nodes<T>& nodes) {}
  };
}  // end namespace
#endif
