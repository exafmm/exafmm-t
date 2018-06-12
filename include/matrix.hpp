#ifndef _PVFMM_MATRIX_HPP_
#define _PVFMM_MATRIX_HPP_
#include "profile.hpp"

namespace pvfmm {
template <class T>
class Permutation {
 public:
  std::vector<size_t> perm;
  std::vector<T> scal;

  Permutation() {}

  Permutation(size_t size) {
    perm.resize(size);
    scal.resize(size);
    std::iota(perm.begin(), perm.end(), 0.);
    std::fill(scal.begin(), scal.end(), 1.);
  }

  size_t Dim() const {
    return perm.size();
  }

  Permutation<T> Transpose() {
    size_t size=perm.size();
    Permutation<T> P_r(size);
    std::vector<size_t>& perm_r=P_r.perm;
    std::vector<T>& scal_r=P_r.scal;
    for(size_t i=0; i<size; i++) {
      perm_r[perm[i]]=i;
      scal_r[perm[i]]=scal[i];
    }
    return P_r;
  }

  Permutation<T> operator*(const Permutation<T>& P) {
    size_t size=perm.size();
    assert(P.Dim()==size);
    Permutation<T> P_r(size);
    std::vector<size_t>& perm_r=P_r.perm;
    std::vector<T>& scal_r=P_r.scal;
    for(size_t i=0; i<size; i++) {
      perm_r[i]=perm[P.perm[i]];
      scal_r[i]=scal[P.perm[i]]*P.scal[i];
    }
    return P_r;
  }
};
}//end namespace
#endif //_PVFMM_MATRIX_HPP_
