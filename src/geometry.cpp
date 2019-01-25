#include "geometry.h"

namespace exafmm_t {
  std::vector<std::vector<ivec3>> REL_COORD;
  std::vector<std::vector<int>> hash_lut;     // coord_hash -> index in rel_coord

  // alpha is the ratio r_surface/r_node
  // 2.95 for upward check surface / downward equivalent surface
  // 1.05 for upward equivalent surface / downward check surface
  RealVec surface(int p, real_t* c, real_t alpha, int level, bool is_mapping){
    size_t n = 6*(p-1)*(p-1) + 2;
    RealVec coord(n*3);
    coord[0] = -1.0;
    coord[1] = -1.0;
    coord[2] = -1.0;
    size_t count = 1;
    for(int i=0; i<p-1; i++) {
      for(int j=0; j<p-1; j++) {
        coord[count*3  ] = -1.0;
        coord[count*3+1] = (2.0*(i+1)-p+1) / (p-1);
        coord[count*3+2] = (2.0*j-p+1) / (p-1);
        count++;
      }
    }
    for(int i=0; i<p-1; i++) {
      for(int j=0; j<p-1; j++) {
        coord[count*3  ] = (2.0*i-p+1) / (p-1);
        coord[count*3+1] = -1.0;
        coord[count*3+2] = (2.0*(j+1)-p+1) / (p-1);
        count++;
      }
    }
    for(int i=0; i<p-1; i++) {
      for(int j=0; j<p-1; j++) {
        coord[count*3  ] = (2.0*(i+1)-p+1) / (p-1);
        coord[count*3+1] = (2.0*j-p+1) / (p-1);
        coord[count*3+2] = -1.0;
        count++;
      }
    }
    for(size_t i=0; i<(n/2)*3; i++) {
      coord[count*3+i] = -coord[i];
    }
    real_t r = is_mapping ? 0.5 : R0;
    r *= powf(0.5, level);
    real_t b = alpha * r;
    real_t br = b - r;
    for(size_t i=0; i<n; i++){
      coord[i*3+0] = (coord[i*3+0]+1.0)*b + c[0]- br;
      coord[i*3+1] = (coord[i*3+1]+1.0)*b + c[1]- br;
      coord[i*3+2] = (coord[i*3+2]+1.0)*b + c[2]- br;
    }
    return coord;
  }

  RealVec convolution_grid(real_t* c, int level){
    real_t r = R0 * powf(0.5, level-1);
    real_t a = r * 1.05;
    real_t coord[3] = {c[0], c[1], c[2]};
    int n1 = P * 2;
    int n2 = n1 * n1;
    int n3 = n1 * n1 * n1;
    RealVec grid(n3*3);
    for(int i=0; i<n1; i++) {
      for(int j=0; j<n1; j++) {
        for(int k=0; k<n1; k++) {
          grid[(i+n1*j+n2*k)*3+0] = (i-P)*a/(P-1) + coord[0];
          grid[(i+n1*j+n2*k)*3+1] = (j-P)*a/(P-1) + coord[1];
          grid[(i+n1*j+n2*k)*3+2] = (k-P)*a/(P-1) + coord[2];
        }
      }
    }
    return grid;
  }

  //! return x + 10y + 100z + 555
  int hash(ivec3& coord) {
    const int n = 5;
    return ((coord[2]+n) * (2*n) + (coord[1]+n)) *(2*n) + (coord[0]+n);
  }

  void init_rel_coord(int max_r, int min_r, int step, Mat_Type t) {
    const int max_hash = 2000;
    int n1 = (max_r*2)/step+1;
    int n2 = (min_r*2)/step-1;
    int count = n1*n1*n1 - (min_r>0?n2*n2*n2:0);
    hash_lut[t].resize(max_hash, -1);
    for(int k=-max_r; k<=max_r; k+=step) {
      for(int j=-max_r; j<=max_r; j+=step) {
        for(int i=-max_r; i<=max_r; i+=step) {
          if(abs(i)>=min_r || abs(j)>=min_r || abs(k)>=min_r) {
            ivec3 coord;
            coord[0] = i;
            coord[1] = j;
            coord[2] = k;
            REL_COORD[t].push_back(coord);
            hash_lut[t][hash(coord)] = REL_COORD[t].size() - 1;
          }
        }
      }
    }
  }

  void init_rel_coord() {
    REL_COORD.resize(Type_Count);
    hash_lut.resize(Type_Count);
    init_rel_coord(1, 1, 2, M2M_Type);
    init_rel_coord(1, 1, 2, L2L_Type);
    init_rel_coord(3, 3, 2, P2P0_Type);
    init_rel_coord(1, 0, 1, P2P1_Type);
    init_rel_coord(3, 3, 2, P2P2_Type);
    init_rel_coord(3, 2, 1, M2L_Helper_Type);
    init_rel_coord(1, 1, 1, M2L_Type);
    init_rel_coord(5, 5, 2, M2P_Type);
    init_rel_coord(5, 5, 2, P2L_Type);
  }
} // end namespace
