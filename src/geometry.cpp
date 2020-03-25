#include "geometry.h"

namespace exafmm_t {
  std::vector<std::vector<ivec3>> REL_COORD;
  std::vector<std::vector<int>> HASH_LUT;       // coord_hash -> index in rel_coord
  std::vector<std::vector<int>> M2L_INDEX_MAP;  // [M2L_relpos_idx][octant] -> M2L_Helper_relpos_idx


  RealVec surface(int p, real_t r0, int level, real_t* c, real_t alpha) {
    int n = 6*(p-1)*(p-1) + 2;
    RealVec coord(n*3);
    coord[0] = -1.0;
    coord[1] = -1.0;
    coord[2] = -1.0;
    int count = 1;
    for (int i=0; i<p-1; i++) {
      for (int j=0; j<p-1; j++) {
        coord[count*3  ] = -1.0;
        coord[count*3+1] = (2.0*(i+1)-p+1) / (p-1);
        coord[count*3+2] = (2.0*j-p+1) / (p-1);
        count++;
      }
    }
    for (int i=0; i<p-1; i++) {
      for (int j=0; j<p-1; j++) {
        coord[count*3  ] = (2.0*i-p+1) / (p-1);
        coord[count*3+1] = -1.0;
        coord[count*3+2] = (2.0*(j+1)-p+1) / (p-1);
        count++;
      }
    }
    for (int i=0; i<p-1; i++) {
      for (int j=0; j<p-1; j++) {
        coord[count*3  ] = (2.0*(i+1)-p+1) / (p-1);
        coord[count*3+1] = (2.0*j-p+1) / (p-1);
        coord[count*3+2] = -1.0;
        count++;
      }
    }
    for (int i=0; i<(n/2)*3; i++) {
      coord[count*3+i] = -coord[i];
    }
    real_t r = r0 * powf(0.5, level);
    real_t b = alpha * r;
    for (int i=0; i<n; i++){
      coord[i*3+0] = coord[i*3+0]*b + c[0];
      coord[i*3+1] = coord[i*3+1]*b + c[1];
      coord[i*3+2] = coord[i*3+2]*b + c[2];
    }
    return coord;
  }

  RealVec convolution_grid(int p, real_t r0, int level, real_t* c) {
    real_t d = 2 * r0 * powf(0.5, level);
    real_t a = d * 1.05;  // side length of upward equivalent/downward check box
    int n1 = p * 2;
    int n2 = n1 * n1;
    int n3 = n1 * n1 * n1;
    RealVec grid(n3*3);
    for (int i=0; i<n1; i++) {
      for (int j=0; j<n1; j++) {
        for (int k=0; k<n1; k++) {
          grid[(i+n1*j+n2*k)*3+0] = (i-p)*a/(p-1) + c[0];
          grid[(i+n1*j+n2*k)*3+1] = (j-p)*a/(p-1) + c[1];
          grid[(i+n1*j+n2*k)*3+2] = (k-p)*a/(p-1) + c[2];
        }
      }
    }
    return grid;
  }

  std::vector<int> generate_surf2conv_up(int p) {
    int n1 = 2*p;
    real_t c[3];
    for (int d=0; d<3; d++) c[d] = 0.5*(p-1);
    RealVec surf = surface(p, 0.5, 0, c, real_t(p-1));
    std::vector<int> map(6*(p-1)*(p-1)+2);
    for (size_t i=0; i<map.size(); i++) {
      map[i] = (int)(p-1-surf[i*3])
             + ((int)(p-1-surf[i*3+1])) * n1
             + ((int)(p-1-surf[i*3+2])) * n1 * n1;
    }
    return map;
  }

  std::vector<int> generate_surf2conv_dn(int p) {
    int n1 = 2*p;
    real_t c[3];
    for (int d=0; d<3; d++) c[d] = 0.5*(p-1);
    RealVec surf = surface(p, 0.5, 0, c, real_t(p-1));
    std::vector<int> map(6*(p-1)*(p-1)+2);
    for (size_t i=0; i<map.size(); i++) {
      map[i] = (int)(2*p-1-surf[i*3])
             + ((int)(2*p-1-surf[i*3+1])) * n1
             + ((int)(2*p-1-surf[i*3+2])) * n1 * n1;
    }
    return map;
  }

  // return x + 10y + 100z + 555
  int hash(ivec3& coord) {
    const int n = 5;
    return ((coord[2]+n) * (2*n) + (coord[1]+n)) *(2*n) + (coord[0]+n);
  }

  void init_rel_coord(int max_r, int min_r, int step, Mat_Type t) {
    const int max_hash = 2000;
    HASH_LUT[t].resize(max_hash, -1);
    for (int k=-max_r; k<=max_r; k+=step) {
      for (int j=-max_r; j<=max_r; j+=step) {
        for (int i=-max_r; i<=max_r; i+=step) {
          if (abs(i)>=min_r || abs(j)>=min_r || abs(k)>=min_r) {
            ivec3 coord;
            coord[0] = i;
            coord[1] = j;
            coord[2] = k;
            REL_COORD[t].push_back(coord);
            HASH_LUT[t][hash(coord)] = REL_COORD[t].size() - 1;
          }
        }
      }
    }
  }

  void generate_M2L_index_map() {
    int npos = REL_COORD[M2L_Type].size();   // number of relative coords for M2L_Type
    M2L_INDEX_MAP.resize(npos, std::vector<int>(NCHILD*NCHILD));
#pragma omp parallel for
    for (int i=0; i<npos; ++i) {
      for (int j1=0; j1<NCHILD; ++j1) {
        for (int j2=0; j2<NCHILD; ++j2) {
          ivec3& parent_rel_coord = REL_COORD[M2L_Type][i];
          ivec3  child_rel_coord;
          child_rel_coord[0] = parent_rel_coord[0]*2 - (j1/1)%2 + (j2/1)%2;
          child_rel_coord[1] = parent_rel_coord[1]*2 - (j1/2)%2 + (j2/2)%2;
          child_rel_coord[2] = parent_rel_coord[2]*2 - (j1/4)%2 + (j2/4)%2;
          int coord_hash = hash(child_rel_coord);
          int child_rel_idx = HASH_LUT[M2L_Helper_Type][coord_hash];
          int j = j2*NCHILD + j1;
          M2L_INDEX_MAP[i][j] = child_rel_idx;
        }
      }
    }
  }

  void init_rel_coord() {
    REL_COORD.resize(Type_Count);
    HASH_LUT.resize(Type_Count);
    init_rel_coord(1, 1, 2, M2M_Type);
    init_rel_coord(1, 1, 2, L2L_Type);
    init_rel_coord(3, 3, 2, P2P0_Type);
    init_rel_coord(1, 0, 1, P2P1_Type);
    init_rel_coord(3, 3, 2, P2P2_Type);
    init_rel_coord(3, 2, 1, M2L_Helper_Type);
    init_rel_coord(1, 1, 1, M2L_Type);
    init_rel_coord(5, 5, 2, M2P_Type);
    init_rel_coord(5, 5, 2, P2L_Type);
    generate_M2L_index_map();
  }
} // end namespace
