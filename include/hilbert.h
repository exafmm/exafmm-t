#ifndef hilbert_h
#define hilbert_h
#include <cstdlib>
#include <stdint.h>
#include "exafmm_t.h"
#define EXAFMM_HILBERT 0 //! Set this to 1 for Hilbert

namespace exafmm_t {
  //! Levelwise offset of Hilbert key
  inline uint64_t levelOffset(int level) {
    return (((uint64_t)1 << 3 * level) - 1) / 7;
  }

  //! Get level from Hilbert key
  int getLevel(uint64_t i) {
    int level = -1;
    uint64_t offset = 0;
    while (i >= offset) {
      level++;
      offset += (uint64_t)1 << 3 * level;
    }
    return level;
  }

  //! Get parent's Hilbert key
  uint64_t getParent(uint64_t i) {
    int level = getLevel(i);
    return (i - levelOffset(level)) / 8 + levelOffset(level-1);
  }

  //! Get first child's Hilbert key
  uint64_t getChild(uint64_t i) {
    int level = getLevel(i);
    return (i - levelOffset(level)) * 8 + levelOffset(level+1);
  }

  //! Determine which octant the key belongs to
  int getOctant(uint64_t key, bool offset=true) {
    int level = getLevel(key);
    if (offset) key -= levelOffset(level);
    return key & 7;
  }

  //! Get Hilbert key from 3-D index
  uint64_t getKey(ivec3 iX, int level, bool offset=true) {
#if EXAFMM_HILBERT
    int M = 1 << (level - 1);
    for (int Q=M; Q>1; Q>>=1) {
      int R = Q - 1;
      for (int d=0; d<3; d++) {
        if (iX[d] & Q) iX[0] ^= R;
        else {
          int t = (iX[0] ^ iX[d]) & R;
          iX[0] ^= t;
          iX[d] ^= t;
        }
      }
    }
    for (int d=1; d<3; d++) iX[d] ^= iX[d-1];
    int t = 0;
    for (int Q=M; Q>1; Q>>=1)
      if (iX[2] & Q) t ^= Q - 1;
    for (int d=0; d<3; d++) iX[d] ^= t;
#endif
    uint64_t i = 0;
    for (int l=0; l<level; l++) {
      i |= (iX[2] & (uint64_t)1 << l) << 2*l;
      i |= (iX[1] & (uint64_t)1 << l) << (2*l + 1);
      i |= (iX[0] & (uint64_t)1 << l) << (2*l + 2);
    }
    if (offset) i += levelOffset(level);
    return i;
  }

  //! Get 3-D index from Hilbert key
  ivec3 get3DIndex(uint64_t i) {
    int level = getLevel(i);
    i -= levelOffset(level);
    ivec3 iX = 0;
    for (int l=0; l<level; l++) {
      iX[2] |= (i & (uint64_t)1 << 3*l) >> 2*l;
      iX[1] |= (i & (uint64_t)1 << (3*l + 1)) >> (2*l + 1);
      iX[0] |= (i & (uint64_t)1 << (3*l + 2)) >> (2*l + 2);
    }
#if EXAFMM_HILBERT
    int N = 2 << (level - 1);
    int t = iX[2] >> 1;
    for (int d=2; d>0; d--) iX[d] ^= iX[d-1];
    iX[0] ^= t;
    for (int Q=2; Q!=N; Q<<=1) {
      int R = Q - 1;
      for (int d=2; d>=0; d--) {
        if (iX[d] & Q) iX[0] ^= R;
        else {
          t = (iX[0] ^ iX[d]) & R;
          iX[0] ^= t;
          iX[d] ^= t;
        }
      }
    }
#endif
    return iX;
  }

  //! Get 3-D index from Hilbert key without level offset
  ivec3 get3DIndex(uint64_t i, int level) {
    ivec3 iX = 0;
    for (int l=0; l<level; l++) {
      iX[2] |= (i & (uint64_t)1 << 3*l) >> 2*l;
      iX[1] |= (i & (uint64_t)1 << (3*l + 1)) >> (2*l + 1);
      iX[0] |= (i & (uint64_t)1 << (3*l + 2)) >> (2*l + 2);
    }
#if EXAFMM_HILBERT
    int N = 2 << (level - 1);
    int t = iX[2] >> 1;
    for (int d=2; d>0; d--) iX[d] ^= iX[d-1];
    iX[0] ^= t;
    for (int Q=2; Q!=N; Q<<=1) {
      int R = Q - 1;
      for (int d=2; d>=0; d--) {
        if (iX[d] & Q) iX[0] ^= R;
        else {
          t = (iX[0] ^ iX[d]) & R;
          iX[0] ^= t;
          iX[d] ^= t;
        }
      }
    }
#endif
    return iX;
  }

  //! Get 3-D index from coordinates
  ivec3 get3DIndex(vec3 X, int level) {
    real_t dx = 2 * R0 / (1 << level);
    ivec3 iX;
    for (int d=0; d<3; d++) {
      iX[d] = floor((X[d] - Xmin0[d]) / dx);
    }
    return iX;
  }

  //! Get coordinates from 3-D index
  vec3 getCoordinates(ivec3 iX, int level) {
    real_t dx = 2 * R0 / (1 << level);
    vec3 X;
    for (int d=0; d<3; d++) {
      X[d] = (iX[d] + 0.5) * dx + Xmin0[d];
    }
    return X;
  }
}
#endif
