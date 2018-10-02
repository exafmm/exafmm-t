#ifndef dataset_h
#define dataset_h
#include "exafmm_t.h"

namespace exafmm_t {
  Bodies cube(int numBodies, int seed) {
    Bodies bodies(numBodies);
    srand48(seed);
    for (int b=0; b<numBodies; b++) {
      for (int d=0; d<3; d++) {
        bodies[b].X[d] = drand48();
      }
#if COMPLEX
      bodies[b].q = complex_t(drand48()-0.5, drand48()-0.5);
#else
      bodies[b].q = drand48() - 0.5;
#endif
    }
    return bodies;
  }

  // generate plummer distribution in 0 to 1 cube
  Bodies plummer(int numBodies, int seed) {
    Bodies bodies(numBodies);
    srand48(seed);
    int i = 0;
    int Xmax = 0;
    while (i < numBodies) {
      real_t X1 = drand48();
      real_t X2 = drand48();
      real_t X3 = drand48();
      real_t R = 1.0 / sqrt( (pow(X1, -2.0 / 3.0) - 1.0) );
      if (R < 100) {
        real_t Z = (1.0 - 2.0 * X2) * R;
        real_t X = sqrt(R * R - Z * Z) * std::cos(2.0 * M_PI * X3);
        real_t Y = sqrt(R * R - Z * Z) * std::sin(2.0 * M_PI * X3);
        bodies[i].X[0] = X;
        bodies[i].X[1] = Y;
        bodies[i].X[2] = Z;
#if COMPLEX
        bodies[i].q = complex_t(drand48()-0.5, drand48()-0.5);
#else
        bodies[i].q = drand48() - 0.5;
#endif
        for (int d=0; d<3; d++) {
          Xmax = Xmax > fabs(bodies[i].X[d]) ? Xmax : fabs(bodies[i].X[d]);
        }
        i++;
      }
    }
    real_t scale = 0.5 / (Xmax + 1);
    for (int i=0; i<numBodies; i++) {
      for (int d=0; d<3; d++) {
        bodies[i].X[d] = bodies[i].X[d]*scale + 0.5;
      }
    }
    return bodies;
  }

  Bodies nonuniform(int numBodies, int seed) {
    srand48(seed);
    Bodies bodies(numBodies);
    for (int i=0; i<numBodies; i++) {
      if (i < 0.1*numBodies) {
        for (int d=0; d<3; d++) {
          bodies[i].X[d] = drand48()*0.5;
        }
      }
      else if (i < 0.2*numBodies) {
        for (int d=0; d<3; d++) {
          bodies[i].X[d] = 0.5 + drand48()*0.5;
        }
      } else {
        for (int d=0; d<3; d++) {
          bodies[i].X[d] = drand48();
        }
      }
#if COMPLEX
      bodies[i].q = complex_t(drand48()-0.5, drand48()-0.5);
#else
      bodies[i].q = drand48() - 0.5;
#endif
    }
    return bodies;
  }

  Bodies initBodies(int numBodies, const char * distribution, int seed) {
    Bodies bodies;
    switch (distribution[0]) {
      case 'c':
        bodies = cube(numBodies, seed);
        break;
      case 'p':
        bodies = plummer(numBodies, seed);
        break;
      default:
        fprintf(stderr, "Unknown data distribution %s\n", distribution);
    }
    return bodies;
  }
}
#endif
