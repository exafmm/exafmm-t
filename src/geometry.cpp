#include "geometry.h"

namespace exafmm_t {
  // alpha is the ratio r_surface/r_node
  // 2.95 for upward check surface / downward equivalent surface
  // 1.05 for upward equivalent surface / downward check surface
  RealVec surface(int p, real_t* c, real_t alpha, int level, bool is_mapping){
    size_t n_=(6*(p-1)*(p-1)+2);
    RealVec coord(n_*3);
    coord[0]=coord[1]=coord[2]=-1.0;
    size_t cnt=1;
    for(int i=0;i<p-1;i++)
      for(int j=0;j<p-1;j++){
        coord[cnt*3  ]=-1.0;
        coord[cnt*3+1]=(2.0*(i+1)-p+1)/(p-1);
        coord[cnt*3+2]=(2.0*j-p+1)/(p-1);
        cnt++;
      }
    for(int i=0;i<p-1;i++)
      for(int j=0;j<p-1;j++){
        coord[cnt*3  ]=(2.0*i-p+1)/(p-1);
        coord[cnt*3+1]=-1.0;
        coord[cnt*3+2]=(2.0*(j+1)-p+1)/(p-1);
        cnt++;
      }
    for(int i=0;i<p-1;i++)
      for(int j=0;j<p-1;j++){
        coord[cnt*3  ]=(2.0*(i+1)-p+1)/(p-1);
        coord[cnt*3+1]=(2.0*j-p+1)/(p-1);
        coord[cnt*3+2]=-1.0;
        cnt++;
      }
    for(size_t i=0;i<(n_/2)*3;i++) coord[cnt*3+i]=-coord[i];
    real_t r = is_mapping ? 0.5 : R0;
    r *= powf(0.5, level);
    real_t b = alpha*r;
    real_t br = b-r;
    for(size_t i=0;i<n_;i++){
      coord[i*3+0]=(coord[i*3+0]+1.0)*b+c[0]-br;
      coord[i*3+1]=(coord[i*3+1]+1.0)*b+c[1]-br;
      coord[i*3+2]=(coord[i*3+2]+1.0)*b+c[2]-br;
    }
    return coord;
  }

  RealVec convolution_grid(real_t* c, int level){
    real_t r = R0*powf(0.5,level-1);
    real_t a=r*1.05;
    real_t coord[3]={c[0],c[1],c[2]};
    int n1=MULTIPOLE_ORDER*2;
    int n2=n1*n1;
    int n3=n1*n1*n1;
    RealVec grid(n3*3);
    for(int i=0;i<n1;i++)
    for(int j=0;j<n1;j++)
    for(int k=0;k<n1;k++){
      grid[(i+n1*j+n2*k)*3+0]=(i-MULTIPOLE_ORDER)*a/(MULTIPOLE_ORDER-1)+coord[0];
      grid[(i+n1*j+n2*k)*3+1]=(j-MULTIPOLE_ORDER)*a/(MULTIPOLE_ORDER-1)+coord[1];
      grid[(i+n1*j+n2*k)*3+2]=(k-MULTIPOLE_ORDER)*a/(MULTIPOLE_ORDER-1)+coord[2];
    }
    return grid;
  }
} // end namespace
