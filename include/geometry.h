#ifndef geometry_h
#define geometry_h
#include "pvfmm.h"

namespace pvfmm {
  // alpha is the ratio r_surface/r_cell, see 2015Malhotra, page 7 
  // 2.95 for upward check surface / downward equivalent surface
  // 1.05 for upward equivalent surface / downward check surface
  std::vector<real_t> surface(int p, real_t* c, real_t alpha, int depth){
    size_t n_=(6*(p-1)*(p-1)+2);
    std::vector<real_t> coord(n_*3);
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
    real_t r = 0.5*powf(0.5,depth);
    real_t b = alpha*r;
    for(size_t i=0;i<n_;i++){
      coord[i*3+0]=(coord[i*3+0]+1.0)*b+c[0];
      coord[i*3+1]=(coord[i*3+1]+1.0)*b+c[1];
      coord[i*3+2]=(coord[i*3+2]+1.0)*b+c[2];
    }
    return coord;
  }

  std::vector<real_t> u_check_surf(int p, real_t* c, int depth){
    real_t r=0.5*powf(0.5,depth);
    real_t coord[3]={(real_t)(c[0]-r*1.95),(real_t)(c[1]-r*1.95),(real_t)(c[2]-r*1.95)};
    return surface(p,coord,2.95,depth);
  }

  std::vector<real_t> u_equiv_surf(int p, real_t* c, int depth){
    real_t r=0.5*powf(0.5,depth);
    real_t coord[3]={(real_t)(c[0]-r*0.05),(real_t)(c[1]-r*0.05),(real_t)(c[2]-r*0.05)};
    return surface(p,coord,1.05,depth);
  }

  std::vector<real_t> d_check_surf(int p, real_t* c, int depth){
    real_t r=0.5*powf(0.5,depth);
    real_t coord[3]={(real_t)(c[0]-r*0.05),(real_t)(c[1]-r*0.05),(real_t)(c[2]-r*0.05)};
    return surface(p,coord,1.05,depth);
  }

  std::vector<real_t> d_equiv_surf(int p, real_t* c, int depth){
    real_t r=0.5*powf(0.5,depth);
    real_t coord[3]={(real_t)(c[0]-r*1.95),(real_t)(c[1]-r*1.95),(real_t)(c[2]-r*1.95)};
    return surface(p,coord,2.95,depth);
  }

  std::vector<real_t> conv_grid(int p, real_t* c, int depth){
    real_t r=powf(0.5,depth);
    real_t a=r*1.05;
    real_t coord[3]={c[0],c[1],c[2]};
    int n1=p*2;
    int n2=n1*n1;
    int n3=n1*n1*n1;
    std::vector<real_t> grid(n3*3);
    for(int i=0;i<n1;i++)
    for(int j=0;j<n1;j++)
    for(int k=0;k<n1;k++){
      grid[(i+n1*j+n2*k)*3+0]=(i-p)*a/(p-1)+coord[0];
      grid[(i+n1*j+n2*k)*3+1]=(j-p)*a/(p-1)+coord[1];
      grid[(i+n1*j+n2*k)*3+2]=(k-p)*a/(p-1)+coord[2];
    }
    return grid;
  }
} // end namespace
#endif
