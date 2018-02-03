#ifndef geometry_h
#define geometry_h
#include "pvfmm.h"
#include "matrix.hpp"
#include "vector.hpp"

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

  Permutation<real_t> equiv_surf_perm(size_t m, size_t p_indx, const Permutation<real_t>& ker_perm, const Vector<real_t>* scal_exp=NULL){
    real_t eps=1e-10;
    int dof=ker_perm.Dim();

    real_t c[3]={-0.5,-0.5,-0.5};
    std::vector<real_t> trg_coord=d_check_surf(m,c,0);
    int n_trg=trg_coord.size()/3;

    Permutation<real_t> P=Permutation<real_t>(n_trg*dof);
    if(p_indx==ReflecX || p_indx==ReflecY || p_indx==ReflecZ) {
      for(int i=0;i<n_trg;i++)
      for(int j=0;j<n_trg;j++){
        if(fabs(trg_coord[i*3+0]-trg_coord[j*3+0]*(p_indx==ReflecX?-1.0:1.0))<eps)
        if(fabs(trg_coord[i*3+1]-trg_coord[j*3+1]*(p_indx==ReflecY?-1.0:1.0))<eps)
        if(fabs(trg_coord[i*3+2]-trg_coord[j*3+2]*(p_indx==ReflecZ?-1.0:1.0))<eps){
          for(int k=0;k<dof;k++){
            P.perm[j*dof+k]=i*dof+ker_perm.perm[k];
          }
        }
      }
    }else if(p_indx==SwapXY || p_indx==SwapXZ){
      for(int i=0;i<n_trg;i++)
      for(int j=0;j<n_trg;j++){
        if(fabs(trg_coord[i*3+0]-trg_coord[j*3+(p_indx==SwapXY?1:2)])<eps)
        if(fabs(trg_coord[i*3+1]-trg_coord[j*3+(p_indx==SwapXY?0:1)])<eps)
        if(fabs(trg_coord[i*3+2]-trg_coord[j*3+(p_indx==SwapXY?2:0)])<eps){
          for(int k=0;k<dof;k++){
            P.perm[j*dof+k]=i*dof+ker_perm.perm[k];
          }
        }
      }
    }else{
      for(int j=0;j<n_trg;j++){
        for(int k=0;k<dof;k++){
          P.perm[j*dof+k]=j*dof+ker_perm.perm[k];
        }
      }
    }

    if(scal_exp && p_indx==Scaling) {
      assert(dof==scal_exp->Dim());
      Vector<real_t> scal(scal_exp->Dim());
      for(size_t i=0;i<scal.Dim();i++){
        scal[i]=powf(2.0,(*scal_exp)[i]);
      }
      for(int j=0;j<n_trg;j++){
        for(int i=0;i<dof;i++){
          P.scal[j*dof+i]*=scal[i];
        }
      }
    }
    {
      for(int j=0;j<n_trg;j++){
        for(int i=0;i<dof;i++){
          P.scal[j*dof+i]*=ker_perm.scal[i];
        }
      }
    }
    return P;
  }
} // end namespace
#endif