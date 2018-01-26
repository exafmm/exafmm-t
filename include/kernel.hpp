#ifndef _PVFMM_FMM_KERNEL_HPP_
#define _PVFMM_FMM_KERNEL_HPP_

namespace pvfmm{

#ifdef __AVX__
  inline __m256 zero_intrin(const float){
    return _mm256_setzero_ps();
  }

  inline __m256d zero_intrin(const double){
    return _mm256_setzero_pd();
  }

  inline __m256 set_intrin(const float& a){
    return _mm256_set_ps(a,a,a,a,a,a,a,a);
  }

  inline __m256d set_intrin(const double& a){
    return _mm256_set_pd(a,a,a,a);
  }

  inline __m256 load_intrin(const float* a){
    return _mm256_load_ps(a);
  }

  inline __m256d load_intrin(const double* a){
    return _mm256_load_pd(a);
  }

  inline void store_intrin(float* a, const __m256& b){
    return _mm256_store_ps(a,b);
  }

  inline void store_intrin(double* a, const __m256d& b){
    return _mm256_store_pd(a,b);
  }

  inline __m256 mul_intrin(const __m256& a, const __m256& b){
    return _mm256_mul_ps(a,b);
  }

  inline __m256d mul_intrin(const __m256d& a, const __m256d& b){
    return _mm256_mul_pd(a,b);
  }

  inline __m256 add_intrin(const __m256& a, const __m256& b){
    return _mm256_add_ps(a,b);
  }

  inline __m256d add_intrin(const __m256d& a, const __m256d& b){
    return _mm256_add_pd(a,b);
  }

  inline __m256 sub_intrin(const __m256& a, const __m256& b){
    return _mm256_sub_ps(a,b);
  }

  inline __m256d sub_intrin(const __m256d& a, const __m256d& b){
    return _mm256_sub_pd(a,b);
  }

  inline __m128 rsqrt_approx_intrin(const __m128& r2){
    return _mm_andnot_ps(_mm_cmpeq_ps(r2,_mm_setzero_ps()),_mm_rsqrt_ps(r2));
  }

  inline __m256 rsqrt_approx_intrin(const __m256& r2){
    return _mm256_andnot_ps(_mm256_cmp_ps(r2,_mm256_setzero_ps(),_CMP_EQ_OS),_mm256_rsqrt_ps(r2));
  }

  inline __m256d rsqrt_approx_intrin(const __m256d& r2){
    return _mm256_cvtps_pd(rsqrt_approx_intrin(_mm256_cvtpd_ps(r2)));
  }

  inline void rsqrt_newton_intrin(__m256& rinv, const __m256& r2, const float& nwtn_const){
    rinv=mul_intrin(rinv,sub_intrin(set_intrin(nwtn_const),mul_intrin(r2,mul_intrin(rinv,rinv))));
  }

  inline void rsqrt_newton_intrin(__m256d& rinv, const __m256d& r2, const double& nwtn_const){
    rinv=mul_intrin(rinv,sub_intrin(set_intrin(nwtn_const),mul_intrin(r2,mul_intrin(rinv,rinv))));
  }

#else
#ifdef __SSE3__
  inline __m128 zero_intrin(const float){
    return _mm_setzero_ps();
  }

  inline __m128d zero_intrin(const double){
    return _mm_setzero_pd();
  }

  inline __m128 set_intrin(const float& a){
    return _mm_set1_ps(a);
  }

  inline __m128d set_intrin(const double& a){
    return _mm_set1_pd(a);
  }

  inline __m128 load_intrin(const float* a){
    return _mm_load_ps(a);
  }

  inline __m128d load_intrin(const double* a){
    return _mm_load_pd(a);
  }

  inline void store_intrin(float* a, const __m128& b){
    return _mm_store_ps(a,b);
  }

  inline void store_intrin(double* a, const __m128d& b){
    return _mm_store_pd(a,b);
  }

  inline __m128 mul_intrin(const __m128& a, const __m128& b){
    return _mm_mul_ps(a,b);
  }

  inline __m128d mul_intrin(const __m128d& a, const __m128d& b){
    return _mm_mul_pd(a,b);
  }

  inline __m128 add_intrin(const __m128& a, const __m128& b){
    return _mm_add_ps(a,b);
  }

  inline __m128d add_intrin(const __m128d& a, const __m128d& b){
    return _mm_add_pd(a,b);
  }

  inline __m128 sub_intrin(const __m128& a, const __m128& b){
    return _mm_sub_ps(a,b);
  }

  inline __m128d sub_intrin(const __m128d& a, const __m128d& b){
    return _mm_sub_pd(a,b);
  }

  inline __m128 rsqrt_approx_intrin(const __m128& r2){
    return _mm_andnot_ps(_mm_cmpeq_ps(r2,_mm_setzero_ps()),_mm_rsqrt_ps(r2));
  }

  inline __m128d rsqrt_approx_intrin(const __m128d& r2){
    return _mm_cvtps_pd(rsqrt_approx_intrin(_mm_cvtpd_ps(r2)));
  }

  inline void rsqrt_newton_intrin(__m128& rinv, const __m128& r2, const float& nwtn_const){
    rinv=mul_intrin(rinv,sub_intrin(set_intrin(nwtn_const),mul_intrin(r2,mul_intrin(rinv,rinv))));
  }

  inline void rsqrt_newton_intrin(__m128d& rinv, const __m128d& r2, const double& nwtn_const){
    rinv=mul_intrin(rinv,sub_intrin(set_intrin(nwtn_const),mul_intrin(r2,mul_intrin(rinv,rinv))));
  }

#endif //__SSE3__
#endif //__AVX__

  inline Vec_t rsqrt_intrin2(Vec_t r2){
    Vec_t rinv=rsqrt_approx_intrin(r2);
    rsqrt_newton_intrin(rinv,r2,Real_t(3));
    rsqrt_newton_intrin(rinv,r2,Real_t(12));
    return rinv;
  }

struct Kernel{
  public:

  typedef void (*Ker_t)(Real_t* r_src, int src_cnt, Real_t* v_src, int dof,
                        Real_t* r_trg, int trg_cnt, Real_t* k_out);

  Kernel(Ker_t poten, const char* name, std::pair<int,int> k_dim) {
    ker_dim[0]=k_dim.first;
    ker_dim[1]=k_dim.second;
    ker_poten=poten;
    ker_name=std::string(name);
    k_s2m=NULL;
    k_s2l=NULL;
    k_s2t=NULL;
    k_m2m=NULL;
    k_m2l=NULL;
    k_m2t=NULL;
    k_l2l=NULL;
    k_l2t=NULL;
    scale_invar=true;
    src_scal.Resize(ker_dim[0]); src_scal.SetZero();
    trg_scal.Resize(ker_dim[1]); trg_scal.SetZero();
    perm_vec.resize(Perm_Count);
    for(size_t p_type=0;p_type<C_Perm;p_type++){
      perm_vec[p_type       ]=Permutation<Real_t>(ker_dim[0]);
      perm_vec[p_type+C_Perm]=Permutation<Real_t>(ker_dim[1]);
    }
    init=false;
  }

  void Initialize(bool verbose=false) const{
    if(init) return;
    init=true;
    Real_t eps=1.0;
    while(eps+(Real_t)1.0>1.0) eps*=0.5;
    Real_t scal=1.0;
    if(ker_dim[0]*ker_dim[1]>0){
      Matrix<Real_t> M_scal(ker_dim[0],ker_dim[1]);
      size_t N=1024;
      Real_t eps_=N*eps;
      Real_t src_coord[3]={0,0,0};
      std::vector<Real_t> trg_coord1(N*3);
      Matrix<Real_t> M1(N,ker_dim[0]*ker_dim[1]);
      while(true){
	Real_t abs_sum=0;
	for(size_t i=0;i<N/2;i++){
	  Real_t x,y,z,r;
	  do{
	    x=(drand48()-0.5);
	    y=(drand48()-0.5);
	    z=(drand48()-0.5);
	    r=sqrtf(x*x+y*y+z*z);
	  }while(r<0.25);
	  trg_coord1[i*3+0]=x*scal;
	  trg_coord1[i*3+1]=y*scal;
	  trg_coord1[i*3+2]=z*scal;
	}
	for(size_t i=N/2;i<N;i++){
	  Real_t x,y,z,r;
	  do{
	    x=(drand48()-0.5);
	    y=(drand48()-0.5);
	    z=(drand48()-0.5);
	    r=sqrtf(x*x+y*y+z*z);
	  }while(r<0.25);
	  trg_coord1[i*3+0]=x*1.0/scal;
	  trg_coord1[i*3+1]=y*1.0/scal;
	  trg_coord1[i*3+2]=z*1.0/scal;
	}
	for(size_t i=0;i<N;i++){
	  BuildMatrix(&src_coord [          0], 1,
		      &trg_coord1[i*3], 1, &(M1[i][0]));
	  for(size_t j=0;j<ker_dim[0]*ker_dim[1];j++){
	    abs_sum+=fabs(M1[i][j]);
	  }
	}
	if(abs_sum>sqrtf(eps) || scal<eps) break;
	scal=scal*0.5;
      }

      std::vector<Real_t> trg_coord2(N*3);
      Matrix<Real_t> M2(N,ker_dim[0]*ker_dim[1]);
      for(size_t i=0;i<N*3;i++){
	trg_coord2[i]=trg_coord1[i]*0.5;
      }
      for(size_t i=0;i<N;i++){
	BuildMatrix(&src_coord [          0], 1,
		    &trg_coord2[i*3], 1, &(M2[i][0]));
      }

      for(size_t i=0;i<ker_dim[0]*ker_dim[1];i++){
	Real_t dot11=0, dot12=0, dot22=0;
	for(size_t j=0;j<N;j++){
	  dot11+=M1[j][i]*M1[j][i];
	  dot12+=M1[j][i]*M2[j][i];
	  dot22+=M2[j][i]*M2[j][i];
	}
	Real_t max_val=std::max<Real_t>(dot11,dot22);
	if(dot11>max_val*eps &&
	   dot22>max_val*eps ){
	  Real_t s=dot12/dot11;
	  M_scal[0][i]=log(s)/log(2.0);
	  Real_t err=sqrtf(0.5*(dot22/dot11)/(s*s)-0.5);
	  if(err>eps_){
	    scale_invar=false;
	    M_scal[0][i]=0.0;
	  }
	}else if(dot11>max_val*eps ||
		 dot22>max_val*eps ){
	  scale_invar=false;
	  M_scal[0][i]=0.0;
	}else{
	  M_scal[0][i]=-1;
	}
      }
      src_scal.Resize(ker_dim[0]); src_scal.SetZero();
      trg_scal.Resize(ker_dim[1]); trg_scal.SetZero();
      if(scale_invar){
	Matrix<Real_t> b(ker_dim[0]*ker_dim[1]+1,1); b.SetZero();
	memcpy(&b[0][0],&M_scal[0][0],ker_dim[0]*ker_dim[1]*sizeof(Real_t));
	Matrix<Real_t> M(ker_dim[0]*ker_dim[1]+1,ker_dim[0]+ker_dim[1]); M.SetZero();
	M[ker_dim[0]*ker_dim[1]][0]=1;
	for(size_t i0=0;i0<ker_dim[0];i0++)
	  for(size_t i1=0;i1<ker_dim[1];i1++){
	    size_t j=i0*ker_dim[1]+i1;
	    if(fabs(b[j][0])>=0){
	      M[j][ 0+        i0]=1;
	      M[j][i1+ker_dim[0]]=1;
	    }
	  }
	Matrix<Real_t> x=M.pinv()*b;
	for(size_t i=0;i<ker_dim[0];i++){
	  src_scal[i]=x[i][0];
	}
	for(size_t i=0;i<ker_dim[1];i++){
	  trg_scal[i]=x[ker_dim[0]+i][0];
	}
	for(size_t i0=0;i0<ker_dim[0];i0++)
	  for(size_t i1=0;i1<ker_dim[1];i1++){
	    if(M_scal[i0][i1]>=0){
	      if(fabs(src_scal[i0]+trg_scal[i1]-M_scal[i0][i1])>eps_){
		scale_invar=false;
	      }
	    }
	  }
      }
      if(!scale_invar){
	src_scal.SetZero();
	trg_scal.SetZero();
      }
    }
    if(ker_dim[0]*ker_dim[1]>0){
      size_t N=1024;
      Real_t eps_=N*eps;
      Real_t src_coord[3]={0,0,0};
      std::vector<Real_t> trg_coord1(N*3);
      std::vector<Real_t> trg_coord2(N*3);
      for(size_t i=0;i<N/2;i++){
	Real_t x,y,z,r;
	do{
	  x=(drand48()-0.5);
	  y=(drand48()-0.5);
	  z=(drand48()-0.5);
	  r=sqrtf(x*x+y*y+z*z);
	}while(r<0.25);
	trg_coord1[i*3+0]=x*scal;
	trg_coord1[i*3+1]=y*scal;
	trg_coord1[i*3+2]=z*scal;
      }
      for(size_t i=N/2;i<N;i++){
	Real_t x,y,z,r;
	do{
	  x=(drand48()-0.5);
	  y=(drand48()-0.5);
	  z=(drand48()-0.5);
	  r=sqrtf(x*x+y*y+z*z);
	}while(r<0.25);
	trg_coord1[i*3+0]=x*1.0/scal;
	trg_coord1[i*3+1]=y*1.0/scal;
	trg_coord1[i*3+2]=z*1.0/scal;
      }
      for(size_t p_type=0;p_type<C_Perm;p_type++){
	switch(p_type){
        case ReflecX:
          for(size_t i=0;i<N;i++){
            trg_coord2[i*3+0]=-trg_coord1[i*3+0];
            trg_coord2[i*3+1]= trg_coord1[i*3+1];
            trg_coord2[i*3+2]= trg_coord1[i*3+2];
          }
          break;
        case ReflecY:
          for(size_t i=0;i<N;i++){
            trg_coord2[i*3+0]= trg_coord1[i*3+0];
            trg_coord2[i*3+1]=-trg_coord1[i*3+1];
            trg_coord2[i*3+2]= trg_coord1[i*3+2];
          }
          break;
        case ReflecZ:
          for(size_t i=0;i<N;i++){
            trg_coord2[i*3+0]= trg_coord1[i*3+0];
            trg_coord2[i*3+1]= trg_coord1[i*3+1];
            trg_coord2[i*3+2]=-trg_coord1[i*3+2];
          }
          break;
        case SwapXY:
          for(size_t i=0;i<N;i++){
            trg_coord2[i*3+0]= trg_coord1[i*3+1];
            trg_coord2[i*3+1]= trg_coord1[i*3+0];
            trg_coord2[i*3+2]= trg_coord1[i*3+2];
          }
          break;
        case SwapXZ:
          for(size_t i=0;i<N;i++){
            trg_coord2[i*3+0]= trg_coord1[i*3+2];
            trg_coord2[i*3+1]= trg_coord1[i*3+1];
            trg_coord2[i*3+2]= trg_coord1[i*3+0];
          }
          break;
        default:
          for(size_t i=0;i<N;i++){
            trg_coord2[i*3+0]= trg_coord1[i*3+0];
            trg_coord2[i*3+1]= trg_coord1[i*3+1];
            trg_coord2[i*3+2]= trg_coord1[i*3+2];
          }
	}
	Matrix<long long> M11, M22;
	{
	  Matrix<Real_t> M1(N,ker_dim[0]*ker_dim[1]); M1.SetZero();
	  Matrix<Real_t> M2(N,ker_dim[0]*ker_dim[1]); M2.SetZero();
	  for(size_t i=0;i<N;i++){
	    BuildMatrix(&src_coord [          0], 1,
			&trg_coord1[i*3], 1, &(M1[i][0]));
	    BuildMatrix(&src_coord [          0], 1,
			&trg_coord2[i*3], 1, &(M2[i][0]));
	  }
	  Matrix<Real_t> dot11(ker_dim[0]*ker_dim[1],ker_dim[0]*ker_dim[1]);dot11.SetZero();
	  Matrix<Real_t> dot12(ker_dim[0]*ker_dim[1],ker_dim[0]*ker_dim[1]);dot12.SetZero();
	  Matrix<Real_t> dot22(ker_dim[0]*ker_dim[1],ker_dim[0]*ker_dim[1]);dot22.SetZero();
	  std::vector<Real_t> norm1(ker_dim[0]*ker_dim[1]);
	  std::vector<Real_t> norm2(ker_dim[0]*ker_dim[1]);
	  {
	    for(size_t k=0;k<N;k++)
	      for(size_t i=0;i<ker_dim[0]*ker_dim[1];i++)
		for(size_t j=0;j<ker_dim[0]*ker_dim[1];j++){
		  dot11[i][j]+=M1[k][i]*M1[k][j];
		  dot12[i][j]+=M1[k][i]*M2[k][j];
		  dot22[i][j]+=M2[k][i]*M2[k][j];
		}
	    for(size_t i=0;i<ker_dim[0]*ker_dim[1];i++){
	      norm1[i]=sqrtf(dot11[i][i]);
	      norm2[i]=sqrtf(dot22[i][i]);
	    }
	    for(size_t i=0;i<ker_dim[0]*ker_dim[1];i++)
	      for(size_t j=0;j<ker_dim[0]*ker_dim[1];j++){
		dot11[i][j]/=(norm1[i]*norm1[j]);
		dot12[i][j]/=(norm1[i]*norm2[j]);
		dot22[i][j]/=(norm2[i]*norm2[j]);
	      }
	  }
	  long long flag=1;
	  M11.Resize(ker_dim[0],ker_dim[1]); M11.SetZero();
	  M22.Resize(ker_dim[0],ker_dim[1]); M22.SetZero();
	  for(size_t i=0;i<ker_dim[0]*ker_dim[1];i++){
	    if(norm1[i]>eps_ && M11[0][i]==0){
	      for(size_t j=0;j<ker_dim[0]*ker_dim[1];j++){
		if(fabs(norm1[i]-norm1[j])<eps_ && fabs(fabs(dot11[i][j])-1.0)<eps_){
		  M11[0][j]=(dot11[i][j]>0?flag:-flag);
		}
		if(fabs(norm1[i]-norm2[j])<eps_ && fabs(fabs(dot12[i][j])-1.0)<eps_){
		  M22[0][j]=(dot12[i][j]>0?flag:-flag);
		}
	      }
	      flag++;
	    }
	  }
	}
	Matrix<long long> P1, P2;
	{
	  Matrix<long long>& P=P1;
	  Matrix<long long>  M1=M11;
	  Matrix<long long>  M2=M22;
	  for(size_t i=0;i<M1.Dim(0);i++){
	    for(size_t j=0;j<M1.Dim(1);j++){
	      if(M1[i][j]<0) M1[i][j]=-M1[i][j];
	      if(M2[i][j]<0) M2[i][j]=-M2[i][j];
	    }
	    std::sort(&M1[i][0],&M1[i][M1.Dim(1)]);
	    std::sort(&M2[i][0],&M2[i][M2.Dim(1)]);
	  }
	  P.Resize(M1.Dim(0),M1.Dim(0));
	  for(size_t i=0;i<M1.Dim(0);i++)
	    for(size_t j=0;j<M1.Dim(0);j++){
	      P[i][j]=1;
	      for(size_t k=0;k<M1.Dim(1);k++)
		if(M1[i][k]!=M2[j][k]){
		  P[i][j]=0;
		  break;
		}
	    }
	}
	{
	  Matrix<long long>& P=P2;
	  Matrix<long long>  M1=M11.Transpose();
	  Matrix<long long>  M2=M22.Transpose();
	  for(size_t i=0;i<M1.Dim(0);i++){
	    for(size_t j=0;j<M1.Dim(1);j++){
	      if(M1[i][j]<0) M1[i][j]=-M1[i][j];
	      if(M2[i][j]<0) M2[i][j]=-M2[i][j];
	    }
	    std::sort(&M1[i][0],&M1[i][M1.Dim(1)]);
	    std::sort(&M2[i][0],&M2[i][M2.Dim(1)]);
	  }
	  P.Resize(M1.Dim(0),M1.Dim(0));
	  for(size_t i=0;i<M1.Dim(0);i++)
	    for(size_t j=0;j<M1.Dim(0);j++){
	      P[i][j]=1;
	      for(size_t k=0;k<M1.Dim(1);k++)
		if(M1[i][k]!=M2[j][k]){
		  P[i][j]=0;
		  break;
		}
	    }
	}
	std::vector<Permutation<long long> > P1vec, P2vec;
	{
	  Matrix<long long>& Pmat=P1;
	  std::vector<Permutation<long long> >& Pvec=P1vec;
	  Permutation<long long> P(Pmat.Dim(0));
	  std::vector<size_t>& perm=P.perm;
	  //perm.SetZero();
          std::fill(perm.begin(), perm.end(), 0);
	  for(size_t i=0;i<P.Dim();i++)
	    for(size_t j=0;j<P.Dim();j++){
	      if(Pmat[i][j]){
		perm[i]=j;
		break;
	      }
	    }
	  std::vector<size_t> perm_tmp;
	  while(true){
	    perm_tmp=perm;
	    std::sort(&perm_tmp[0],&perm_tmp[0]+perm_tmp.size());
	    for(size_t i=0;i<perm_tmp.size();i++){
	      if(perm_tmp[i]!=i) break;
	      if(i==perm_tmp.size()-1){
		Pvec.push_back(P);
	      }
	    }
	    bool last=false;
	    for(size_t i=0;i<P.Dim();i++){
	      size_t tmp=perm[i];
	      for(size_t j=perm[i]+1;j<P.Dim();j++){
		if(Pmat[i][j]){
		  perm[i]=j;
		  break;
		}
	      }
	      if(perm[i]>tmp) break;
	      for(size_t j=0;j<P.Dim();j++){
		if(Pmat[i][j]){
		  perm[i]=j;
		  break;
		}
	      }
	      if(i==P.Dim()-1) last=true;
	    }
	    if(last) break;
	  }
	}
	{
	  Matrix<long long>& Pmat=P2;
	  std::vector<Permutation<long long> >& Pvec=P2vec;
	  Permutation<long long> P(Pmat.Dim(0));
	  std::vector<size_t>& perm=P.perm;
	  //perm.SetZero();
          std::fill(perm.begin(), perm.end(), 0);
	  for(size_t i=0;i<P.Dim();i++)
	    for(size_t j=0;j<P.Dim();j++){
	      if(Pmat[i][j]){
		perm[i]=j;
		break;
	      }
	    }
	  std::vector<size_t> perm_tmp;
	  while(true){
	    perm_tmp=perm;
	    std::sort(&perm_tmp[0],&perm_tmp[0]+perm_tmp.size());
	    for(size_t i=0;i<perm_tmp.size();i++){
	      if(perm_tmp[i]!=i) break;
	      if(i==perm_tmp.size()-1){
		Pvec.push_back(P);
	      }
	    }
	    bool last=false;
	    for(size_t i=0;i<P.Dim();i++){
	      size_t tmp=perm[i];
	      for(size_t j=perm[i]+1;j<P.Dim();j++){
		if(Pmat[i][j]){
		  perm[i]=j;
		  break;
		}
	      }
	      if(perm[i]>tmp) break;
	      for(size_t j=0;j<P.Dim();j++){
		if(Pmat[i][j]){
		  perm[i]=j;
		  break;
		}
	      }
	      if(i==P.Dim()-1) last=true;
	    }
	    if(last) break;
	  }
	}
	{
	  std::vector<Permutation<long long> > P1vec_, P2vec_;
	  Matrix<long long>  M1=M11;
	  Matrix<long long>  M2=M22;
	  for(size_t i=0;i<M1.Dim(0);i++){
	    for(size_t j=0;j<M1.Dim(1);j++){
	      if(M1[i][j]<0) M1[i][j]=-M1[i][j];
	      if(M2[i][j]<0) M2[i][j]=-M2[i][j];
	    }
	  }
	  Matrix<long long> M;
	  for(size_t i=0;i<P1vec.size();i++)
	    for(size_t j=0;j<P2vec.size();j++){
	      M=P1vec[i]*M2*P2vec[j];
	      for(size_t k=0;k<M.Dim(0)*M.Dim(1);k++){
		if(M[0][k]!=M1[0][k]) break;
		if(k==M.Dim(0)*M.Dim(1)-1){
		  P1vec_.push_back(P1vec[i]);
		  P2vec_.push_back(P2vec[j]);
		}
	      }
	    }
	  P1vec=P1vec_;
	  P2vec=P2vec_;
	}
	Permutation<Real_t> P1_, P2_;
	{
	  for(size_t k=0;k<P1vec.size();k++){
	    Permutation<long long> P1=P1vec[k];
	    Permutation<long long> P2=P2vec[k];
	    Matrix<long long>  M1=   M11   ;
	    Matrix<long long>  M2=P1*M22*P2;
	    Matrix<Real_t> M(M1.Dim(0)*M1.Dim(1)+1,M1.Dim(0)+M1.Dim(1));
	    M.SetZero(); M[M1.Dim(0)*M1.Dim(1)][0]=1.0;
	    for(size_t i=0;i<M1.Dim(0);i++)
	      for(size_t j=0;j<M1.Dim(1);j++){
		size_t k=i*M1.Dim(1)+j;
		M[k][          i]= M1[i][j];
		M[k][M1.Dim(0)+j]=-M2[i][j];
	      }
	    M=M.pinv();
	    {
	      Permutation<long long> P1_(M1.Dim(0));
	      Permutation<long long> P2_(M1.Dim(1));
	      for(size_t i=0;i<M1.Dim(0);i++){
		P1_.scal[i]=(M[i][M1.Dim(0)*M1.Dim(1)]>0?1:-1);
	      }
	      for(size_t i=0;i<M1.Dim(1);i++){
		P2_.scal[i]=(M[M1.Dim(0)+i][M1.Dim(0)*M1.Dim(1)]>0?1:-1);
	      }
	      P1=P1_*P1 ;
	      P2=P2 *P2_;
	    }
	    bool done=true;
	    Matrix<long long> Merr=P1*M22*P2-M11;
	    for(size_t i=0;i<Merr.Dim(0)*Merr.Dim(1);i++){
	      if(Merr[0][i]){
		done=false;
		break;
	      }
	    }
	    if(done){
	      P1_=Permutation<Real_t>(P1.Dim());
	      P2_=Permutation<Real_t>(P2.Dim());
	      for(size_t i=0;i<P1.Dim();i++){
		P1_.perm[i]=P1.perm[i];
		P1_.scal[i]=P1.scal[i];
	      }
	      for(size_t i=0;i<P2.Dim();i++){
		P2_.perm[i]=P2.perm[i];
		P2_.scal[i]=P2.scal[i];
	      }
	      break;
	    }
	  }
	}
	perm_vec[p_type       ]=P1_.Transpose();
	perm_vec[p_type+C_Perm]=P2_;
      }
      for(size_t i=0;i<2*C_Perm;i++){
	if(perm_vec[i].Dim()==0){
	  perm_vec.resize(0);
	  std::cout<<"no-symmetry for: "<<ker_name<<'\n';
	  break;
	}
      }
    }
    {
      if(!k_s2m) k_s2m=this;
      if(!k_s2l) k_s2l=this;
      if(!k_s2t) k_s2t=this;
      if(!k_m2m) k_m2m=this;
      if(!k_m2l) k_m2l=this;
      if(!k_m2t) k_m2t=this;
      if(!k_l2l) k_l2l=this;
      if(!k_l2t) k_l2t=this;
      assert(k_s2t->ker_dim[0]==ker_dim[0]);
      assert(k_s2m->ker_dim[0]==k_s2l->ker_dim[0]);
      assert(k_s2m->ker_dim[0]==k_s2t->ker_dim[0]);
      assert(k_m2m->ker_dim[0]==k_m2l->ker_dim[0]);
      assert(k_m2m->ker_dim[0]==k_m2t->ker_dim[0]);
      assert(k_l2l->ker_dim[0]==k_l2t->ker_dim[0]);
      assert(k_s2t->ker_dim[1]==ker_dim[1]);
      assert(k_s2m->ker_dim[1]==k_m2m->ker_dim[1]);
      assert(k_s2l->ker_dim[1]==k_l2l->ker_dim[1]);
      assert(k_m2l->ker_dim[1]==k_l2l->ker_dim[1]);
      assert(k_s2t->ker_dim[1]==k_m2t->ker_dim[1]);
      assert(k_s2t->ker_dim[1]==k_l2t->ker_dim[1]);
      k_s2m->Initialize(verbose);
      k_s2l->Initialize(verbose);
      k_s2t->Initialize(verbose);
      k_m2m->Initialize(verbose);
      k_m2l->Initialize(verbose);
      k_m2t->Initialize(verbose);
      k_l2l->Initialize(verbose);
      k_l2t->Initialize(verbose);
    }
  }

  void BuildMatrix(Real_t* r_src, int src_cnt, Real_t* r_trg, int trg_cnt, Real_t* k_out) const{
    memset(k_out, 0, src_cnt*ker_dim[0]*trg_cnt*ker_dim[1]*sizeof(Real_t));
    for(int i=0;i<src_cnt;i++)
      for(int j=0;j<ker_dim[0];j++){
	std::vector<Real_t> v_src(ker_dim[0],0);
	v_src[j]=1.0;
	ker_poten(&r_src[i*3], 1, &v_src[0], 1, r_trg, trg_cnt,
		  &k_out[(i*ker_dim[0]+j)*trg_cnt*ker_dim[1]]);
      }
  }

  int ker_dim[2];
  std::string ker_name;
  Ker_t ker_poten;

  mutable bool init;
  mutable bool scale_invar;
  mutable Vector<Real_t> src_scal;
  mutable Vector<Real_t> trg_scal;
  mutable std::vector<Permutation<Real_t> > perm_vec;

  mutable const Kernel* k_s2m;
  mutable const Kernel* k_s2l;
  mutable const Kernel* k_s2t;
  mutable const Kernel* k_m2m;
  mutable const Kernel* k_m2l;
  mutable const Kernel* k_m2t;
  mutable const Kernel* k_l2l;
  mutable const Kernel* k_l2t;

};

Real_t machine_eps(){
  Real_t eps=1.0;
  while(eps+(Real_t)1.0>1.0) eps*=0.5;
  return eps;
}

inline void cheb_poly(int d, const Real_t* in, int n, Real_t* out){
  if(d==0){
    for(int i=0;i<n;i++)
      out[i]=(fabs(in[i])<=1?1.0:0);
  }else if(d==1){
    for(int i=0;i<n;i++){
      out[i]=(fabs(in[i])<=1?1.0:0);
      out[i+n]=(fabs(in[i])<=1?in[i]:0);
    }
  }else{
    for(int j=0;j<n;j++){
      Real_t x=(fabs(in[j])<=1?in[j]:0);
      Real_t y0=(fabs(in[j])<=1?1.0:0);
      out[j]=y0;
      out[j+n]=x;

      Real_t y1=x;
      Real_t* y2=&out[2*n+j];
      for(int i=2;i<=d;i++){
        *y2=2*x*y1-y0;
        y0=y1;
        y1=*y2;
        y2=&y2[n];
      }
    }
  }
}

void quad_rule(int n, Real_t* x, Real_t* w){
  static std::vector<Vector<Real_t> > x_lst(10000);
  static std::vector<Vector<Real_t> > w_lst(10000);
  assert(n<10000);
  bool done=false;
#pragma omp critical (QUAD_RULE)
  if(x_lst[n].Dim()>0){
    Vector<Real_t>& x_=x_lst[n];
    Vector<Real_t>& w_=w_lst[n];
    for(int i=0;i<n;i++){
      x[i]=x_[i];
      w[i]=w_[i];
    }
    done=true;
  }
  if(done) return;
  Vector<Real_t> x_(n);
  Vector<Real_t> w_(n);
  {
    for(int i=0;i<n;i++){
      x_[i]=-cos((Real_t)(2.0*i+1.0)/(2.0*n)*M_PI);
      w_[i]=0;
    }
    Matrix<Real_t> M(n,n);
    cheb_poly(n-1, &x_[0], n, &M[0][0]);
    for(size_t i=0;i<n;i++) M[0][i]/=2.0;

    std::vector<Real_t> w_sample(n,0);
    for(long i=0;i<n;i+=2) w_sample[i]=-((Real_t)2.0/(i+1)/(i-1));
    for(size_t i=0;i<n;i++)
    for(size_t j=0;j<n;j++){
      M[i][j]*=w_sample[i];
    }
    for(size_t i=0;i<n;i++)
    for(size_t j=0;j<n;j++){
      w_[j]+=M[i][j]*2/n;
    }
  }
#pragma omp critical (QUAD_RULE)
  {
    x_lst[n]=x_;
    w_lst[n]=w_;
  }
  quad_rule(n, x, w);
}

std::vector<Real_t> integ_pyramid(int m, Real_t* s, Real_t r, int nx, const Kernel& kernel, int* perm){
  int ny=nx;
  int nz=nx;

  Real_t eps=machine_eps()*64;
  int k_dim=kernel.ker_dim[0]*kernel.ker_dim[1];

  std::vector<Real_t> qp_x(nx), qw_x(nx);
  std::vector<Real_t> qp_y(ny), qw_y(ny);
  std::vector<Real_t> qp_z(nz), qw_z(nz);
  std::vector<Real_t> p_x(nx*m);
  std::vector<Real_t> p_y(ny*m);
  std::vector<Real_t> p_z(nz*m);

  std::vector<Real_t> x_;
  {
    x_.push_back(s[0]);
    x_.push_back(fabs(1.0-s[0])+s[0]);
    x_.push_back(fabs(1.0-s[1])+s[0]);
    x_.push_back(fabs(1.0+s[1])+s[0]);
    x_.push_back(fabs(1.0-s[2])+s[0]);
    x_.push_back(fabs(1.0+s[2])+s[0]);
    std::sort(x_.begin(),x_.end());
    for(int i=0;i<x_.size();i++){
      if(x_[i]<-1.0) x_[i]=-1.0;
      if(x_[i]> 1.0) x_[i]= 1.0;
    }

    std::vector<Real_t> x_new;
    Real_t x_jump=fabs(1.0-s[0]);
    if(fabs(1.0-s[1])>eps) x_jump=std::min(x_jump,(Real_t)fabs(1.0-s[1]));
    if(fabs(1.0+s[1])>eps) x_jump=std::min(x_jump,(Real_t)fabs(1.0+s[1]));
    if(fabs(1.0-s[2])>eps) x_jump=std::min(x_jump,(Real_t)fabs(1.0-s[2]));
    if(fabs(1.0+s[2])>eps) x_jump=std::min(x_jump,(Real_t)fabs(1.0+s[2]));
    for(int k=0; k<x_.size()-1; k++){
      Real_t x0=x_[k];
      Real_t x1=x_[k+1];

      Real_t A0=0;
      Real_t A1=0;
      {
        Real_t y0=s[1]-(x0-s[0]); if(y0<-1.0) y0=-1.0; if(y0> 1.0) y0= 1.0;
        Real_t y1=s[1]+(x0-s[0]); if(y1<-1.0) y1=-1.0; if(y1> 1.0) y1= 1.0;
        Real_t z0=s[2]-(x0-s[0]); if(z0<-1.0) z0=-1.0; if(z0> 1.0) z0= 1.0;
        Real_t z1=s[2]+(x0-s[0]); if(z1<-1.0) z1=-1.0; if(z1> 1.0) z1= 1.0;
        A0=(y1-y0)*(z1-z0);
      }
      {
        Real_t y0=s[1]-(x1-s[0]); if(y0<-1.0) y0=-1.0; if(y0> 1.0) y0= 1.0;
        Real_t y1=s[1]+(x1-s[0]); if(y1<-1.0) y1=-1.0; if(y1> 1.0) y1= 1.0;
        Real_t z0=s[2]-(x1-s[0]); if(z0<-1.0) z0=-1.0; if(z0> 1.0) z0= 1.0;
        Real_t z1=s[2]+(x1-s[0]); if(z1<-1.0) z1=-1.0; if(z1> 1.0) z1= 1.0;
        A1=(y1-y0)*(z1-z0);
      }
      Real_t V=0.5*(A0+A1)*(x1-x0);
      if(V<eps) continue;

      if(!x_new.size()) x_new.push_back(x0);
      x_jump=std::max(x_jump,x0-s[0]);
      while(s[0]+x_jump*1.5<x1){
        x_new.push_back(s[0]+x_jump);
        x_jump*=2.0;
      }
      if(x_new.back()+eps<x1) x_new.push_back(x1);
    }
    assert(x_new.size()<30);
    x_.swap(x_new);
  }

  int err;
  Real_t *k_out, *I0, *I1, *I2;
  err = posix_memalign((void**)&k_out, MEM_ALIGN,   ny*nz*k_dim*sizeof(Real_t));
  err = posix_memalign((void**)&I0,    MEM_ALIGN,   ny*m *k_dim*sizeof(Real_t));
  err = posix_memalign((void**)&I1,    MEM_ALIGN,   m *m *k_dim*sizeof(Real_t));
  err = posix_memalign((void**)&I2,    MEM_ALIGN,m *m *m *k_dim*sizeof(Real_t));
  for (int j=0; j<m*m*m*k_dim; j++) I2[j] = 0;
  if(x_.size()>1)
  for(int k=0; k<x_.size()-1; k++){
    Real_t x0=x_[k];
    Real_t x1=x_[k+1];
    {
      std::vector<Real_t> qp(nx);
      std::vector<Real_t> qw(nx);
      quad_rule(nx,&qp[0],&qw[0]);
      for(int i=0; i<nx; i++)
        qp_x[i]=(x1-x0)*qp[i]/2.0+(x1+x0)/2.0;
      qw_x=qw;
    }
    cheb_poly(m-1,&qp_x[0],nx,&p_x[0]);

    for(int i=0; i<nx; i++){
      Real_t y0=s[1]-(qp_x[i]-s[0]); if(y0<-1.0) y0=-1.0; if(y0> 1.0) y0= 1.0;
      Real_t y1=s[1]+(qp_x[i]-s[0]); if(y1<-1.0) y1=-1.0; if(y1> 1.0) y1= 1.0;
      Real_t z0=s[2]-(qp_x[i]-s[0]); if(z0<-1.0) z0=-1.0; if(z0> 1.0) z0= 1.0;
      Real_t z1=s[2]+(qp_x[i]-s[0]); if(z1<-1.0) z1=-1.0; if(z1> 1.0) z1= 1.0;

      {
        std::vector<Real_t> qp(ny);
        std::vector<Real_t> qw(ny);
        quad_rule(ny,&qp[0],&qw[0]);
        for(int j=0; j<ny; j++)
          qp_y[j]=(y1-y0)*qp[j]/2.0+(y1+y0)/2.0;
        qw_y=qw;
      }
      {
        std::vector<Real_t> qp(nz);
        std::vector<Real_t> qw(nz);
        quad_rule(nz,&qp[0],&qw[0]);
        for(int j=0; j<nz; j++)
          qp_z[j]=(z1-z0)*qp[j]/2.0+(z1+z0)/2.0;
        qw_z=qw;
      }
      cheb_poly(m-1,&qp_y[0],ny,&p_y[0]);
      cheb_poly(m-1,&qp_z[0],nz,&p_z[0]);
      {
        Real_t src[3]={0,0,0};
        std::vector<Real_t> trg(ny*nz*3);
        for(int i0=0; i0<ny; i0++){
          size_t indx0=i0*nz*3;
          for(int i1=0; i1<nz; i1++){
            size_t indx1=indx0+i1*3;
            trg[indx1+perm[0]]=(s[0]-qp_x[i ])*r*0.5*perm[1];
            trg[indx1+perm[2]]=(s[1]-qp_y[i0])*r*0.5*perm[3];
            trg[indx1+perm[4]]=(s[2]-qp_z[i1])*r*0.5*perm[5];
          }
        }
        {
          Matrix<Real_t> k_val(ny*nz*kernel.ker_dim[0],kernel.ker_dim[1]);
          kernel.BuildMatrix(&src[0],1,&trg[0],ny*nz,&k_val[0][0]);
          Matrix<Real_t> k_val_t(kernel.ker_dim[1],ny*nz*kernel.ker_dim[0],&k_out[0], false);
          k_val_t=k_val.Transpose();
        }
        for(int kk=0; kk<k_dim; kk++){
          for(int i0=0; i0<ny; i0++){
            size_t indx=(kk*ny+i0)*nz;
            for(int i1=0; i1<nz; i1++){
              k_out[indx+i1] *= qw_y[i0]*qw_z[i1];
            }
          }
        }
      }

      for (int j=0; j<ny*m*k_dim; j++) I0[j] = 0;
      for(int kk=0; kk<k_dim; kk++){
        for(int i0=0; i0<ny; i0++){
          size_t indx0=(kk*ny+i0)*nz;
          size_t indx1=(kk*ny+i0)* m;
          for(int i2=0; i2<m; i2++){
            for(int i1=0; i1<nz; i1++){
              I0[indx1+i2] += k_out[indx0+i1]*p_z[i2*nz+i1];
            }
          }
        }
      }

      for (int j=0; j<m*m*k_dim; j++) I1[j] = 0;
      for(int kk=0; kk<k_dim; kk++){
        for(int i2=0; i2<ny; i2++){
          size_t indx0=(kk*ny+i2)*m;
          for(int i0=0; i0<m; i0++){
            size_t indx1=(kk* m+i0)*m;
            Real_t py=p_y[i0*ny+i2];
            for(int i1=0; i0+i1<m; i1++){
              I1[indx1+i1] += I0[indx0+i1]*py;
            }
          }
        }
      }

      Real_t v=(x1-x0)*(y1-y0)*(z1-z0);
      for(int kk=0; kk<k_dim; kk++){
        for(int i0=0; i0<m; i0++){
          Real_t px=p_x[i+i0*nx]*qw_x[i]*v;
          for(int i1=0; i0+i1<m; i1++){
            size_t indx0= (kk*m+i1)*m;
            size_t indx1=((kk*m+i0)*m+i1)*m;
            for(int i2=0; i0+i1+i2<m; i2++){
              I2[indx1+i2] += I1[indx0+i2]*px;
            }
          }
        }
      }
    }
  }
  for(int i=0;i<m*m*m*k_dim;i++)
    I2[i]=I2[i]*r*r*r/64.0;

  if(x_.size()>1)
  Profile::Add_FLOP(( 2*ny*nz*m*k_dim
                     +ny*m*(m+1)*k_dim
                     +2*m*(m+1)*k_dim
                     +m*(m+1)*(m+2)/3*k_dim)*nx*(x_.size()-1));

  std::vector<Real_t> I2_(&I2[0], &I2[0]+m*m*m*k_dim);
  free(k_out);
  free(I0);
  free(I1);
  free(I2);
  return I2_;
}

std::vector<Real_t> integ(int m, Real_t* s, Real_t r, int n, const Kernel& kernel){
  int k_dim=kernel.ker_dim[0]*kernel.ker_dim[1];
  Real_t s_[3];
  s_[0]=s[0]*2.0/r-1.0;
  s_[1]=s[1]*2.0/r-1.0;
  s_[2]=s[2]*2.0/r-1.0;

  Real_t s1[3];
  int perm[6];
  std::vector<Real_t> U_[6];

  s1[0]= s_[0];s1[1]=s_[1];s1[2]=s_[2];
  perm[0]= 0; perm[2]= 1; perm[4]= 2;
  perm[1]= 1; perm[3]= 1; perm[5]= 1;
  U_[0]=integ_pyramid(m,s1,r,n,kernel,perm);

  s1[0]=-s_[0];s1[1]=s_[1];s1[2]=s_[2];
  perm[0]= 0; perm[2]= 1; perm[4]= 2;
  perm[1]=-1; perm[3]= 1; perm[5]= 1;
  U_[1]=integ_pyramid(m,s1,r,n,kernel,perm);

  s1[0]= s_[1];s1[1]=s_[0];s1[2]=s_[2];
  perm[0]= 1; perm[2]= 0; perm[4]= 2;
  perm[1]= 1; perm[3]= 1; perm[5]= 1;
  U_[2]=integ_pyramid(m,s1,r,n,kernel,perm);

  s1[0]=-s_[1];s1[1]=s_[0];s1[2]=s_[2];
  perm[0]= 1; perm[2]= 0; perm[4]= 2;
  perm[1]=-1; perm[3]= 1; perm[5]= 1;
  U_[3]=integ_pyramid(m,s1,r,n,kernel,perm);

  s1[0]= s_[2];s1[1]=s_[0];s1[2]=s_[1];
  perm[0]= 2; perm[2]= 0; perm[4]= 1;
  perm[1]= 1; perm[3]= 1; perm[5]= 1;
  U_[4]=integ_pyramid(m,s1,r,n,kernel,perm);

  s1[0]=-s_[2];s1[1]=s_[0];s1[2]=s_[1];
  perm[0]= 2; perm[2]= 0; perm[4]= 1;
  perm[1]=-1; perm[3]= 1; perm[5]= 1;
  U_[5]=integ_pyramid(m,s1,r,n,kernel,perm);

  std::vector<Real_t> U; U.assign(m*m*m*k_dim,0);
  for(int kk=0; kk<k_dim; kk++){
    for(int i=0;i<m;i++){
      for(int j=0;j<m;j++){
        for(int k=0;k<m;k++){
          U[kk*m*m*m + k*m*m + j*m + i]+=U_[0][kk*m*m*m + i*m*m + j*m + k];
          U[kk*m*m*m + k*m*m + j*m + i]+=U_[1][kk*m*m*m + i*m*m + j*m + k]*(i%2?-1.0:1.0);
        }
      }
    }
  }

  for(int kk=0; kk<k_dim; kk++){
    for(int i=0; i<m; i++){
      for(int j=0; j<m; j++){
        for(int k=0; k<m; k++){
          U[kk*m*m*m + k*m*m + i*m + j]+=U_[2][kk*m*m*m + i*m*m + j*m + k];
          U[kk*m*m*m + k*m*m + i*m + j]+=U_[3][kk*m*m*m + i*m*m + j*m + k]*(i%2?-1.0:1.0);
        }
      }
    }
  }

  for(int kk=0; kk<k_dim; kk++){
    for(int i=0; i<m; i++){
      for(int j=0; j<m; j++){
        for(int k=0; k<m; k++){
          U[kk*m*m*m + i*m*m + k*m + j]+=U_[4][kk*m*m*m + i*m*m + j*m + k];
          U[kk*m*m*m + i*m*m + k*m + j]+=U_[5][kk*m*m*m + i*m*m + j*m + k]*(i%2?-1.0:1.0);
        }
      }
    }
  }
  return U;
}

std::vector<Real_t> cheb_integ(int m, Real_t* s_, Real_t r_, const Kernel& kernel){
  Real_t eps=machine_eps();
  Real_t r=r_;
  Real_t s[3]={s_[0],s_[1],s_[2]};
  int n=m+2;
  Real_t err=1.0;
  int k_dim=kernel.ker_dim[0]*kernel.ker_dim[1];
  std::vector<Real_t> U=integ(m+1,s,r,n,kernel);
  std::vector<Real_t> U_;
  while(err>eps*n){
    n=(int)round(n*1.3);
    if(n>300){
      std::cout<<"Cheb_Integ::Failed to converge.[";
      std::cout<<((double)err )<<",";
      std::cout<<((double)s[0])<<",";
      std::cout<<((double)s[1])<<",";
      std::cout<<((double)s[2])<<"]\n";
      break;
    }
    U_=integ(m+1,s,r,n,kernel);
    err=0;
    for(int i=0;i<(m+1)*(m+1)*(m+1)*k_dim;i++)
      if(fabs(U[i]-U_[i])>err)
        err=fabs(U[i]-U_[i]);
    U=U_;
  }
  std::vector<Real_t> U0(((m+1)*(m+2)*(m+3)*k_dim)/6);
  {
    int indx=0;
    const int* ker_dim=kernel.ker_dim;
    for(int l0=0;l0<ker_dim[0];l0++)
    for(int l1=0;l1<ker_dim[1];l1++)
    for(int i=0;i<=m;i++)
    for(int j=0;i+j<=m;j++)
    for(int k=0;i+j+k<=m;k++){
      U0[indx]=U[(k+(j+(i+(l0*ker_dim[1]+l1)*(m+1))*(m+1))*(m+1))];
      indx++;
    }
  }
  return U0;
}

template<void (*A)(Real_t*, int, Real_t*, int, Real_t*, int, Real_t*)>
Kernel BuildKernel(const char* name, std::pair<int,int> k_dim,
    const Kernel* k_s2m=NULL, const Kernel* k_s2l=NULL, const Kernel* k_s2t=NULL,
    const Kernel* k_m2m=NULL, const Kernel* k_m2l=NULL, const Kernel* k_m2t=NULL,
		      const Kernel* k_l2l=NULL, const Kernel* k_l2t=NULL) {
  Kernel K(A, name, k_dim);
  K.k_s2m=k_s2m;
  K.k_s2l=k_s2l;
  K.k_s2t=k_s2t;
  K.k_m2m=k_m2m;
  K.k_m2l=k_m2l;
  K.k_m2t=k_m2t;
  K.k_l2l=k_l2l;
  K.k_l2t=k_l2t;
  return K;
}

template <class Real_t, int SRC_DIM, int TRG_DIM, void (*uKernel)(Matrix<Real_t>&, Matrix<Real_t>&, Matrix<Real_t>&, Matrix<Real_t>&)>
void generic_kernel(Real_t* r_src, int src_cnt, Real_t* v_src, int dof, Real_t* r_trg, int trg_cnt, Real_t* v_trg){
  assert(dof==1);
#if FLOAT
  int VecLen=8;
#else
  int VecLen=4;
#endif
#define STACK_BUFF_SIZE 4096
  Real_t stack_buff[STACK_BUFF_SIZE+MEM_ALIGN];
  Real_t* buff=NULL;
  Matrix<Real_t> src_coord;
  Matrix<Real_t> src_value;
  Matrix<Real_t> trg_coord;
  Matrix<Real_t> trg_value;
  {
    size_t src_cnt_, trg_cnt_;
    src_cnt_=((src_cnt+VecLen-1)/VecLen)*VecLen;
    trg_cnt_=((trg_cnt+VecLen-1)/VecLen)*VecLen;
    size_t buff_size=src_cnt_*(3+SRC_DIM)+
                     trg_cnt_*(3+TRG_DIM);
    if(buff_size>STACK_BUFF_SIZE){
      int err = posix_memalign((void**)&buff, MEM_ALIGN, buff_size*sizeof(Real_t));
    }
    Real_t* buff_ptr=buff;
    if(!buff_ptr){
      uintptr_t ptr=(uintptr_t)stack_buff;
      static uintptr_t     ALIGN_MINUS_ONE=MEM_ALIGN-1;
      static uintptr_t NOT_ALIGN_MINUS_ONE=~ALIGN_MINUS_ONE;
      ptr=((ptr+ALIGN_MINUS_ONE) & NOT_ALIGN_MINUS_ONE);
      buff_ptr=(Real_t*)ptr;
    }
    src_coord.ReInit(3, src_cnt_,buff_ptr,false);  buff_ptr+=3*src_cnt_;
    src_value.ReInit(  SRC_DIM, src_cnt_,buff_ptr,false);  buff_ptr+=  SRC_DIM*src_cnt_;
    trg_coord.ReInit(3, trg_cnt_,buff_ptr,false);  buff_ptr+=3*trg_cnt_;
    trg_value.ReInit(  TRG_DIM, trg_cnt_,buff_ptr,false);
    {
      size_t i=0;
      for(   ;i<src_cnt ;i++){
        for(size_t j=0;j<3;j++){
          src_coord[j][i]=r_src[i*3+j];
        }
      }
      for(   ;i<src_cnt_;i++){
        for(size_t j=0;j<3;j++){
          src_coord[j][i]=0;
        }
      }
    }
    {
      size_t i=0;
      for(   ;i<src_cnt ;i++){
        for(size_t j=0;j<SRC_DIM;j++){
          src_value[j][i]=v_src[i*SRC_DIM+j];
        }
      }
      for(   ;i<src_cnt_;i++){
        for(size_t j=0;j<SRC_DIM;j++){
          src_value[j][i]=0;
        }
      }
    }
    {
      size_t i=0;
      for(   ;i<trg_cnt ;i++){
        for(size_t j=0;j<3;j++){
          trg_coord[j][i]=r_trg[i*3+j];
        }
      }
      for(   ;i<trg_cnt_;i++){
        for(size_t j=0;j<3;j++){
          trg_coord[j][i]=0;
        }
      }
    }
    {
      size_t i=0;
      for(   ;i<trg_cnt_;i++){
        for(size_t j=0;j<TRG_DIM;j++){
          trg_value[j][i]=0;
        }
      }
    }
  }
  uKernel(src_coord,src_value,trg_coord,trg_value);
  {
    for(size_t i=0;i<trg_cnt ;i++){
      for(size_t j=0;j<TRG_DIM;j++){
        v_trg[i*TRG_DIM+j]+=trg_value[j][i];
      }
    }
  }
  if(buff){
    free(buff);
  }
}

void laplace_poten_uKernel(Matrix<Real_t>& src_coord, Matrix<Real_t>& src_value, Matrix<Real_t>& trg_coord, Matrix<Real_t>& trg_value){
#define SRC_BLK 1000
  size_t VecLen=sizeof(Vec_t)/sizeof(Real_t);
  Real_t nwtn_scal=1;
  for(int i=0;i<2;i++){
    nwtn_scal=2*nwtn_scal*nwtn_scal*nwtn_scal;
  }
  const Real_t zero = 0;
  const Real_t OOFP = 1.0/(4*nwtn_scal*M_PI);
  size_t src_cnt_=src_coord.Dim(1);
  size_t trg_cnt_=trg_coord.Dim(1);
  for(size_t sblk=0;sblk<src_cnt_;sblk+=SRC_BLK){
    size_t src_cnt=src_cnt_-sblk;
    if(src_cnt>SRC_BLK) src_cnt=SRC_BLK;
    for(size_t t=0;t<trg_cnt_;t+=VecLen){
      Vec_t tx=load_intrin(&trg_coord[0][t]);
      Vec_t ty=load_intrin(&trg_coord[1][t]);
      Vec_t tz=load_intrin(&trg_coord[2][t]);
      Vec_t tv=zero_intrin(zero);
      for(size_t s=sblk;s<sblk+src_cnt;s++){
        Vec_t dx=sub_intrin(tx,set_intrin(src_coord[0][s]));
        Vec_t dy=sub_intrin(ty,set_intrin(src_coord[1][s]));
        Vec_t dz=sub_intrin(tz,set_intrin(src_coord[2][s]));
        Vec_t sv=              set_intrin(src_value[0][s]) ;
        Vec_t r2=        mul_intrin(dx,dx) ;
        r2=add_intrin(r2,mul_intrin(dy,dy));
        r2=add_intrin(r2,mul_intrin(dz,dz));
        Vec_t rinv=rsqrt_intrin2(r2);
        tv=add_intrin(tv,mul_intrin(rinv,sv));
      }
      Vec_t oofp=set_intrin(OOFP);
      tv=add_intrin(mul_intrin(tv,oofp),load_intrin(&trg_value[0][t]));
      store_intrin(&trg_value[0][t],tv);
    }
  }
  {
    Profile::Add_FLOP((long long)trg_cnt_*(long long)src_cnt_*20);
  }
#undef SRC_BLK
}

void laplace_poten(Real_t* r_src, int src_cnt, Real_t* v_src, int dof, Real_t* r_trg, int trg_cnt, Real_t* v_trg){
  generic_kernel<Real_t, 1, 1, laplace_poten_uKernel>(r_src, src_cnt, v_src, dof, r_trg, trg_cnt, v_trg);
}

void laplace_grad_uKernel(Matrix<Real_t>& src_coord, Matrix<Real_t>& src_value, Matrix<Real_t>& trg_coord, Matrix<Real_t>& trg_value){
#define SRC_BLK 500
  size_t VecLen=sizeof(Vec_t)/sizeof(Real_t);
  Real_t nwtn_scal=1;
  for(int i=0;i<2;i++){
    nwtn_scal=2*nwtn_scal*nwtn_scal*nwtn_scal;
  }
  const Real_t zero = 0;
  const Real_t OOFP = -1.0/(4*nwtn_scal*nwtn_scal*nwtn_scal*M_PI);
  size_t src_cnt_=src_coord.Dim(1);
  size_t trg_cnt_=trg_coord.Dim(1);
  for(size_t sblk=0;sblk<src_cnt_;sblk+=SRC_BLK){
    size_t src_cnt=src_cnt_-sblk;
    if(src_cnt>SRC_BLK) src_cnt=SRC_BLK;
    for(size_t t=0;t<trg_cnt_;t+=VecLen){
      Vec_t tx=load_intrin(&trg_coord[0][t]);
      Vec_t ty=load_intrin(&trg_coord[1][t]);
      Vec_t tz=load_intrin(&trg_coord[2][t]);
      Vec_t tv0=zero_intrin(zero);
      Vec_t tv1=zero_intrin(zero);
      Vec_t tv2=zero_intrin(zero);
      for(size_t s=sblk;s<sblk+src_cnt;s++){
        Vec_t dx=sub_intrin(tx,set_intrin(src_coord[0][s]));
        Vec_t dy=sub_intrin(ty,set_intrin(src_coord[1][s]));
        Vec_t dz=sub_intrin(tz,set_intrin(src_coord[2][s]));
        Vec_t sv=              set_intrin(src_value[0][s]) ;
        Vec_t r2=        mul_intrin(dx,dx) ;
        r2=add_intrin(r2,mul_intrin(dy,dy));
        r2=add_intrin(r2,mul_intrin(dz,dz));
        Vec_t rinv=rsqrt_intrin2(r2);
        Vec_t r3inv=mul_intrin(mul_intrin(rinv,rinv),rinv);
        sv=mul_intrin(sv,r3inv);
        tv0=add_intrin(tv0,mul_intrin(sv,dx));
        tv1=add_intrin(tv1,mul_intrin(sv,dy));
        tv2=add_intrin(tv2,mul_intrin(sv,dz));
      }
      Vec_t oofp=set_intrin(OOFP);
      tv0=add_intrin(mul_intrin(tv0,oofp),load_intrin(&trg_value[0][t]));
      tv1=add_intrin(mul_intrin(tv1,oofp),load_intrin(&trg_value[1][t]));
      tv2=add_intrin(mul_intrin(tv2,oofp),load_intrin(&trg_value[2][t]));
      store_intrin(&trg_value[0][t],tv0);
      store_intrin(&trg_value[1][t],tv1);
      store_intrin(&trg_value[2][t],tv2);
    }
  }
  {
    Profile::Add_FLOP((long long)trg_cnt_*(long long)src_cnt_*27);
  }
#undef SRC_BLK
}

void laplace_grad(Real_t* r_src, int src_cnt, Real_t* v_src, int dof, Real_t* r_trg, int trg_cnt, Real_t* v_trg){
  generic_kernel<Real_t, 1, 3, laplace_grad_uKernel>(r_src, src_cnt, v_src, dof, r_trg, trg_cnt, v_trg);
}

}//end namespace

#endif //_PVFMM_FMM_KERNEL_HPP_

