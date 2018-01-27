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

  inline vec_t rsqrt_intrin2(vec_t r2){
    vec_t rinv=rsqrt_approx_intrin(r2);
    rsqrt_newton_intrin(rinv,r2,real_t(3));
    rsqrt_newton_intrin(rinv,r2,real_t(12));
    return rinv;
  }

struct Kernel{
  public:
  typedef void (*Ker_t)(real_t* r_src, int src_cnt, real_t* v_src, int dof,
                        real_t* r_trg, int trg_cnt, real_t* k_out);

  int ker_dim[2];
  std::string ker_name;
  Ker_t ker_poten;

  mutable bool init;
  mutable bool scale_invar;
  mutable std::vector<real_t> src_scal;
  mutable std::vector<real_t> trg_scal;
  mutable std::vector<Permutation<real_t> > perm_vec;

  mutable const Kernel* k_s2m;
  mutable const Kernel* k_s2l;
  mutable const Kernel* k_s2t;
  mutable const Kernel* k_m2m;
  mutable const Kernel* k_m2l;
  mutable const Kernel* k_m2t;
  mutable const Kernel* k_l2l;
  mutable const Kernel* k_l2t;

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
    src_scal.resize(ker_dim[0]); 
    std::fill(src_scal.begin(), src_scal.end(), 0.);
    trg_scal.resize(ker_dim[1]); 
    std::fill(trg_scal.begin(), trg_scal.end(), 0.);
    perm_vec.resize(Perm_Count);
    for(size_t p_type=0;p_type<C_Perm;p_type++){
      perm_vec[p_type       ]=Permutation<real_t>(ker_dim[0]);
      perm_vec[p_type+C_Perm]=Permutation<real_t>(ker_dim[1]);
    }
    init=false;
  }

  void Initialize(bool verbose=false) const{
    if(init) return;
    init=true;
    real_t eps=1.0;
    while(eps+(real_t)1.0>1.0) eps*=0.5;
    real_t scal=1.0;
    if(ker_dim[0]*ker_dim[1]>0){
      Matrix<real_t> M_scal(ker_dim[0],ker_dim[1]);
      size_t N=1024;
      real_t eps_=N*eps;
      real_t src_coord[3]={0,0,0};
      std::vector<real_t> trg_coord1(N*3);
      Matrix<real_t> M1(N,ker_dim[0]*ker_dim[1]);
      while(true){
	real_t abs_sum=0;
	for(size_t i=0;i<N/2;i++){
	  real_t x,y,z,r;
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
	  real_t x,y,z,r;
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

      std::vector<real_t> trg_coord2(N*3);
      Matrix<real_t> M2(N,ker_dim[0]*ker_dim[1]);
      for(size_t i=0;i<N*3;i++){
	trg_coord2[i]=trg_coord1[i]*0.5;
      }
      for(size_t i=0;i<N;i++){
	BuildMatrix(&src_coord [          0], 1,
		    &trg_coord2[i*3], 1, &(M2[i][0]));
      }

      for(size_t i=0;i<ker_dim[0]*ker_dim[1];i++){
	real_t dot11=0, dot12=0, dot22=0;
	for(size_t j=0;j<N;j++){
	  dot11+=M1[j][i]*M1[j][i];
	  dot12+=M1[j][i]*M2[j][i];
	  dot22+=M2[j][i]*M2[j][i];
	}
	real_t max_val=std::max<real_t>(dot11,dot22);
	if(dot11>max_val*eps &&
	   dot22>max_val*eps ){
	  real_t s=dot12/dot11;
	  M_scal[0][i]=log(s)/log(2.0);
	  real_t err=sqrtf(0.5*(dot22/dot11)/(s*s)-0.5);
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
      src_scal.resize(ker_dim[0]); 
      std::fill(src_scal.begin(), src_scal.end(), 0.);
      trg_scal.resize(ker_dim[1]); 
      std::fill(trg_scal.begin(), trg_scal.end(), 0.);
      if(scale_invar){
	Matrix<real_t> b(ker_dim[0]*ker_dim[1]+1,1); b.SetZero();
	memcpy(&b[0][0],&M_scal[0][0],ker_dim[0]*ker_dim[1]*sizeof(real_t));
	Matrix<real_t> M(ker_dim[0]*ker_dim[1]+1,ker_dim[0]+ker_dim[1]); M.SetZero();
	M[ker_dim[0]*ker_dim[1]][0]=1;
	for(size_t i0=0;i0<ker_dim[0];i0++)
	  for(size_t i1=0;i1<ker_dim[1];i1++){
	    size_t j=i0*ker_dim[1]+i1;
	    if(fabs(b[j][0])>=0){
	      M[j][ 0+        i0]=1;
	      M[j][i1+ker_dim[0]]=1;
	    }
	  }
	Matrix<real_t> x=M.pinv()*b;
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
        std::fill(src_scal.begin(), src_scal.end(), 0.);
        std::fill(trg_scal.begin(), trg_scal.end(), 0.);
      }
    }
    if(ker_dim[0]*ker_dim[1]>0){
      size_t N=1024;
      real_t eps_=N*eps;
      real_t src_coord[3]={0,0,0};
      std::vector<real_t> trg_coord1(N*3);
      std::vector<real_t> trg_coord2(N*3);
      for(size_t i=0;i<N/2;i++){
	real_t x,y,z,r;
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
	real_t x,y,z,r;
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
	  Matrix<real_t> M1(N,ker_dim[0]*ker_dim[1]); M1.SetZero();
	  Matrix<real_t> M2(N,ker_dim[0]*ker_dim[1]); M2.SetZero();
	  for(size_t i=0;i<N;i++){
	    BuildMatrix(&src_coord [          0], 1,
			&trg_coord1[i*3], 1, &(M1[i][0]));
	    BuildMatrix(&src_coord [          0], 1,
			&trg_coord2[i*3], 1, &(M2[i][0]));
	  }
	  Matrix<real_t> dot11(ker_dim[0]*ker_dim[1],ker_dim[0]*ker_dim[1]);dot11.SetZero();
	  Matrix<real_t> dot12(ker_dim[0]*ker_dim[1],ker_dim[0]*ker_dim[1]);dot12.SetZero();
	  Matrix<real_t> dot22(ker_dim[0]*ker_dim[1],ker_dim[0]*ker_dim[1]);dot22.SetZero();
	  std::vector<real_t> norm1(ker_dim[0]*ker_dim[1]);
	  std::vector<real_t> norm2(ker_dim[0]*ker_dim[1]);
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
	Permutation<real_t> P1_, P2_;
	{
	  for(size_t k=0;k<P1vec.size();k++){
	    Permutation<long long> P1=P1vec[k];
	    Permutation<long long> P2=P2vec[k];
	    Matrix<long long>  M1=   M11   ;
	    Matrix<long long>  M2=P1*M22*P2;
	    Matrix<real_t> M(M1.Dim(0)*M1.Dim(1)+1,M1.Dim(0)+M1.Dim(1));
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
	      P1_=Permutation<real_t>(P1.Dim());
	      P2_=Permutation<real_t>(P2.Dim());
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

  void BuildMatrix(real_t* r_src, int src_cnt, real_t* r_trg, int trg_cnt, real_t* k_out) const{
    memset(k_out, 0, src_cnt*ker_dim[0]*trg_cnt*ker_dim[1]*sizeof(real_t));
    for(int i=0;i<src_cnt;i++)
      for(int j=0;j<ker_dim[0];j++){
	std::vector<real_t> v_src(ker_dim[0],0);
	v_src[j]=1.0;
	ker_poten(&r_src[i*3], 1, &v_src[0], 1, r_trg, trg_cnt,
		  &k_out[(i*ker_dim[0]+j)*trg_cnt*ker_dim[1]]);
      }
  }
};

template<void (*A)(real_t*, int, real_t*, int, real_t*, int, real_t*)>
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

template <class real_t, int SRC_DIM, int TRG_DIM, void (*uKernel)(Matrix<real_t>&, Matrix<real_t>&, Matrix<real_t>&, Matrix<real_t>&)>
void generic_kernel(real_t* r_src, int src_cnt, real_t* v_src, int dof, real_t* r_trg, int trg_cnt, real_t* v_trg){
  assert(dof==1);
#if FLOAT
  int VecLen=8;
#else
  int VecLen=4;
#endif
#define STACK_BUFF_SIZE 4096
  real_t stack_buff[STACK_BUFF_SIZE+MEM_ALIGN];
  real_t* buff=NULL;
  Matrix<real_t> src_coord;
  Matrix<real_t> src_value;
  Matrix<real_t> trg_coord;
  Matrix<real_t> trg_value;
  {
    size_t src_cnt_, trg_cnt_;
    src_cnt_=((src_cnt+VecLen-1)/VecLen)*VecLen;
    trg_cnt_=((trg_cnt+VecLen-1)/VecLen)*VecLen;
    size_t buff_size=src_cnt_*(3+SRC_DIM)+
                     trg_cnt_*(3+TRG_DIM);
    if(buff_size>STACK_BUFF_SIZE){
      int err = posix_memalign((void**)&buff, MEM_ALIGN, buff_size*sizeof(real_t));
    }
    real_t* buff_ptr=buff;
    if(!buff_ptr){
      uintptr_t ptr=(uintptr_t)stack_buff;
      static uintptr_t     ALIGN_MINUS_ONE=MEM_ALIGN-1;
      static uintptr_t NOT_ALIGN_MINUS_ONE=~ALIGN_MINUS_ONE;
      ptr=((ptr+ALIGN_MINUS_ONE) & NOT_ALIGN_MINUS_ONE);
      buff_ptr=(real_t*)ptr;
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

void laplace_poten_uKernel(Matrix<real_t>& src_coord, Matrix<real_t>& src_value, Matrix<real_t>& trg_coord, Matrix<real_t>& trg_value){
#define SRC_BLK 1000
  size_t VecLen=sizeof(vec_t)/sizeof(real_t);
  real_t nwtn_scal=1;
  for(int i=0;i<2;i++){
    nwtn_scal=2*nwtn_scal*nwtn_scal*nwtn_scal;
  }
  const real_t zero = 0;
  const real_t OOFP = 1.0/(4*nwtn_scal*M_PI);
  size_t src_cnt_=src_coord.Dim(1);
  size_t trg_cnt_=trg_coord.Dim(1);
  for(size_t sblk=0;sblk<src_cnt_;sblk+=SRC_BLK){
    size_t src_cnt=src_cnt_-sblk;
    if(src_cnt>SRC_BLK) src_cnt=SRC_BLK;
    for(size_t t=0;t<trg_cnt_;t+=VecLen){
      vec_t tx=load_intrin(&trg_coord[0][t]);
      vec_t ty=load_intrin(&trg_coord[1][t]);
      vec_t tz=load_intrin(&trg_coord[2][t]);
      vec_t tv=zero_intrin(zero);
      for(size_t s=sblk;s<sblk+src_cnt;s++){
        vec_t dx=sub_intrin(tx,set_intrin(src_coord[0][s]));
        vec_t dy=sub_intrin(ty,set_intrin(src_coord[1][s]));
        vec_t dz=sub_intrin(tz,set_intrin(src_coord[2][s]));
        vec_t sv=              set_intrin(src_value[0][s]) ;
        vec_t r2=        mul_intrin(dx,dx) ;
        r2=add_intrin(r2,mul_intrin(dy,dy));
        r2=add_intrin(r2,mul_intrin(dz,dz));
        vec_t rinv=rsqrt_intrin2(r2);
        tv=add_intrin(tv,mul_intrin(rinv,sv));
      }
      vec_t oofp=set_intrin(OOFP);
      tv=add_intrin(mul_intrin(tv,oofp),load_intrin(&trg_value[0][t]));
      store_intrin(&trg_value[0][t],tv);
    }
  }
  {
    Profile::Add_FLOP((long long)trg_cnt_*(long long)src_cnt_*20);
  }
#undef SRC_BLK
}

void laplace_poten(real_t* r_src, int src_cnt, real_t* v_src, int dof, real_t* r_trg, int trg_cnt, real_t* v_trg){
  generic_kernel<real_t, 1, 1, laplace_poten_uKernel>(r_src, src_cnt, v_src, dof, r_trg, trg_cnt, v_trg);
}

void laplace_grad_uKernel(Matrix<real_t>& src_coord, Matrix<real_t>& src_value, Matrix<real_t>& trg_coord, Matrix<real_t>& trg_value){
#define SRC_BLK 500
  size_t VecLen=sizeof(vec_t)/sizeof(real_t);
  real_t nwtn_scal=1;
  for(int i=0;i<2;i++){
    nwtn_scal=2*nwtn_scal*nwtn_scal*nwtn_scal;
  }
  const real_t zero = 0;
  const real_t OOFP = -1.0/(4*nwtn_scal*nwtn_scal*nwtn_scal*M_PI);
  size_t src_cnt_=src_coord.Dim(1);
  size_t trg_cnt_=trg_coord.Dim(1);
  for(size_t sblk=0;sblk<src_cnt_;sblk+=SRC_BLK){
    size_t src_cnt=src_cnt_-sblk;
    if(src_cnt>SRC_BLK) src_cnt=SRC_BLK;
    for(size_t t=0;t<trg_cnt_;t+=VecLen){
      vec_t tx=load_intrin(&trg_coord[0][t]);
      vec_t ty=load_intrin(&trg_coord[1][t]);
      vec_t tz=load_intrin(&trg_coord[2][t]);
      vec_t tv0=zero_intrin(zero);
      vec_t tv1=zero_intrin(zero);
      vec_t tv2=zero_intrin(zero);
      for(size_t s=sblk;s<sblk+src_cnt;s++){
        vec_t dx=sub_intrin(tx,set_intrin(src_coord[0][s]));
        vec_t dy=sub_intrin(ty,set_intrin(src_coord[1][s]));
        vec_t dz=sub_intrin(tz,set_intrin(src_coord[2][s]));
        vec_t sv=              set_intrin(src_value[0][s]) ;
        vec_t r2=        mul_intrin(dx,dx) ;
        r2=add_intrin(r2,mul_intrin(dy,dy));
        r2=add_intrin(r2,mul_intrin(dz,dz));
        vec_t rinv=rsqrt_intrin2(r2);
        vec_t r3inv=mul_intrin(mul_intrin(rinv,rinv),rinv);
        sv=mul_intrin(sv,r3inv);
        tv0=add_intrin(tv0,mul_intrin(sv,dx));
        tv1=add_intrin(tv1,mul_intrin(sv,dy));
        tv2=add_intrin(tv2,mul_intrin(sv,dz));
      }
      vec_t oofp=set_intrin(OOFP);
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

void laplace_grad(real_t* r_src, int src_cnt, real_t* v_src, int dof, real_t* r_trg, int trg_cnt, real_t* v_trg){
  generic_kernel<real_t, 1, 3, laplace_grad_uKernel>(r_src, src_cnt, v_src, dof, r_trg, trg_cnt, v_trg);
}

}//end namespace

#endif //_PVFMM_FMM_KERNEL_HPP_

