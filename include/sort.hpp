#ifndef _PVFMM_OMP_UTILS_H_
#define _PVFMM_OMP_UTILS_H_

namespace pvfmm{

  template <class T>
  void merge(T A_,T A_last,T B_,T B_last,T C_,int p){
    typedef typename std::iterator_traits<T>::difference_type _DiffType;
    typedef typename std::iterator_traits<T>::value_type _ValType;
    _DiffType N1=A_last-A_;
    _DiffType N2=B_last-B_;
    if(N1==0 && N2==0) return;
    if(N1==0 || N2==0){
      _ValType* A=(N1==0? &B_[0]: &A_[0]);
      _DiffType N=(N1==0?  N2  :  N1   );
#pragma omp parallel for
      for(int i=0;i<p;i++){
	_DiffType indx1=( i   *N)/p;
	_DiffType indx2=((i+1)*N)/p;
	memcpy(&C_[indx1], &A[indx1], (indx2-indx1)*sizeof(_ValType));
      }
      return;
    }
    int n=10;
    _ValType* split=new _ValType [p*n*2];
    _DiffType* split_size=new _DiffType [p*n*2];
#pragma omp parallel for
    for(int i=0;i<p;i++){
      for(int j=0;j<n;j++){
	int indx=i*n+j;
	_DiffType indx1=(indx*N1)/(p*n);
	split   [indx]=A_[indx1];
	split_size[indx]=indx1+(std::lower_bound(B_,B_last,split[indx])-B_);

	indx1=(indx*N2)/(p*n);
	indx+=p*n;
	split   [indx]=B_[indx1];
	split_size[indx]=indx1+(std::lower_bound(A_,A_last,split[indx])-A_);
      }
    }
    _DiffType* split_indx_A=new _DiffType [p+1];
    _DiffType* split_indx_B=new _DiffType [p+1];
    split_indx_A[0]=0;
    split_indx_B[0]=0;
    split_indx_A[p]=N1;
    split_indx_B[p]=N2;
#pragma omp parallel for
    for(int i=1;i<p;i++){
      _DiffType req_size=(i*(N1+N2))/p;

      int j=std::lower_bound(&split_size[0],&split_size[p*n],req_size,std::less<_DiffType>())-&split_size[0];
      if(j>=p*n)
	j=p*n-1;
      _ValType  split1     =split     [j];
      _DiffType split_size1=split_size[j];

      j=(std::lower_bound(&split_size[p*n],&split_size[p*n*2],req_size,std::less<_DiffType>())-&split_size[p*n])+p*n;
      if(j>=2*p*n)
	j=2*p*n-1;
      if(std::abs(split_size[j]-req_size)<std::abs(split_size1-req_size)){
	split1     =split   [j];
	split_size1=split_size[j];
      }

      split_indx_A[i]=std::lower_bound(A_,A_last,split1)-A_;
      split_indx_B[i]=std::lower_bound(B_,B_last,split1)-B_;
    }
    delete[] split;
    delete[] split_size;
#pragma omp parallel for
    for(int i=0;i<p;i++){
      T C=C_+split_indx_A[i]+split_indx_B[i];
      std::merge(A_+split_indx_A[i],A_+split_indx_A[i+1],B_+split_indx_B[i],B_+split_indx_B[i+1],C);
    }
    delete[] split_indx_A;
    delete[] split_indx_B;
  }

  template <class T>
  void merge_sort(T A,T A_last){
    typedef typename std::iterator_traits<T>::difference_type _DiffType;
    typedef typename std::iterator_traits<T>::value_type _ValType;
    int p=omp_get_max_threads();
    _DiffType N=A_last-A;
    if(N<2*p){
      std::sort(A,A_last);
      return;
    }
    _DiffType* split=new _DiffType [p+1];
    split[p]=N;
#pragma omp parallel for
    for(int id=0;id<p;id++){
      split[id]=(id*N)/p;
    }
#pragma omp parallel for
    for(int id=0;id<p;id++){
      std::sort(A+split[id],A+split[id+1]);
    }
    _ValType* B=new _ValType [N];
    _ValType* A_=&A[0];
    _ValType* B_=&B[0];
    for(int j=1;j<p;j=j*2){
      for(int i=0;i<p;i=i+2*j){
	if(i+j<p){
	  merge(A_+split[i],A_+split[i+j],A_+split[i+j],A_+split[(i+2*j<=p?i+2*j:p)],B_+split[i],p);
	}else{
#pragma omp parallel for
	  for(int k=split[i];k<split[p];k++)
	    B_[k]=A_[k];
	}
      }
      _ValType* tmp_swap=A_;
      A_=B_;
      B_=tmp_swap;
    }
    if(A_!=&A[0]){
#pragma omp parallel for
      for(int i=0;i<N;i++)
	A[i]=A_[i];
    }
    delete[] split;
    delete[] B;
  }

  template <class T, class I>
  void scan(T* A, T* B,I cnt){
    int p=omp_get_max_threads();
    if(cnt<(I)100*p){
      for(I i=1;i<cnt;i++)
	B[i]=B[i-1]+A[i-1];
      return;
    }
    I step_size=cnt/p;
#pragma omp parallel for
    for(int i=0; i<p; i++){
      int start=i*step_size;
      int end=start+step_size;
      if(i==p-1) end=cnt;
      if(i!=0)B[start]=0;
      for(I j=(I)start+1; j<(I)end; j++)
	B[j]=B[j-1]+A[j-1];
    }
    T* sum=new T [p];
    sum[0]=0;
    for(int i=1;i<p;i++)
      sum[i]=sum[i-1]+B[i*step_size-1]+A[i*step_size-1];
#pragma omp parallel for
    for(int i=1; i<p; i++){
      int start=i*step_size;
      int end=start+step_size;
      if(i==p-1) end=cnt;
      T sum_=sum[i];
      for(I j=(I)start; j<(I)end; j++)
	B[j]+=sum_;
    }
    delete[] sum;
  }

    template<typename A, typename B>
  struct SortPair{
    int operator<(const SortPair<A,B>& p1) const{ return key<p1.key;}
    A key;
    B data;
  };

  template<typename T>
  int HyperQuickSort(const std::vector<T>& arr_, std::vector<T>& SortedElem){
    srand(0);
    long long nelem = arr_.size();
    std::vector<T> arr=arr_;
    merge_sort(&arr[0], &arr[0]+nelem);
    SortedElem.resize(nelem);
    memcpy(&SortedElem[0], &arr[0], nelem*sizeof(T));
    return 0;
  }

  template<typename T>
  int SortIndex(const std::vector<T>& key, std::vector<size_t>& index, const T* split_key_){
    typedef SortPair<T,size_t> Pair_t;
    std::vector<Pair_t> parray(key.size());
    long long loc_size=key.size();
#pragma omp parallel for
    for(size_t i=0;i<loc_size;i++){
      parray[i].key=key[i];
      parray[i].data=i;
    }
    std::vector<Pair_t> psorted;
    HyperQuickSort(parray, psorted);
    index.resize(psorted.size());
#pragma omp parallel for
    for(size_t i=0;i<psorted.size();i++){
      index[i]=psorted[i].data;
    }
    return 0;
  }

  template<typename T>
  int Forward(Vector<T>& data_, const std::vector<size_t>& index){
    typedef SortPair<size_t,size_t> Pair_t;
    long long data_size=index.size();
    long long loc_size[2]={(long long)(data_.Dim()*sizeof(T)), data_size};
    if(loc_size[0]==0 || loc_size[1]==0) return 0;
    size_t data_dim=loc_size[0]/loc_size[1];
    Vector<char> buffer(data_size*data_dim);
    std::vector<Pair_t> psorted(data_size);
    {
#pragma omp parallel for
      for(size_t i=0;i<data_size;i++){
	psorted[i].key=index[i];
	psorted[i].data=i;
      }
      merge_sort(&psorted[0], &psorted[0]+data_size);
    }
    {
      char* data=(char*)&data_[0];
#pragma omp parallel for
      for(size_t i=0;i<data_size;i++){
	for(size_t j=0;j<data_dim;j++)
	  buffer[i*data_dim+j]=data[i*data_dim+j];
      }
#pragma omp parallel for
      for(size_t i=0;i<data_size;i++){
	size_t src_indx=psorted[i].key*data_dim;
	size_t trg_indx=psorted[i].data*data_dim;
	for(size_t j=0;j<data_dim;j++)
	  data[trg_indx+j]=buffer[src_indx+j];
      }
    }
    return 0;
  }

}//end namespace

#endif //_PVFMM_OMP_UTILS_H_
