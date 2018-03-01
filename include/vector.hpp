#ifndef _PVFMM_VECTOR_HPP_
#define _PVFMM_VECTOR_HPP_

namespace pvfmm{
#define MEM_ALIGN 64
template <class T>
class Vector{
public:
  int dim;
  int capacity;
  bool own_data;
  T* data_ptr;

  Vector(){
    dim=0;
    capacity=0;
    own_data=true;
    data_ptr=NULL;
  }

  Vector(int dim_, T* data_=NULL, bool own_data_=true){
    dim=dim_;
    capacity=dim;
    own_data=own_data_;
    if(own_data){
      if(dim>0){
	int err = posix_memalign((void**)&data_ptr, MEM_ALIGN, dim*sizeof(T));
	if(data_!=NULL) memcpy(data_ptr,data_,dim*sizeof(T));
      }else data_ptr=NULL;
    }else
      data_ptr=data_;
  }

  Vector(const Vector<T>& V){
    dim=V.dim;
    capacity=dim;
    own_data=true;
    if(dim>0){
      int err = posix_memalign((void**)&data_ptr, MEM_ALIGN, dim*sizeof(T));
      memcpy(data_ptr,V.data_ptr,dim*sizeof(T));
    }else
      data_ptr=NULL;
  }

  Vector(const std::vector<T>& V){
    dim=V.size();
    capacity=dim;
    own_data=true;
    if(dim>0){
      int err = posix_memalign((void**)&data_ptr, MEM_ALIGN, dim*sizeof(T));
      memcpy(data_ptr,&V[0],dim*sizeof(T));
    }else
      data_ptr=NULL;
  }

  ~Vector(){
    if(own_data){
      if(data_ptr!=NULL){
	free(data_ptr);
      }
    }
    data_ptr=NULL;
    capacity=0;
    dim=0;
  }

  void ReInit3(int dim_, T* data_, bool own_data_){
    dim=dim_;
    capacity=dim;
    own_data=own_data_;
    data_ptr=data_;
  }

  void Resize(int dim_){
    dim=dim_;
    if(capacity<dim_){
      capacity=dim;
      own_data=true;
      int err = posix_memalign((void**)&data_ptr, MEM_ALIGN, dim*sizeof(T));
    }
  }

  void Write(const char* fname){
    FILE* f1=fopen(fname,"wb+");
    if(f1==NULL){
      std::cout<<"Unable to open file for writing:"<<fname<<'\n';
      return;
    }
    int dim_=dim;
    fwrite(&dim_,sizeof(int),2,f1);
    fwrite(data_ptr,sizeof(T),dim,f1);
    fclose(f1);
  }

  inline int Dim() const{
    return dim;
  }

  inline int Capacity() const{
    return capacity;
  }

  void SetZero(){
    if(dim>0)
      memset(data_ptr,0,dim*sizeof(T));
  }

  Vector<T>& operator=(const Vector<T>& V){
    Resize(V.dim);
    memcpy(data_ptr,V.data_ptr,dim*sizeof(T));
    return *this;
  }

  inline T& operator[](int j) const{
    assert(dim>0?j<dim:j==0); //TODO Change to (j<dim)
    return data_ptr[j];
  }


};

}//end namespace

#endif //_PVFMM_VECTOR_HPP_
