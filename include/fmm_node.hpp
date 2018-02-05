#ifndef _PVFMM_FMM_NODE_HPP_
#define _PVFMM_FMM_NODE_HPP_
#include "pvfmm.h"
namespace pvfmm{

class FMM_Node {
 public:
  int depth;
  int max_depth;
  int path2node;
  FMM_Node* parent;
  FMM_Node** child;
  int status;
  size_t max_pts;
  size_t node_id;
  real_t coord[3];
  FMM_Node * colleague[27];
  Vector<real_t> pt_coord;
  Vector<real_t> pt_value;
  Vector<size_t> pt_scatter;
  Vector<real_t> src_coord;
  Vector<real_t> src_value;
  Vector<size_t> src_scatter;
  Vector<real_t> surf_coord;
  Vector<real_t> surf_value;
  Vector<size_t> surf_scatter;
  Vector<real_t> trg_coord;
  Vector<real_t> trg_value;
  Vector<size_t> trg_scatter;
  size_t pt_cnt[2];
  std::vector<FMM_Node*> interac_list[Type_Count];
  FMM_Data* fmm_data;

  FMM_Node() : depth(0), max_depth(MAX_DEPTH), parent(NULL), child(NULL), status(1) {
    fmm_data=NULL;
  }

  ~FMM_Node(){
    if(fmm_data!=NULL) delete fmm_data;
    fmm_data=NULL;
    if(!child) return;
    int n=(1UL<<3);
    for(int i=0;i<n;i++){
      if(child[i]!=NULL)
	delete child[i];
    }
    delete[] child;
    child=NULL;
  }

  void Initialize(FMM_Node* parent_, int path2node_, InitData* data_){
    parent=parent_;
    depth=(parent==NULL?0:parent->depth+1);
    if(data_!=NULL){
      max_depth=data_->max_depth;
      if(max_depth>MAX_DEPTH) max_depth=MAX_DEPTH;
    }else if(parent!=NULL){
      max_depth=parent->max_depth;
    }
    assert(path2node_>=0 && path2node_<(int)(1U<<3));
    path2node=path2node_;
    real_t coord_offset=((real_t)1.0)/((real_t)(((uint64_t)1)<<depth));
    if(!parent_){
      for(int j=0;j<3;j++) coord[j]=0;
    }else if(parent_){
      int flag=1;
      for(int j=0;j<3;j++){
	coord[j]=parent_->coord[j]+
	  ((Path2Node() & flag)?coord_offset:0.0f);
	flag=flag<<1;
      }
    }
    int n=27;
    for(int i=0;i<n;i++) colleague[i]=NULL;
    InitData* data=dynamic_cast<InitData*>(data_);
    if(data_){
      max_pts=data->max_pts;
      pt_coord=data->coord;
      pt_value=data->value;
    }else if(parent){
      max_pts =parent->max_pts;
    }
  }

  void NodeDataVec(std::vector<Vector<real_t>*>& coord,
                   std::vector<Vector<real_t>*>& value,
                   std::vector<Vector<size_t>*>& scatter){
    coord  .push_back(&pt_coord  );
    value  .push_back(&pt_value  );
    scatter.push_back(&pt_scatter);
    coord  .push_back(&src_coord  );
    value  .push_back(&src_value  );
    scatter.push_back(&src_scatter);
    coord  .push_back(&surf_coord  );
    value  .push_back(&surf_value  );
    scatter.push_back(&surf_scatter);
    coord  .push_back(&trg_coord  );
    value  .push_back(&trg_value  );
    scatter.push_back(&trg_scatter);
  }

  void Truncate() {
    if(!child) return;
    SetStatus(1);
    int n=(1UL<<3);
    for(int i=0;i<n;i++){
      if(child[i]!=NULL)
	delete child[i];
    }
    delete[] child;
    child=NULL;
  }

  FMM_Data*& FMMData() {
    return fmm_data;
  }

  FMM_Node* NewNode() {
    FMM_Node* n=new FMM_Node();
    if(fmm_data!=NULL) n->fmm_data=new FMM_Data();
    n->max_depth=max_depth;
    n->max_pts=max_pts;
    return n;
  }

  void Subdivide(){
    if(!IsLeaf()) return;
    if(child) return;
    SetStatus(1);
    int n = 8;
    child=new FMM_Node* [n];
    for(int i=0;i<n;i++){
      child[i]=NewNode();
      child[i]->parent=this;
      child[i]->Initialize(this,i,NULL);
    }
    int nchld = 8;

    std::vector<Vector<real_t>*> pt_c;
    std::vector<Vector<real_t>*> pt_v;
    std::vector<Vector<size_t>*> pt_s;
    NodeDataVec(pt_c, pt_v, pt_s);

    std::vector<std::vector<Vector<real_t>*> > chld_pt_c(nchld);
    std::vector<std::vector<Vector<real_t>*> > chld_pt_v(nchld);
    std::vector<std::vector<Vector<size_t>*> > chld_pt_s(nchld);
    for(size_t i=0;i<nchld;i++){
      Child(i)->NodeDataVec(chld_pt_c[i], chld_pt_v[i], chld_pt_s[i]);
    }

    real_t* c=Coord();
    real_t s=powf(0.5,depth+1);
    for(size_t j=0;j<pt_c.size();j++){
      if(!pt_c[j] || !pt_c[j]->Dim()) continue;
      Vector<real_t>& coord=*pt_c[j];
      size_t npts=coord.Dim()/3;

      Vector<size_t> cdata(nchld+1);
      for(size_t i=0;i<nchld+1;i++){
        long long pt1=-1, pt2=npts;
        while(pt2-pt1>1){
          long long pt3=(pt1+pt2)/2;
          assert(pt3<npts);
          if(pt3<0) pt3=0;
          int ch_id=(coord[pt3*3+0]>=c[0]+s)*1+
            (coord[pt3*3+1]>=c[1]+s)*2+
            (coord[pt3*3+2]>=c[2]+s)*4;
          if(ch_id< i) pt1=pt3;
          if(ch_id>=i) pt2=pt3;
        }
        cdata[i]=pt2;
      }

      if(pt_c[j]){
        Vector<real_t>& vec=*pt_c[j];
        size_t dof=vec.Dim()/npts;
        assert(dof>0);
        for(size_t i=0;i<nchld;i++){
          Vector<real_t>& chld_vec=*chld_pt_c[i][j];
          chld_vec.Resize((cdata[i+1]-cdata[i])*dof);
          for (int k=cdata[i]*dof; k<cdata[i+1]*dof; k++) {
            chld_vec[k-cdata[i]*dof] = vec[k];
          }
        }
        vec.Resize(0);
      }
      if(pt_v[j]){
        Vector<real_t>& vec=*pt_v[j];
        size_t dof=vec.Dim()/npts;
        for(size_t i=0;i<nchld;i++){
          Vector<real_t>& chld_vec=*chld_pt_v[i][j];
          chld_vec.Resize((cdata[i+1]-cdata[i])*dof);
          for (int k=cdata[i]*dof; k<cdata[i+1]*dof; k++) {
            chld_vec[k-cdata[i]*dof] = vec[k];
          }
        }
        vec.Resize(0);
      }
      if(pt_s[j]){
        Vector<size_t>& vec=*pt_s[j];
        size_t dof=vec.Dim()/npts;
        for(size_t i=0;i<nchld;i++){
          Vector<size_t>& chld_vec=*chld_pt_s[i][j];
          chld_vec.Resize((cdata[i+1]-cdata[i])*dof);
          for (int k=cdata[i]*dof; k<cdata[i+1]*dof; k++) {
            chld_vec[k-cdata[i]*dof] = vec[k];
          }
        }
        vec.Resize(0);
      }
    }
  }

  bool IsLeaf() {
    return child == NULL;
  }

  int& GetStatus() {
    return status;
  }

  void SetStatus(int flag) {
    status=(status|flag);
    if(parent && !(parent->GetStatus() & flag))
      parent->SetStatus(flag);
  }

  FMM_Node* Child(int id){
    assert(id<8);
    if(child==NULL) return NULL;
    return child[id];
  }

  FMM_Node* Parent(){
    return parent;
  }

  inline MortonId GetMortonId() {
    assert(coord);
    real_t s=0.25/(1UL<<MAX_DEPTH);
    return MortonId(coord[0]+s,coord[1]+s,coord[2]+s, depth);
  }

  inline void SetCoord(MortonId& mid) {
    assert(coord);
    mid.GetCoord(coord);
    depth=mid.GetDepth();
  }

  int Path2Node(){
    return path2node;
  }

  void SetParent(FMM_Node* p, int path2node_) {
    assert(path2node_>=0 && path2node_<(1<<3));
    assert(p==NULL?true:p->Child(path2node_)==this);
    parent=p;
    path2node=path2node_;
    depth=(parent==NULL?0:parent->depth+1);
    if(parent!=NULL) max_depth=parent->max_depth;
  }

  void SetChild(FMM_Node* c, int id) {
    assert(id<(1<<3));
    child[id]=c;
    if(c!=NULL) child[id]->SetParent(this,id);
  }

  FMM_Node * Colleague(int index) {
    return colleague[index];
  }

  void SetColleague(FMM_Node * node_, int index) {
    colleague[index]=node_;
  }

  real_t* Coord() {
    assert(coord!=NULL);
    return coord;
  }

};

}//end namespace

#endif //_PVFMM_FMM_NODE_HPP_
