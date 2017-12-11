#ifndef _PVFMM_FMM_NODE_HPP_
#define _PVFMM_FMM_NODE_HPP_

namespace pvfmm{

class FMM_Data{
 public:
  ~FMM_Data(){}
  Vector<Real_t> upward_equiv;
  Vector<Real_t> dnward_equiv;
};

class FMM_Node {

 public:
  int depth;
  int max_depth;
  int path2node;
  FMM_Node* parent;
  FMM_Node** child;
  int status;
  bool ghost;
  size_t max_pts;
  size_t node_id;
  long long weight;
  Real_t coord[3];
  FMM_Node * colleague[27];
  Vector<Real_t> pt_coord;
  Vector<Real_t> pt_value;
  Vector<size_t> pt_scatter;
  Vector<Real_t> src_coord;
  Vector<Real_t> src_value;
  Vector<size_t> src_scatter;
  Vector<Real_t> surf_coord;
  Vector<Real_t> surf_value;
  Vector<size_t> surf_scatter;
  Vector<Real_t> trg_coord;
  Vector<Real_t> trg_value;
  Vector<size_t> trg_scatter;
  size_t pt_cnt[2];
  std::vector<FMM_Node*> interac_list[Type_Count];

  FMM_Data* fmm_data;

  class NodeData {
    public:
     ~NodeData(){};
     void Clear(){}
     int max_depth;
     size_t max_pts;
     Vector<Real_t> coord;
     Vector<Real_t> value;
     Vector<Real_t> src_coord;
     Vector<Real_t> src_value;
     Vector<Real_t> surf_coord;
     Vector<Real_t> surf_value;
     Vector<Real_t> trg_coord;
     Vector<Real_t> trg_value;
  };

  FMM_Node() : depth(0), max_depth(MAX_DEPTH), parent(NULL), child(NULL), status(1),
	       ghost(false), weight(1) {
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

  void Initialize(FMM_Node* parent_, int path2node_, FMM_Node::NodeData* data_){
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
    Real_t coord_offset=((Real_t)1.0)/((Real_t)(((uint64_t)1)<<depth));
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
    NodeData* data=dynamic_cast<NodeData*>(data_);
    if(data_){
      max_pts=data->max_pts;
      pt_coord=data->coord;
      pt_value=data->value;
      src_coord=data->src_coord;
      src_value=data->src_value;
      surf_coord=data->surf_coord;
      surf_value=data->surf_value;
      trg_coord=data->trg_coord;
      trg_value=data->trg_value;
    }else if(parent){
      max_pts =parent->max_pts;
      SetGhost(parent->IsGhost());
    }
  }

  void NodeDataVec(std::vector<Vector<Real_t>*>& coord,
                   std::vector<Vector<Real_t>*>& value,
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
    int n=(1UL<<3);
    child=new FMM_Node* [n];
    for(int i=0;i<n;i++){
      child[i]=NewNode();
      child[i]->parent=this;
      child[i]->Initialize(this,i,NULL);
    }
    int nchld=(1UL<<3);
    if(!IsGhost()){
      std::vector<Vector<Real_t>*> pt_c;
      std::vector<Vector<Real_t>*> pt_v;
      std::vector<Vector<size_t>*> pt_s;
      NodeDataVec(pt_c, pt_v, pt_s);

      std::vector<std::vector<Vector<Real_t>*> > chld_pt_c(nchld);
      std::vector<std::vector<Vector<Real_t>*> > chld_pt_v(nchld);
      std::vector<std::vector<Vector<size_t>*> > chld_pt_s(nchld);
      for(size_t i=0;i<nchld;i++){
	Child(i)->NodeDataVec(chld_pt_c[i], chld_pt_v[i], chld_pt_s[i]);
      }

      Real_t* c=Coord();
      Real_t s=powf(0.5,depth+1);
      for(size_t j=0;j<pt_c.size();j++){
	if(!pt_c[j] || !pt_c[j]->Dim()) continue;
	Vector<Real_t>& coord=*pt_c[j];
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
	  Vector<Real_t>& vec=*pt_c[j];
	  size_t dof=vec.Dim()/npts;
          assert(dof>0);
          for(size_t i=0;i<nchld;i++){
            Vector<Real_t>& chld_vec=*chld_pt_c[i][j];
            chld_vec.Resize((cdata[i+1]-cdata[i])*dof);
            for (int k=cdata[i]*dof; k<cdata[i+1]*dof; k++) {
              chld_vec[k-cdata[i]*dof] = vec[k];
            }
          }
	  vec.Resize(0);
	}
	if(pt_v[j]){
	  Vector<Real_t>& vec=*pt_v[j];
	  size_t dof=vec.Dim()/npts;
          for(size_t i=0;i<nchld;i++){
            Vector<Real_t>& chld_vec=*chld_pt_v[i][j];
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
  }

  bool IsLeaf() {
    return child == NULL;
  }

  bool IsGhost() {
    return ghost;
  }

  void SetGhost(bool x) {
    ghost=x;
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
    assert(id<(1<<3));
    if(child==NULL) return NULL;
    return child[id];
  }

  FMM_Node* Parent(){
    return parent;
  }

  inline MortonId GetMortonId() {
    assert(coord);
    Real_t s=0.25/(1UL<<MAX_DEPTH);
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

  Real_t* Coord() {
    assert(coord!=NULL);
    return coord;
  }

};

}//end namespace

#endif //_PVFMM_FMM_NODE_HPP_
