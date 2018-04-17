#ifndef _PVFMM_MORTONID_HPP_
#define _PVFMM_MORTONID_HPP_

namespace pvfmm {

class MortonId {

 public:

  inline MortonId():x(0), y(0), z(0), depth(0) {}

  inline MortonId(MortonId m, uint8_t depth_):x(m.x), y(m.y), z(m.z), depth(depth_) {
    assert(depth<=MAX_DEPTH);
    uint64_t mask=~((((uint64_t)1)<<(MAX_DEPTH-depth))-1);
    x=x & mask;
    y=y & mask;
    z=z & mask;
  }

  inline MortonId(real_t x_f, real_t y_f, real_t z_f, uint8_t depth_=MAX_DEPTH) : depth(depth_) {
    static uint64_t max_int=((uint64_t)1)<<(MAX_DEPTH);
    x=(uint64_t)floor(x_f*max_int);
    y=(uint64_t)floor(y_f*max_int);
    z=(uint64_t)floor(z_f*max_int);
    assert(depth<=MAX_DEPTH);
    uint64_t mask=~((((uint64_t)1)<<(MAX_DEPTH-depth))-1);
    x=x & mask;
    y=y & mask;
    z=z & mask;
  }

  inline MortonId(real_t* coord, uint8_t depth_=MAX_DEPTH) : depth(depth_) {
    static uint64_t max_int=((uint64_t)1)<<(MAX_DEPTH);
    x=(uint64_t)floor(coord[0]*max_int);
    y=(uint64_t)floor(coord[1]*max_int);
    z=(uint64_t)floor(coord[2]*max_int);
    assert(depth<=MAX_DEPTH);
    uint64_t mask=~((((uint64_t)1)<<(MAX_DEPTH-depth))-1);
    x=x & mask;
    y=y & mask;
    z=z & mask;
  }

  inline unsigned int GetDepth() const {
    return depth;
  }

  inline void GetCoord(real_t* coord) {
    static uint64_t max_int=((uint64_t)1)<<(MAX_DEPTH);
    static real_t s=1.0/((real_t)max_int);
    coord[0]=x*s;
    coord[1]=y*s;
    coord[2]=z*s;
  }

  inline MortonId NextId() const {
    MortonId m=*this;
    uint64_t mask=((uint64_t)1)<<(MAX_DEPTH-depth);
    int i;
    for(i=depth; i>=0; i--) {
      m.x=(m.x^mask);
      if((m.x & mask))
        break;
      m.y=(m.y^mask);
      if((m.y & mask))
        break;
      m.z=(m.z^mask);
      if((m.z & mask))
        break;
      mask=(mask<<1);
    }
    if(i<0) i=0;
    m.depth=(uint8_t)i;
    return m;
  }

  inline MortonId getAncestor(uint8_t ancestor_level) const {
    MortonId m=*this;
    m.depth=ancestor_level;
    uint64_t mask=(((uint64_t)1)<<(MAX_DEPTH))-(((uint64_t)1)<<(MAX_DEPTH-ancestor_level));
    m.x=(m.x & mask);
    m.y=(m.y & mask);
    m.z=(m.z & mask);
    return m;
  }

  inline MortonId getDFD(uint8_t level=MAX_DEPTH) const {
    MortonId m=*this;
    m.depth=level;
    return m;
  }

  void NbrList(std::vector<MortonId>& nbrs, uint8_t level, int periodic) const {
    nbrs.clear();
    static unsigned int dim=3;
    static unsigned int nbr_cnt=powf(3, dim);
    static uint64_t maxCoord=(((uint64_t)1)<<(MAX_DEPTH));
    uint64_t mask=maxCoord-(((uint64_t)1)<<(MAX_DEPTH-level));
    uint64_t pX=x & mask;
    uint64_t pY=y & mask;
    uint64_t pZ=z & mask;
    MortonId mid_tmp;
    mask=(((uint64_t)1)<<(MAX_DEPTH-level));
    for(int i=0; i<nbr_cnt; i++) {
      int64_t dX = ((i/1)%3-1)*mask;
      int64_t dY = ((i/3)%3-1)*mask;
      int64_t dZ = ((i/9)%3-1)*mask;
      int64_t newX=(int64_t)pX+dX;
      int64_t newY=(int64_t)pY+dY;
      int64_t newZ=(int64_t)pZ+dZ;
      if(!periodic) {
        if(newX>=0 && newX<(int64_t)maxCoord)
          if(newY>=0 && newY<(int64_t)maxCoord)
            if(newZ>=0 && newZ<(int64_t)maxCoord) {
              mid_tmp.x=newX;
              mid_tmp.y=newY;
              mid_tmp.z=newZ;
              mid_tmp.depth=level;
              nbrs.push_back(mid_tmp);
            }
      } else {
        if(newX<0) newX+=maxCoord;
        if(newX>=(int64_t)maxCoord) newX-=maxCoord;
        if(newY<0) newY+=maxCoord;
        if(newY>=(int64_t)maxCoord) newY-=maxCoord;
        if(newZ<0) newZ+=maxCoord;
        if(newZ>=(int64_t)maxCoord) newZ-=maxCoord;
        mid_tmp.x=newX;
        mid_tmp.y=newY;
        mid_tmp.z=newZ;
        mid_tmp.depth=level;
        nbrs.push_back(mid_tmp);
      }
    }
  }

  std::vector<MortonId> Children() const {
    static int dim=3;
    static int c_cnt=(1UL<<dim);
    static uint64_t maxCoord=(((uint64_t)1)<<(MAX_DEPTH));
    std::vector<MortonId> child(c_cnt);
    uint64_t mask=maxCoord-(((uint64_t)1)<<(MAX_DEPTH-depth));
    uint64_t pX=x & mask;
    uint64_t pY=y & mask;
    uint64_t pZ=z & mask;
    mask=(((uint64_t)1)<<(MAX_DEPTH-(depth+1)));
    for(int i=0; i<c_cnt; i++) {
      child[i].x=pX+mask*((i/1)%2);
      child[i].y=pY+mask*((i/2)%2);
      child[i].z=pZ+mask*((i/4)%2);
      child[i].depth=(uint8_t)(depth+1);
    }
    return child;
  }

  inline int operator<(const MortonId& m) const {
    if(x==m.x && y==m.y && z==m.z) return depth<m.depth;
    uint64_t x_=(x^m.x);
    uint64_t y_=(y^m.y);
    uint64_t z_=(z^m.z);
    if((z_>x_ || ((z_^x_)<x_ && (z_^x_)<z_)) && (z_>y_ || ((z_^y_)<y_ && (z_^y_)<z_)))
      return z<m.z;
    if(y_>x_ || ((y_^x_)<x_ && (y_^x_)<y_))
      return y<m.y;
    return x<m.x;
  }

  inline int operator>(const MortonId& m) const {
    if(x==m.x && y==m.y && z==m.z) return depth>m.depth;
    uint64_t x_=(x^m.x);
    uint64_t y_=(y^m.y);
    uint64_t z_=(z^m.z);
    if((z_>x_ || ((z_^x_)<x_ && (z_^x_)<z_)) && (z_>y_ || ((z_^y_)<y_ && (z_^y_)<z_)))
      return z>m.z;
    if((y_>x_ || ((y_^x_)<x_ && (y_^x_)<y_)))
      return y>m.y;
    return x>m.x;
  }

  inline int operator==(const MortonId& m) const {
    return (x==m.x && y==m.y && z==m.z && depth==m.depth);
  }

  inline int isAncestor(MortonId const & other) const {
    return other.depth>depth && other.getAncestor(depth)==*this;
  }

  friend std::ostream& operator<<(std::ostream& out, const MortonId & mid);

 public:

  uint64_t x, y, z;
  uint8_t depth;

};

}//end namespace

#endif //_PVFMM_MORTONID_HPP_
