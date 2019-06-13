#ifndef intrinsics_h
#define intrinsics_h

namespace exafmm_t {
#ifdef __AVX512F__
  inline void matmult_8x8x2(double*& M_, double*& IN0, double*& IN1, double*& OUT0, double*& OUT1){
    double* in0_ = IN0;
    double* in1_ = IN1;

    __m512d out00, out10, out01, out11;
    __m512d one = _mm512_set1_pd(1.0);
    out00 = _mm512_load_pd(OUT0);
    out01 = _mm512_load_pd(OUT0+8);
    out10 = _mm512_load_pd(OUT1);
    out11 = _mm512_load_pd(OUT1+8);

    for(int i=0;i<8;i+=2){
      __m512d in00, in00_r, in10, in10_r;
      __m512d m00, mt0, mtt0;
      __m512d temp;

      // (in00, in01, in00, in01, in00, in01, in00, in01)
      in00 = _mm512_broadcast_f64x4(_mm256_broadcast_pd((const __m128d*)in0_));
      in10 = _mm512_broadcast_f64x4(_mm256_broadcast_pd((const __m128d*)in1_));
      // (in01, in00, in01, in00, in01, in00, in01, in00)
      in00_r = _mm512_permute_pd(in00, 85);
      in10_r = _mm512_permute_pd(in10, 85);
      // M column1 row0~4 * IN0 row1/IN1 row1
      // column 1, 4*2(real and imag)
      m00 = _mm512_load_pd(M_);
      // M shuffle (M0, M0, M2, M2, M4, M4, M6, M6)
      mt0 = _mm512_unpacklo_pd(m00, m00);
      // M shuffle (M1, M1, M3, M3, M5, M5, M7, M7)
      mtt0 = _mm512_unpackhi_pd(m00, m00);
      temp = _mm512_mul_pd(mt0, in00);
      out00 = _mm512_add_pd(out00, _mm512_fmaddsub_pd(temp, one, _mm512_mul_pd(mtt0, in00_r)));
      temp = _mm512_mul_pd(mt0, in10);
      out10 = _mm512_add_pd(out10, _mm512_fmaddsub_pd(temp, one, _mm512_mul_pd(mtt0, in10_r)));

      // M column1 row5~8 * IN0 row1/IN1 row1
      m00 = _mm512_load_pd(M_+8);
      mt0 = _mm512_unpacklo_pd(m00, m00);
      mtt0 = _mm512_unpackhi_pd(m00, m00);
      temp = _mm512_mul_pd(mt0, in00);
      out01 = _mm512_add_pd(out01, _mm512_fmaddsub_pd(temp, one, _mm512_mul_pd(mtt0, in00_r)));
      temp = _mm512_mul_pd(mt0, in10);
      out11 = _mm512_add_pd(out11, _mm512_fmaddsub_pd(temp, one, _mm512_mul_pd(mtt0, in10_r)));

      // M column2 row0~4 * IN0 row2/In1 row2
      in00 = _mm512_broadcast_f64x4(_mm256_broadcast_pd((const __m128d*)(in0_+2)));
      in10 = _mm512_broadcast_f64x4(_mm256_broadcast_pd((const __m128d*)(in1_+2)));
      in00_r = _mm512_permute_pd(in00, 85);
      in10_r = _mm512_permute_pd(in10, 85);
      m00 = _mm512_load_pd(M_+16);
      mt0 = _mm512_unpacklo_pd(m00, m00);
      mtt0 = _mm512_unpackhi_pd(m00, m00);
      temp = _mm512_mul_pd(mt0, in00);
      out00 = _mm512_add_pd(out00, _mm512_fmaddsub_pd(temp, one, _mm512_mul_pd(mtt0, in00_r)));
      temp = _mm512_mul_pd(mt0, in10);
      out10 = _mm512_add_pd(out10, _mm512_fmaddsub_pd(temp, one, _mm512_mul_pd(mtt0, in10_r)));

      // M column2 row5~8 * IN0 row2/IN1 row2
      m00 = _mm512_load_pd(M_+24);
      mt0 = _mm512_unpacklo_pd(m00, m00);
      mtt0 = _mm512_unpackhi_pd(m00, m00);
      temp = _mm512_mul_pd(mt0, in00);
      out01 = _mm512_add_pd(out01, _mm512_fmaddsub_pd(temp, one, _mm512_mul_pd(mtt0, in00_r)));
      temp = _mm512_mul_pd(mt0, in10);
      out11 = _mm512_add_pd(out11, _mm512_fmaddsub_pd(temp, one, _mm512_mul_pd(mtt0, in10_r)));

      M_+=32; // Jump to (column+2).
      in0_+=4;
      in1_+=4;
    }
    _mm512_store_pd(OUT0, out00);
    _mm512_store_pd(OUT0+8, out01);
    _mm512_store_pd(OUT1, out10);
    _mm512_store_pd(OUT1+8, out11);
  }
#elif __AVX__
  inline void matmult_8x8x2(double*& M_, double*& IN0, double*& IN1, double*& OUT0, double*& OUT1){
    __m256d out00,out01,out10,out11;
    __m256d out20,out21,out30,out31;
    double* in0__ = IN0;
    double* in1__ = IN1;
    out00 = _mm256_load_pd(OUT0);
    out01 = _mm256_load_pd(OUT1);
    out10 = _mm256_load_pd(OUT0+4);
    out11 = _mm256_load_pd(OUT1+4);
    out20 = _mm256_load_pd(OUT0+8);
    out21 = _mm256_load_pd(OUT1+8);
    out30 = _mm256_load_pd(OUT0+12);
    out31 = _mm256_load_pd(OUT1+12);
    for(int i2=0;i2<8;i2+=2){
      __m256d m00;
      __m256d ot00;
      __m256d mt0,mtt0;
      __m256d in00,in00_r,in01,in01_r;
      in00 = _mm256_broadcast_pd((const __m128d*)in0__);
      in00_r = _mm256_permute_pd(in00,5);
      in01 = _mm256_broadcast_pd((const __m128d*)in1__);
      in01_r = _mm256_permute_pd(in01,5);
      m00 = _mm256_load_pd(M_);
      mt0 = _mm256_unpacklo_pd(m00,m00);
      ot00 = _mm256_mul_pd(mt0,in00);
      mtt0 = _mm256_unpackhi_pd(m00,m00);
      out00 = _mm256_add_pd(out00,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in00_r)));
      ot00 = _mm256_mul_pd(mt0,in01);
      out01 = _mm256_add_pd(out01,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in01_r)));
      m00 = _mm256_load_pd(M_+4);
      mt0 = _mm256_unpacklo_pd(m00,m00);
      ot00 = _mm256_mul_pd(mt0,in00);
      mtt0 = _mm256_unpackhi_pd(m00,m00);
      out10 = _mm256_add_pd(out10,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in00_r)));
      ot00 = _mm256_mul_pd(mt0,in01);
      out11 = _mm256_add_pd(out11,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in01_r)));
      m00 = _mm256_load_pd(M_+8);
      mt0 = _mm256_unpacklo_pd(m00,m00);
      ot00 = _mm256_mul_pd(mt0,in00);
      mtt0 = _mm256_unpackhi_pd(m00,m00);
      out20 = _mm256_add_pd(out20,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in00_r)));
      ot00 = _mm256_mul_pd(mt0,in01);
      out21 = _mm256_add_pd(out21,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in01_r)));
      m00 = _mm256_load_pd(M_+12);
      mt0 = _mm256_unpacklo_pd(m00,m00);
      ot00 = _mm256_mul_pd(mt0,in00);
      mtt0 = _mm256_unpackhi_pd(m00,m00);
      out30 = _mm256_add_pd(out30,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in00_r)));
      ot00 = _mm256_mul_pd(mt0,in01);
      out31 = _mm256_add_pd(out31,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in01_r)));
      in00 = _mm256_broadcast_pd((const __m128d*) (in0__+2));
      in00_r = _mm256_permute_pd(in00,5);
      in01 = _mm256_broadcast_pd((const __m128d*) (in1__+2));
      in01_r = _mm256_permute_pd(in01,5);
      m00 = _mm256_load_pd(M_+16);
      mt0 = _mm256_unpacklo_pd(m00,m00);
      ot00 = _mm256_mul_pd(mt0,in00);
      mtt0 = _mm256_unpackhi_pd(m00,m00);
      out00 = _mm256_add_pd(out00,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in00_r)));
      ot00 = _mm256_mul_pd(mt0,in01);
      out01 = _mm256_add_pd(out01,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in01_r)));
      m00 = _mm256_load_pd(M_+20);
      mt0 = _mm256_unpacklo_pd(m00,m00);
      ot00 = _mm256_mul_pd(mt0,in00);
      mtt0 = _mm256_unpackhi_pd(m00,m00);
      out10 = _mm256_add_pd(out10,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in00_r)));
      ot00 = _mm256_mul_pd(mt0,in01);
      out11 = _mm256_add_pd(out11,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in01_r)));
      m00 = _mm256_load_pd(M_+24);
      mt0 = _mm256_unpacklo_pd(m00,m00);
      ot00 = _mm256_mul_pd(mt0,in00);
      mtt0 = _mm256_unpackhi_pd(m00,m00);
      out20 = _mm256_add_pd(out20,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in00_r)));
      ot00 = _mm256_mul_pd(mt0,in01);
      out21 = _mm256_add_pd(out21,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in01_r)));
      m00 = _mm256_load_pd(M_+28);
      mt0 = _mm256_unpacklo_pd(m00,m00);
      ot00 = _mm256_mul_pd(mt0,in00);
      mtt0 = _mm256_unpackhi_pd(m00,m00);
      out30 = _mm256_add_pd(out30,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in00_r)));
      ot00 = _mm256_mul_pd(mt0,in01);
      out31 = _mm256_add_pd(out31,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in01_r)));
      M_ += 32;
      in0__ += 4;
      in1__ += 4;
    }
    _mm256_store_pd(OUT0,out00);
    _mm256_store_pd(OUT1,out01);
    _mm256_store_pd(OUT0+4,out10);
    _mm256_store_pd(OUT1+4,out11);
    _mm256_store_pd(OUT0+8,out20);
    _mm256_store_pd(OUT1+8,out21);
    _mm256_store_pd(OUT0+12,out30);
    _mm256_store_pd(OUT1+12,out31);
  }
#elif __SSE3__
  inline void matmult_8x8x2(float*& M_, float*& IN0, float*& IN1, float*& OUT0, float*& OUT1){
    __m128 out00,out01,out10,out11;
    __m128 out20,out21,out30,out31;
    float* in0__ = IN0;
    float* in1__ = IN1;
    out00 = _mm_load_ps(OUT0);
    out01 = _mm_load_ps(OUT1);
    out10 = _mm_load_ps(OUT0+4);
    out11 = _mm_load_ps(OUT1+4);
    out20 = _mm_load_ps(OUT0+8);
    out21 = _mm_load_ps(OUT1+8);
    out30 = _mm_load_ps(OUT0+12);
    out31 = _mm_load_ps(OUT1+12);
    for(int i2=0;i2<8;i2+=2){
      __m128 m00;
      __m128 mt0,mtt0;
      __m128 in00,in00_r,in01,in01_r;
      in00 = _mm_castpd_ps(_mm_load_pd1((const double*)in0__));
      in00_r = _mm_shuffle_ps(in00,in00,_MM_SHUFFLE(2,3,0,1));
      in01 = _mm_castpd_ps(_mm_load_pd1((const double*)in1__));
      in01_r = _mm_shuffle_ps(in01,in01,_MM_SHUFFLE(2,3,0,1));
      m00 = _mm_load_ps(M_);
      mt0  = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(2,2,0,0));
      out00= _mm_add_ps   (out00,_mm_mul_ps( mt0,in00  ));
      mtt0 = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(3,3,1,1));
      out00= _mm_addsub_ps(out00,_mm_mul_ps(mtt0,in00_r));
      out01 = _mm_add_ps   (out01,_mm_mul_ps( mt0,in01  ));
      out01 = _mm_addsub_ps(out01,_mm_mul_ps(mtt0,in01_r));
      m00 = _mm_load_ps(M_+4);
      mt0  = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(2,2,0,0));
      out10= _mm_add_ps   (out10,_mm_mul_ps( mt0,in00  ));
      mtt0 = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(3,3,1,1));
      out10= _mm_addsub_ps(out10,_mm_mul_ps(mtt0,in00_r));
      out11 = _mm_add_ps   (out11,_mm_mul_ps( mt0,in01  ));
      out11 = _mm_addsub_ps(out11,_mm_mul_ps(mtt0,in01_r));
      m00 = _mm_load_ps(M_+8);
      mt0  = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(2,2,0,0));
      out20= _mm_add_ps   (out20,_mm_mul_ps( mt0,in00  ));
      mtt0 = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(3,3,1,1));
      out20= _mm_addsub_ps(out20,_mm_mul_ps(mtt0,in00_r));
      out21 = _mm_add_ps   (out21,_mm_mul_ps( mt0,in01  ));
      out21 = _mm_addsub_ps(out21,_mm_mul_ps(mtt0,in01_r));
      m00 = _mm_load_ps(M_+12);
      mt0  = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(2,2,0,0));
      out30= _mm_add_ps   (out30,_mm_mul_ps( mt0,  in00));
      mtt0 = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(3,3,1,1));
      out30= _mm_addsub_ps(out30,_mm_mul_ps(mtt0,in00_r));
      out31 = _mm_add_ps   (out31,_mm_mul_ps( mt0,in01  ));
      out31 = _mm_addsub_ps(out31,_mm_mul_ps(mtt0,in01_r));
      in00 = _mm_castpd_ps(_mm_load_pd1((const double*) (in0__+2)));
      in00_r = _mm_shuffle_ps(in00,in00,_MM_SHUFFLE(2,3,0,1));
      in01 = _mm_castpd_ps(_mm_load_pd1((const double*) (in1__+2)));
      in01_r = _mm_shuffle_ps(in01,in01,_MM_SHUFFLE(2,3,0,1));
      m00 = _mm_load_ps(M_+16);
      mt0  = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(2,2,0,0));
      out00= _mm_add_ps   (out00,_mm_mul_ps( mt0,in00  ));
      mtt0 = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(3,3,1,1));
      out00= _mm_addsub_ps(out00,_mm_mul_ps(mtt0,in00_r));
      out01 = _mm_add_ps   (out01,_mm_mul_ps( mt0,in01  ));
      out01 = _mm_addsub_ps(out01,_mm_mul_ps(mtt0,in01_r));
      m00 = _mm_load_ps(M_+20);
      mt0  = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(2,2,0,0));
      out10= _mm_add_ps   (out10,_mm_mul_ps( mt0,in00  ));
      mtt0 = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(3,3,1,1));
      out10= _mm_addsub_ps(out10,_mm_mul_ps(mtt0,in00_r));
      out11 = _mm_add_ps   (out11,_mm_mul_ps( mt0,in01 ));
      out11 = _mm_addsub_ps(out11,_mm_mul_ps(mtt0,in01_r));
      m00 = _mm_load_ps(M_+24);
      mt0  = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(2,2,0,0));
      out20= _mm_add_ps   (out20,_mm_mul_ps( mt0,in00  ));
      mtt0 = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(3,3,1,1));
      out20= _mm_addsub_ps(out20,_mm_mul_ps(mtt0,in00_r));
      out21 = _mm_add_ps   (out21,_mm_mul_ps( mt0,in01  ));
      out21 = _mm_addsub_ps(out21,_mm_mul_ps(mtt0,in01_r));
      m00 = _mm_load_ps(M_+28);
      mt0  = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(2,2,0,0));
      out30= _mm_add_ps   (out30,_mm_mul_ps( mt0,in00  ));
      mtt0 = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(3,3,1,1));
      out30= _mm_addsub_ps(out30,_mm_mul_ps(mtt0,in00_r));
      out31 = _mm_add_ps   (out31,_mm_mul_ps( mt0,in01  ));
      out31 = _mm_addsub_ps(out31,_mm_mul_ps(mtt0,in01_r));
      M_ += 32;
      in0__ += 4;
      in1__ += 4;
    }
    _mm_store_ps(OUT0,out00);
    _mm_store_ps(OUT1,out01);
    _mm_store_ps(OUT0+4,out10);
    _mm_store_ps(OUT1+4,out11);
    _mm_store_ps(OUT0+8,out20);
    _mm_store_ps(OUT1+8,out21);
    _mm_store_ps(OUT0+12,out30);
    _mm_store_ps(OUT1+12,out31);
  }
#else
  inline void matmult_8x8x2(real_t*& M_, real_t*& IN0, real_t*& IN1, real_t*& OUT0, real_t*& OUT1){
    // Generic code.
    real_t out_reg000, out_reg001, out_reg010, out_reg011;
    real_t out_reg100, out_reg101, out_reg110, out_reg111;
    real_t  in_reg000,  in_reg001,  in_reg010,  in_reg011;
    real_t  in_reg100,  in_reg101,  in_reg110,  in_reg111;
    real_t   m_reg000,   m_reg001,   m_reg010,   m_reg011;
    real_t   m_reg100,   m_reg101,   m_reg110,   m_reg111;
    //#pragma unroll
    for(int i1=0;i1<8;i1+=2){
      real_t* IN0_=IN0;
      real_t* IN1_=IN1;

      out_reg000=OUT0[ 0]; out_reg001=OUT0[ 1];
      out_reg010=OUT0[ 2]; out_reg011=OUT0[ 3];
      out_reg100=OUT1[ 0]; out_reg101=OUT1[ 1];
      out_reg110=OUT1[ 2]; out_reg111=OUT1[ 3];
      //#pragma unroll
      for(int i2=0;i2<8;i2+=2){
        m_reg000=M_[ 0]; m_reg001=M_[ 1];
        m_reg010=M_[ 2]; m_reg011=M_[ 3];
        m_reg100=M_[16]; m_reg101=M_[17];
        m_reg110=M_[18]; m_reg111=M_[19];

        in_reg000=IN0_[0]; in_reg001=IN0_[1];
        in_reg010=IN0_[2]; in_reg011=IN0_[3];
        in_reg100=IN1_[0]; in_reg101=IN1_[1];
        in_reg110=IN1_[2]; in_reg111=IN1_[3];

        out_reg000 += m_reg000*in_reg000 - m_reg001*in_reg001;
        out_reg001 += m_reg000*in_reg001 + m_reg001*in_reg000;
        out_reg010 += m_reg010*in_reg000 - m_reg011*in_reg001;
        out_reg011 += m_reg010*in_reg001 + m_reg011*in_reg000;

        out_reg000 += m_reg100*in_reg010 - m_reg101*in_reg011;
        out_reg001 += m_reg100*in_reg011 + m_reg101*in_reg010;
        out_reg010 += m_reg110*in_reg010 - m_reg111*in_reg011;
        out_reg011 += m_reg110*in_reg011 + m_reg111*in_reg010;

        out_reg100 += m_reg000*in_reg100 - m_reg001*in_reg101;
        out_reg101 += m_reg000*in_reg101 + m_reg001*in_reg100;
        out_reg110 += m_reg010*in_reg100 - m_reg011*in_reg101;
        out_reg111 += m_reg010*in_reg101 + m_reg011*in_reg100;

        out_reg100 += m_reg100*in_reg110 - m_reg101*in_reg111;
        out_reg101 += m_reg100*in_reg111 + m_reg101*in_reg110;
        out_reg110 += m_reg110*in_reg110 - m_reg111*in_reg111;
        out_reg111 += m_reg110*in_reg111 + m_reg111*in_reg110;

        M_+=32; // Jump to (column+2).
        IN0_+=4;
        IN1_+=4;
      }
      OUT0[ 0]=out_reg000; OUT0[ 1]=out_reg001;
      OUT0[ 2]=out_reg010; OUT0[ 3]=out_reg011;
      OUT1[ 0]=out_reg100; OUT1[ 1]=out_reg101;
      OUT1[ 2]=out_reg110; OUT1[ 3]=out_reg111;
      M_+=4-64*2; // Jump back to first column (row+2).
      OUT0+=4;
      OUT1+=4;
    }
  }
#endif
}
#endif
