#ifdef EIGHT
static inline void add_b(COOB *C,  COOB *A , COOB *B){
  
  Mat *c = C->value; Mat *a = A->value; Mat *b = B->value; 
  c[0 ] = add(a[0  ],b[0   ]);
  c[1 ] = add(a[1  ],b[1   ]);
  c[2 ] = add(a[2  ],b[2   ]);
  c[3 ] = add(a[3  ],b[3   ]);
  c[4 ] = add(a[4  ],b[4   ]);
  c[5 ] = add(a[5  ],b[5   ]);
  c[6 ] = add(a[6  ],b[6   ]);
  c[7 ] = add(a[7  ],b[7   ]);
  c[8 ] = add(a[8  ],b[8   ]);
  c[9 ] = add(a[9  ],b[9   ]);
  c[10] = add(a[10 ],b[10  ]);
  c[11] = add(a[11 ],b[11  ]);
  c[12] = add(a[12 ],b[12  ]);
  c[13] = add(a[13 ],b[13  ]);
  c[14] = add(a[14 ],b[14  ]);
  c[15] = add(a[15 ],b[15  ]);
  c[16] = add(a[16 ],b[16  ]);
  c[17] = add(a[17 ],b[17  ]);
  c[18] = add(a[18 ],b[18  ]);
  c[19] = add(a[19 ],b[19  ]);
  c[20] = add(a[20 ],b[20  ]);
  c[21] = add(a[21 ],b[21  ]);
  c[22] = add(a[22 ],b[22  ]);
  c[23] = add(a[23 ],b[23  ]);
  c[24] = add(a[24 ],b[24  ]);
  c[25] = add(a[25 ],b[25  ]);
  c[26] = add(a[26 ],b[26  ]);
  c[27] = add(a[27 ],b[27  ]);
  c[28] = add(a[28 ],b[28  ]);
  c[29] = add(a[29 ],b[29  ]);
  c[30] = add(a[30 ],b[30  ]);
  c[31] = add(a[31 ],b[31  ]);
  c[32] = add(a[32 ],b[32  ]);
  c[33] = add(a[33 ],b[33  ]);
  c[34] = add(a[34 ],b[34  ]);
  c[35] = add(a[35 ],b[35  ]);
  c[36] = add(a[36 ],b[36  ]);
  c[37] = add(a[37 ],b[37  ]);
  c[38] = add(a[38 ],b[38  ]);
  c[39] = add(a[39 ],b[39  ]);
  c[40] = add(a[40 ],b[40  ]);
  c[41] = add(a[41 ],b[41  ]);
  c[42] = add(a[42 ],b[42  ]);
  c[43] = add(a[43 ],b[43  ]);
  c[44] = add(a[44 ],b[44  ]);
  c[45] = add(a[45 ],b[45  ]);
  c[46] = add(a[46 ],b[46  ]);
  c[47] = add(a[47 ],b[47  ]);
  c[48] = add(a[48 ],b[48  ]);
  c[49] = add(a[49 ],b[49  ]);
  c[50] = add(a[50 ],b[50  ]);
  c[51] = add(a[51 ],b[51  ]);
  c[52] = add(a[52 ],b[52  ]);
  c[53] = add(a[53 ],b[53  ]);
  c[54] = add(a[54 ],b[54  ]);
  c[55] = add(a[55 ],b[55  ]);
  c[56] = add(a[56 ],b[56  ]);
  c[57] = add(a[57 ],b[57  ]);
  c[58] = add(a[58 ],b[58  ]);
  c[59] = add(a[59 ],b[59  ]);
  c[60] = add(a[60 ],b[60  ]);
  c[61] = add(a[61 ],b[61  ]);
  c[62] = add(a[62 ],b[62  ]);
  c[63] = add(a[63 ],b[63  ]);
}
static inline void mul_b(COOB *C,  COOB *A , COOB *B){
  Mat *c = C->value; Mat *a = A->value; Mat *b = B->value;
  Mat t ;
  for (int i=0; i<BM_; i++) 
    for (int j=0; j<BN_; j++){ 
      t = 0 ;
      
      
      t= add(t,mul(a[i*BN_+0],b[0*BN_+j]));
      t= add(t,mul(a[i*BN_+1],b[1*BN_+j]));
      t= add(t,mul(a[i*BN_+2],b[2*BN_+j]));
      t= add(t,mul(a[i*BN_+3],b[3*BN_+j]));
      t= add(t,mul(a[i*BN_+4],b[4*BN_+j]));
      t= add(t,mul(a[i*BN_+5],b[5*BN_+j]));
      t= add(t,mul(a[i*BN_+6],b[6*BN_+j]));
      t= add(t,mul(a[i*BN_+7],b[7*BN_+j]));
      
      c[i*BN_+j] = t;
    }
  
}
#endif

#ifdef FOUR
static inline void add_b(COOB *C,  COOB *A , COOB *B){
  
  Mat *c = C->value; Mat *a = A->value; Mat *b = B->value;
  c[0 ] = add(a[0  ],b[0   ]);
  c[1 ] = add(a[1  ],b[1   ]);
  c[2 ] = add(a[2  ],b[2   ]);
  c[3 ] = add(a[3  ],b[3   ]);
  c[4 ] = add(a[4  ],b[4   ]);
  c[5 ] = add(a[5  ],b[5   ]);
  c[6 ] = add(a[6  ],b[6   ]);
  c[7 ] = add(a[7  ],b[7   ]);
  c[8 ] = add(a[8  ],b[8   ]);
  c[9 ] = add(a[9  ],b[9   ]);
  c[10] = add(a[10 ],b[10  ]);
  c[11] = add(a[11 ],b[11  ]);
  c[12] = add(a[12 ],b[12  ]);
  c[13] = add(a[13 ],b[13  ]);
  c[14] = add(a[14 ],b[14  ]);
  c[15] = add(a[15 ],b[15  ]);
}
static inline void mul_b(COOB *C,  COOB *A , COOB *B){
  Mat *c = C->value; Mat *a = A->value; Mat *b = B->value;
  Mat t ;
  for (int i=0; i<BM_; i++) 
    for (int j=0; j<BN_; j++){ 
      t = 0 ;


      t= add(t,mul(a[i*BN_+0],b[0*BN_+j]));
      t= add(t,mul(a[i*BN_+1],b[1*BN_+j]));
      t= add(t,mul(a[i*BN_+2],b[2*BN_+j]));
      t= add(t,mul(a[i*BN_+3],b[3*BN_+j]));
      
      c[i*BN_+j] = t;
    }
  
}

#endif
#ifdef TWO
static inline void add_b(COOB *C,  COOB *A , COOB *B){
  
  Mat *c = C->value; Mat *a = A->value; Mat *b = B->value;
  
  for (int i=0; i<BM_; i++) 
    for (int j=0; j<BN_; j++) 
      c[i*BN_+j] = add(a[i*BN_+j],b[i*BN_+j]);
}
static inline void mul_b(COOB *C,  COOB *A , COOB *B){
  Mat *c = C->value; Mat *a = A->value; Mat *b = B->value;
  Mat t ;
  for (int i=0; i<BM_; i++) 
    for (int j=0; j<BN_; j++){ 
      t = 0 ;

      for (int k=0; k<BM_; k++) 
	t= add(t,mul(a[i*BN_+k],b[k*BN_+j]));
      
      c[i*BN_+j] = t;
    }
  
}


#endif
