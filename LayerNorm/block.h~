/**
 * Like any good program we need to start from somewhere.  Basic
 * definition of operations, we will use in layer norm operations.
 */




#include<math.h>
typedef float   Mat;

// matrix and submatrix (Partition)
struct sub_matrix  {
  int M;  // Phisical dimension Rows
  int N;  // Phisical dimension Cols   
  int m;  // Logical  dimension < M
  int n;  // Logical  dimension< N
  Mat *val;
  
} ;
typedef struct sub_matrix SUBMATRIX;

// Vector
struct sub_vector  {
  int M;  // Phisical dimension Rows
  int m;  // Logical  dimension < M
  Mat *val;
} ;
typedef struct sub_vector SUBVECTOR;


 

#define add(a,b) ((a)+(b))
#define mul(a,b) ((a)*(b))
#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))
#define ceil_div(a,b) ((a)/(b) + (((a)%(b)!=1)?1:0))


#define EM(C, i, j) C.val[(i)*C.N+(j)]
#define EV(C, i)    C.val[(i)]




// + Matrices
static inline void add_m(SUBMATRIX C, SUBMATRIX A , SUBMATRIX B){
  
  int M  = min(min(A.m,B.m),C.m);
  int N  = min(min(A.n,B.n),C.n);


  /* the computation is based on the logical sizes */
  for (int i=0; i<M; i++) 
    for (int j=0; j<N; j++) 
      EM(C, i,j) = add( EM(A,i,j), EM(B, i,j) );
}

// * Matrices
static inline void mul_m(SUBMATRIX C, SUBMATRIX A , SUBMATRIX B){

  int M  = min(A.m,C.m);
  int N  = min(B.n,C.n);
  int K  = min(B.m,A.n);

  Mat t ;
  for (int i=0; i<M; i++) 
    for (int j=0; j<N; j++){ 
      t = 0 ;

      for (int k=0; k<K; k++) 
	t= add(t,mul(EM(A,i,k),EM(B,k,j)));
      
      EM(C,i,j) = t;
    }
  
}

static inline void add_col(SUBMATRIX C,SUBMATRIX A, SUBVECTOR B){
  
  int M  = min(A.m,C.m);
  int N  = min(min(A.n,B.m),C.n);

  /* the computation is based on the logical sizes */
  for (int i=0; i<M; i++) 
    for (int j=0; j<N; j++) 
      EM(C, i,j) = add( EM(A,i,j), EV(B, j) );
}

static inline void mul_col(SUBMATRIX C,SUBMATRIX A, SUBVECTOR B){
  
  int M  = min(A.m,C.m);
  int N  = min(min(A.n,B.m),C.n);

  /* the computation is based on the logical sizes */
  for (int i=0; i<M; i++) 
    for (int j=0; j<N; j++) 
      EM(C, i,j) = mul( EM(A,i,j), EV(B, j) );
      
}
static inline void mul_row(SUBMATRIX C,SUBMATRIX A, SUBVECTOR B){
  
  int M  = min(A.m,min(B.m,C.m));
  int N  = min(A.n,C.n);

  /* the computation is based on the logical sizes */
  for (int i=0; i<M; i++) {
    Mat t = EV(B, i);
    
    for (int j=0; j<N; j++) 
      EM(C, i,j) = mul( EM(A,i,j), t);
  }
}






typedef SUBMATRIX  (*LayerNormFunction)( SUBMATRIX A,  SUBVECTOR gamma, SUBVECTOR  beta);

typedef struct operands_ TOperands;

struct operands_ { 
  int  pi;
  LayerNormFunction m;  // LayerNorm(c,gamma, beta) 
  SUBMATRIX  c;
  SUBVECTOR  g;
  SUBVECTOR  b;
} ;


#ifndef LAYERNORM
#define LAYERNORM 1
extern SUBMATRIX LN12(SUBMATRIX A,  SUBVECTOR gamma, SUBVECTOR  beta);
extern SUBMATRIX LNS(SUBMATRIX A,  SUBVECTOR gamma, SUBVECTOR  beta);
extern SUBMATRIX LNS4(SUBMATRIX A,  SUBVECTOR gamma, SUBVECTOR  beta);
extern SUBMATRIX LN(SUBMATRIX A,  SUBVECTOR gamma, SUBVECTOR  beta);
extern SUBMATRIX LN_P(SUBMATRIX A,  SUBVECTOR gamma, SUBVECTOR  beta, int P);
extern void print_m(SUBMATRIX A);
extern void print_v(SUBVECTOR A);
extern void CREATE_V (SUBVECTOR *A, int M);
extern void init_v (SUBVECTOR A);
extern void DESTROY_V(SUBVECTOR *A);
extern void CREATE_S (SUBMATRIX *A,int M,int N);
extern void init_s (SUBMATRIX A);
extern void DESTROY_S(SUBMATRIX *A);
extern void  copy_s(SUBMATRIX A, SUBMATRIX B);
extern void  compare_s(SUBMATRIX A, SUBMATRIX B);

#endif
