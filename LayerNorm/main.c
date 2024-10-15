
//#define GRAPH_PATH 1


#include <pthread.h>
#define _GNU_SOURCE
#include <sched.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>


#include"block.h"

#define GETTIME

#ifdef GETTIME
#include <sys/time.h>
struct timeval _t1,_t2;
double duration;

#define START_CLOCK   gettimeofday(&_t1,NULL ); 
#define END_CLOCK   gettimeofday(&_t2,NULL);   duration = (_t2.tv_sec-_t1.tv_sec)+ (double)(_t2.tv_usec-_t1.tv_usec)/1000000;//    printf("----------> get time %e sec<------\n",duration); 
#endif /*  GETTIME */


#ifdef CLOCK
#include <time.h>
clock_t _t1,_t2;
double duration;

#define START_CLOCK   _t1 = clock(); 
#define END_CLOCK     _t2 = clock(); duration =  ((double)(_t2-_t1))/CLOCKS_PER_SEC; \
  //  printf("clock time %e s \n", duration); 
#endif

#define MEASURE(X,T)   { START_CLOCK;  X;   END_CLOCK; printf("COLD %f \n", duration); \
    START_CLOCK;  for (int t=0; t<T; t++)  X;   END_CLOCK;		\
  }
inline static double OPS(SUBMATRIX A) {
  
  int pass1 = 3*A.N * A.M ;
  int pass2 = 4*A.N * A.M ;
  return (double)(pass1+ pass2)/1000000000;
}


int main(int argc, char **argv)
{
  int n, m,times;
  SUBMATRIX A,B,C; 
  SUBVECTOR gamma,beta;
  double reg, opt, opt_8;
  //printf("Give matrix size (M):\n");
  scanf("%d",&m);
  //printf("(M) \n");
  scanf("%d",&n);
  

  scanf("%d",&times);

  //printf("CREATE A\n");
  CREATE_S(&A,m,n);
  init_s(A);
  //printf("LNSIZE,%d\n",n );
  //print_m(A);
  //printf("CREATE B\n");
  CREATE_S(&B,m,n);
  copy_s(B, A);
  //printf("CREATE C\n");
  CREATE_S(&C,m,n);
  copy_s(C, A);
  //print_m(A);
  //printf("CREATE Gamma\n");

  CREATE_V(&gamma,n);
  init_v(gamma);
  //print_v(gamma);
  //printf("CREATE Beta\n");
  CREATE_V(&beta,n);
  //init_v(beta);
  //print_v(beta);

  //printf("LNS4 \n");
  MEASURE((A = LNS4(A,gamma,beta)),times);
  reg =  OPS(A)/(duration/times);
  //printf("LNS4, %f\n", OPS(A)/(duration/times)) ;

  //  print_m(A);


  //printf("LN12 \n");
  MEASURE((B = LN12(B,gamma,beta)),times);
  opt = OPS(B)/(duration/times);
    //printf("LN12,%f\n", OPS(B)/(duration/times)) ;
  //print_m(B);


  //printf("LN P \n");
  MEASURE((C = LN_P(C,gamma,beta,8)),(times));
  opt_8 = OPS(C)/(duration/times);
  printf("LN,%d,%f,%f,%f \n", n, reg,opt,opt_8);
  //printf("LN_%d,%f\n", 8, OPS(C)/(duration/times)) ;
  //print_m(C);
  compare_s(A,C);
  compare_s(A,B);  
  //printf("DESTROY beta\n");    
  DESTROY_V(&beta);
  //printf("DESTROY gamma\n");
  DESTROY_V(&gamma);
  //printf("DESTROY C\n");
  
  DESTROY_S(&C);
  //printf("DESTROY B\n");
  
  DESTROY_S(&B);
  //printf("DESTROY A\n");

  DESTROY_S(&A);
  return 0;
}
