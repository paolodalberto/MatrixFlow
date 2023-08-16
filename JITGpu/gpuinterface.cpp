/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#define __HIP_PLATFORM_AMD__
#include <chrono> //#include "utils.hpp"
//using namespace std::chrono;
#include <hip/hip_runtime_api.h>
#include <iomanip>
#include <iostream>
#include  <rocblas/rocblas.h>
//#include <rocsparse/rocsparse.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector> 

//#include <pybind11/pybind11.h>

//namespace py = pybind11;

#define GPUS_  8

static int DEBUG = 1;


#define HIP_CHECK(stat)						\
  {									\
    if(stat != hipSuccess)						\
      std::cerr << "Error: hip error in line " << __LINE__ << std::endl; \
  }

static inline void  ROCBLAS_CHECK(rocblas_status  stat)	
{						
  if(stat != rocblas_status_success) { 		
    std::cerr << stat << " Error: rocblas error in line " << " " << __LINE__ << std::endl;
    switch (stat) {
    case rocblas_status_success:
      std::cerr << "success." << std::endl;
      break;
    case rocblas_status_invalid_handle:
      std::cerr << "handle not initialized, invalid or null." << std::endl;
      break;
    case rocblas_status_not_implemented:
      std::cerr << "function is not implemented." << std::endl;
      break;

    case rocblas_status_invalid_pointer:
      std::cerr << "invalid pointer parameter." << std::endl;
      break;
      
    case rocblas_status_invalid_size:
      std::cerr << "invalid size parameter." << std::endl;
    break;

    case rocblas_status_memory_error:
      std::cerr << "failed memory allocation, copy, dealloc." << std::endl;
    break;

    case rocblas_status_internal_error:
      std::cerr << "other internal library failure." << std::endl;
    break;

    case rocblas_status_invalid_value:
      std::cerr << "invalid value parameter." << std::endl;
    break;



    };
    
  }
}
/*
static inline void  ROCSPARSE_CHECK(rocsparse_status  stat)	
{						
  if(stat != rocsparse_status_success) { 		
    std::cerr << stat << " Error: rocsparse error in line " << " " << __LINE__ << std::endl;
    switch (stat) {
    case rocsparse_status_success:
      std::cerr << "success." << std::endl;
      break;
    case rocsparse_status_invalid_handle:
      std::cerr << "handle not initialized, invalid or null." << std::endl;
      break;
    case rocsparse_status_not_implemented:
      std::cerr << "function is not implemented." << std::endl;
      break;

    case rocsparse_status_invalid_pointer:
      std::cerr << "invalid pointer parameter." << std::endl;
      break;
      
    case rocsparse_status_invalid_size:
      std::cerr << "invalid size parameter." << std::endl;
    break;

    case rocsparse_status_memory_error:
      std::cerr << "failed memory allocation, copy, dealloc." << std::endl;
    break;

    case rocsparse_status_internal_error:
      std::cerr << "other internal library failure." << std::endl;
    break;

    case rocsparse_status_invalid_value:
      std::cerr << "invalid value parameter." << std::endl;
    break;

    case rocsparse_status_arch_mismatch:
      std::cerr << "device arch is not supported." << std::endl;
    break;

    case rocsparse_status_zero_pivot:
      std::cerr << "encountered zero pivot." << std::endl;
    break;

    case rocsparse_status_not_initialized:
      std::cerr << "descriptor has not been initialized." << std::endl;
    break;
    
    case rocsparse_status_type_mismatch:
      std::cerr << "index types do not match." << std::endl;
    break;

    case rocsparse_status_requires_sorted_storage:
      std::cerr << "sorted storage required." << std::endl;
      break;

    };
    
  }
}
*/
namespace py = pybind11;

#include "codextype.h"
#ifndef T
typedef TT double
#endif

std::vector<TT> fastgemm(int gpu,
	     std::vector<TT> a ,
	     std::vector<TT> b, 
	     std::vector<TT> c
	     );


PYBIND11_MODULE(one, m) {
  m.doc() = "pybind11 example plugin"; // optional module docstring
  
  m.def("fastgemm", &fastgemm, "GEMM",
	py::arg("gpu"),
	py::arg("a"),
	py::arg("b"),
	py::arg("c")
	);

}

void Print(const TT* A, int lda, int M, int N) {
  for (int m=0; m<M ; m++) { 
    for (int n=0; n<N ; n++)
      std::cout << A[m*lda+n] << " "; 
    std::cout << "\n ";  
  }
  std::cout << "\n ";  
}


/*
int rocblas_dgeam(int handle, 
		  char transA, 
		  char transB, 
		  int M, 
		  int N, 
		  const TT *alpha, 
		  const TT *A, 
		  int lda, 
		  const TT *B, 
		  int ldb, 
		  const TT *beta, 
		  TT *C, 
		  int ldc) {


  if (DEBUG) std::cout << "GEMA " << M << " " << N << "\n" ;
  if (DEBUG) std::cout << "alpha " << *alpha << " beta " << *beta <<"\n" ;
  if (DEBUG) Print(A,lda,M,N);
  if (DEBUG) Print(B,ldb,M,N);

  for (int m=0; m<M; m++)
    for (int n=0; n<N; n++) 
      C[m*ldc+n] =  (*alpha) * A[m*lda+n] + (*beta)*B[m*ldb+n];
  if (DEBUG) Print(C,ldc,M,N);
  return 1;

}
int rocblas_dgemm(int handle,  
		  char transA, 
		  char transB, 
		  int M, 
		  int N, 
		  int K, 
		  const TT *alpha, 
		  const TT *A, 
		  int lda, 
		  const TT *B, 
		  int ldb, 
		  const TT *beta, 
		  TT *C, 
		  int ldc) {

  TT temp =0;
  if (DEBUG)  {
    std::cout << "GEMM " << M << " " << N << " " << K << "\n" ;
    std::cout << "alpha " << *alpha << " beta " << *beta <<"\n" ;
  }
  if (DEBUG) Print(A,lda,M,K);
  if (DEBUG) Print(B,ldb,K,N);
  for (int m=0; m<M; m++)
    for (int n=0; n<N; n++) { 
      temp = 0;
      for (int k=0; k<K; k++) 
	 temp +=  (*alpha) * A[m*lda+k]*B[k*ldb+n];
      C[m*ldc+n] = temp; 
    }
  if (DEBUG) { 
    Print(C,ldc,M,N);
  }
  return 1;

}
*/
#include "codex.h"


