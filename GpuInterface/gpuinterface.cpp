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

#include  <rocblas/rocblas.h>
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
#include <rocsparse/rocsparse.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector> 

//#include <pybind11/pybind11.h>

//namespace py = pybind11;

#define GPUS_  8

static int DEBUG = 0;
static rocblas_handle rochandle ;
static rocsparse_handle handle;
static rocsparse_mat_info info_csrmv[GPUS_]  = { nullptr,nullptr,nullptr,nullptr, \
					     nullptr,nullptr,nullptr,nullptr} ;
static rocsparse_mat_descr descrA[GPUS_]     = { nullptr,nullptr,nullptr,nullptr, \
					     nullptr,nullptr,nullptr,nullptr} ;


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


void init()
{
  ROCBLAS_CHECK(rocblas_create_handle(&rochandle));
  ROCSPARSE_CHECK(rocsparse_create_handle(&handle));  
}
void endit() {
  for (int i=0; i<GPUS_; i++)
    if (info_csrmv[i]) {
      ROCSPARSE_CHECK(rocsparse_destroy_mat_info(info_csrmv[i]));
      info_csrmv[i] = nullptr;  
      ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descrA[i]));
      descrA[i] = nullptr;
    }
  ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
  ROCBLAS_CHECK(rocblas_destroy_handle(rochandle));
}
void reset() {
  for (int i=0; i<GPUS_; i++)
    if (info_csrmv[i]) {
      ROCSPARSE_CHECK(rocsparse_destroy_mat_info(info_csrmv[i]));
      info_csrmv[i] = nullptr;
      ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descrA[i]));
      descrA[i] = nullptr;
    }
  
}



std::vector<double> csr_mv(int device_id,
       std::vector<rocsparse_int> hAptr,
       std::vector<rocsparse_int> hAcol,
       std::vector<double>        hAval,
       std::vector<double> x,
       std::vector<double> y,
       double halpha,
       double hbeta,
       int _info
       ) {

  rocsparse_mat_info  local_info = nullptr;
  rocsparse_mat_descr local_descrA = nullptr;

  rocsparse_int m = hAptr.size() -1;
  rocsparse_int n = hAcol.size();
  rocsparse_int nnz = hAval.size();


  
  hipDeviceProp_t devProp;

  //HIP_CHECK(hipGetDevice(&device_id));
  HIP_CHECK(hipSetDevice(device_id));
  HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
  if (DEBUG) std::cout << device_id <<" Device: " << devProp.name << std::endl;

  // Offload data to device
  rocsparse_int* dAptr = NULL;
  rocsparse_int* dAcol = NULL;
  double*        dAval = NULL;
  double*        dx    = NULL;
  double*        dy    = NULL;

  if (DEBUG) std::cout << "Allocate " << std::endl;
  
  HIP_CHECK(hipMalloc((void**)&dAptr, sizeof(rocsparse_int) * (m + 1)));
  HIP_CHECK(hipMalloc((void**)&dAcol, sizeof(rocsparse_int) * nnz));
  HIP_CHECK(hipMalloc((void**)&dAval, sizeof(double) * nnz));
  HIP_CHECK(hipMalloc((void**)&dx, sizeof(double) * n));
  HIP_CHECK(hipMalloc((void**)&dy, sizeof(double) * m));
  
  if (DEBUG) std::cout << "Move " << std::endl;
  HIP_CHECK(
	    hipMemcpy(dAptr, hAptr.data(), sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice));
  if (DEBUG) std::cout << "Move haptr" << std::endl;
  HIP_CHECK(hipMemcpy(dAcol, hAcol.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
  if (DEBUG) std::cout << "Move hacol" << std::endl;
  HIP_CHECK(hipMemcpy(dAval, hAval.data(), sizeof(double) * nnz, hipMemcpyHostToDevice));
  if (DEBUG) std::cout << "Move haval" << std::endl;
  if (DEBUG) std::cout << "Move "<<x.size()<< std::endl;
  HIP_CHECK(hipMemcpy(dx, x.data(), sizeof(double) * n, hipMemcpyHostToDevice));
  if (DEBUG) std::cout << "Move "<<y.size()<< std::endl;
  HIP_CHECK(hipMemcpy(dy, y.data(), sizeof(double) * m, hipMemcpyHostToDevice));
  


  // Create meta data
  if (_info != 0) { 
    if (info_csrmv[device_id] == nullptr) {
      ROCSPARSE_CHECK(rocsparse_create_mat_info(&info_csrmv[device_id]));
      ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descrA[device_id]));
      // Analyse CSR matrix
      ROCSPARSE_CHECK(
	  rocsparse_dcsrmv_analysis(
	      handle, rocsparse_operation_none, m, n, nnz,
	      descrA[device_id],
	      dAval, dAptr, dAcol,
	      info_csrmv[device_id]
	  )
      );
    }
  }
  if (info_csrmv[device_id] != nullptr)
    {
      local_descrA = descrA[device_id];
      local_info   = info_csrmv[device_id];
    }
  else {
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&local_descrA));
  } 
  if (DEBUG)std::cout << device_id << " INFO " << local_info << " D " << local_descrA << std::endl;
  if (DEBUG)   std::cout << "Run " << std::endl;
  // Start time measurement
  
  auto start = std::chrono::high_resolution_clock::now();
  ROCSPARSE_CHECK(rocsparse_dcsrmv(handle,
				   rocsparse_operation_none,
				   m,
				   n,
				   nnz,
				   &halpha,
				   local_descrA,
				   dAval,
				   dAptr,
				   dAcol,
				   local_info,
				   dx,
				   &hbeta,
				   dy));

  

  if (DEBUG) std::cout << "Sync " << std::endl;
  // Device synchronization
  HIP_CHECK(hipDeviceSynchronize());
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "\t Time Kernel "  << duration.count()/1000000.0 << std::endl;
  HIP_CHECK(hipMemcpy(y.data(), dy, sizeof(double) * m, hipMemcpyDeviceToHost));
  // clean up descriptor
  if (info_csrmv[device_id] == nullptr) {ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(local_descrA));}

  if (DEBUG) std::cout << "Free " << std::endl;

  // Clear up on device
  HIP_CHECK(hipFree(dAptr));
  HIP_CHECK(hipFree(dAcol));
  HIP_CHECK(hipFree(dAval));
  HIP_CHECK(hipFree(dx));
  HIP_CHECK(hipFree(dy));
  return y;
}	   




std::vector<double> gemm(int device_id,
			 std::vector<double> hc, int ldc, 
			 std::vector<double> ha, int lda, 
			 std::vector<double> hb, int ldb,
			 double alpha,
			 double beta
			 ) {


  rocsparse_int m = hc.size()/ldc;
  rocsparse_int n = ldc;
  rocsparse_int k = lda;

  int size_a = ha.size();
  int size_b = hb.size();
  int size_c = hc.size();
  

  
  auto start = std::chrono::high_resolution_clock::now();

  rocblas_operation transa = rocblas_operation_none, transb = rocblas_operation_transpose;
  hipDeviceProp_t devProp;

  //HIP_CHECK(hipGetDevice(&device_id));
  HIP_CHECK(hipSetDevice(device_id));
  HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
  if (DEBUG) std::cout << device_id <<" Device: " << devProp.name << std::endl;

  // allocate memory on device
  double *da, *db, *dc;
  HIP_CHECK(hipMalloc(&da, size_a * sizeof(double)));
  HIP_CHECK(hipMalloc(&db, size_b * sizeof(double)));
  HIP_CHECK(hipMalloc(&dc, size_c * sizeof(double)));
  
  // copy matrices from host to device
  HIP_CHECK(hipMemcpy(da, ha.data(), sizeof(double) * size_a, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(db, hb.data(), sizeof(double) * size_b, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dc, hc.data(), sizeof(double) * size_c, hipMemcpyHostToDevice));
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "\t Data and Initialization Kernel "  << duration.count()/1000000.0 << std::endl;

  start = std::chrono::high_resolution_clock::now();

  ROCBLAS_CHECK(
		rocblas_dgemm(rochandle, transa, transb, m, n, k, &alpha, da, lda, db, ldb, &beta, dc, ldc));
  
  HIP_CHECK(hipDeviceSynchronize());
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "\t Time Kernel "  << duration.count()/1000000.0 << std::endl;
  // copy output from device to CPU
  start = std::chrono::high_resolution_clock::now();
  HIP_CHECK(hipMemcpy(hc.data(), dc, sizeof(double) * size_c, hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(da));
  HIP_CHECK(hipFree(db));
  HIP_CHECK(hipFree(dc));
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "\t Read data from  Kernel "  << duration.count()/1000000.0 << std::endl;
  
  if (DEBUG) std::cout  << "m, n, k, lda, ldb, ldc = " << m << ", " << n << ", " << k << ", " << lda
	       << ", " << ldb << ", " << ldc << std::endl;
  
  return hc;
}	   
std::vector<double> coo_mv(int device_id,
       std::vector<rocsparse_int> hrow,
       std::vector<rocsparse_int> hcol,
       std::vector<double>        hAval,
       std::vector<double> x,
       std::vector<double> y,
       double halpha,
       double hbeta
       ) {


  rocsparse_mat_descr local_descrA = nullptr;

  rocsparse_int m = y.size();
  rocsparse_int n = x.size();
  rocsparse_int nnz = hAval.size();



  hipDeviceProp_t devProp;

  //HIP_CHECK(hipGetDevice(&device_id));
  HIP_CHECK(hipSetDevice(device_id));
  HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
  if (DEBUG) std::cout << device_id <<" Device: " << devProp.name << std::endl;

  // Offload data to device
  rocsparse_int* drow = NULL;
  rocsparse_int* dcol = NULL;
  double*        dAval = NULL;
  double*        dx    = NULL;
  double*        dy    = NULL;

  if (DEBUG) std::cout << "Allocate " << std::endl;

  HIP_CHECK(hipMalloc((void**)&drow, sizeof(rocsparse_int) * (nnz)));
  HIP_CHECK(hipMalloc((void**)&dcol, sizeof(rocsparse_int) * nnz));
  HIP_CHECK(hipMalloc((void**)&dAval, sizeof(double) * nnz));
  HIP_CHECK(hipMalloc((void**)&dx, sizeof(double) * n));
  HIP_CHECK(hipMalloc((void**)&dy, sizeof(double) * m));
  
  if (DEBUG) std::cout << "Move " << std::endl;
  HIP_CHECK(
	    hipMemcpy(drow, hrow.data(), sizeof(rocsparse_int) * (nnz), hipMemcpyHostToDevice)
	    );
  if (DEBUG) std::cout << "Move haptr" << std::endl;
  HIP_CHECK(hipMemcpy(dcol, hcol.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
  if (DEBUG) std::cout << "Move hacol" << std::endl;
  HIP_CHECK(hipMemcpy(dAval, hAval.data(), sizeof(double) * nnz, hipMemcpyHostToDevice));
  if (DEBUG) std::cout << "Move haval" << std::endl;
  if (DEBUG) std::cout << "Move "<<x.size()<< std::endl;
  HIP_CHECK(hipMemcpy(dx, x.data(), sizeof(double) * n, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dy, y.data(), sizeof(double) * m, hipMemcpyHostToDevice));
  

  if (DEBUG) std::cout << "Description " << std::endl;
  // Matrix descriptor

  ROCSPARSE_CHECK(rocsparse_create_mat_descr(&local_descrA));
  
  if (DEBUG)   std::cout << "Run " << std::endl;
  
  auto start = std::chrono::high_resolution_clock::now();
  ROCSPARSE_CHECK(rocsparse_dcoomv(handle,
				   rocsparse_operation_none,
				   m,
				   n,
				   nnz,
				   &halpha,
				   local_descrA,
				   dAval,
				   drow,
				   dcol,
				   dx,
				   &hbeta,
				   dy));

  

  if (DEBUG) std::cout << "Sync " << std::endl;
  // Device synchronization
  HIP_CHECK(hipDeviceSynchronize());
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "\t Time Kernel "  << duration.count()/1000000.0 << std::endl;
  if (DEBUG) std::cout << "Sync " << std::endl;
  HIP_CHECK(hipMemcpy(y.data(), dy, sizeof(double) * m, hipMemcpyDeviceToHost));
  // clean up descriptor
  ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(local_descrA));

  if (DEBUG) std::cout << "Free " << std::endl;

  // Clear up on device
  HIP_CHECK(hipFree(drow));
  HIP_CHECK(hipFree(dcol));
  HIP_CHECK(hipFree(dAval));
  HIP_CHECK(hipFree(dx));
  HIP_CHECK(hipFree(dy));
  return y;
}	   

namespace py = pybind11;

PYBIND11_MODULE(rocmgpu, m) {
  m.doc() = "pybind11 example plugin"; // optional module docstring
  
  m.def("csr_mv", &csr_mv, "CSR MV",
	py::arg("device_id") ,
	py::arg("hAptr"),
	py::arg("hAcol"),
	py::arg("hAval"),
	py::arg("x"),
	py::arg("y"),
	py::arg("halpha"),
	py::arg("hbeta"),
	py::arg("_info") = 0
	);
  m.def("coo_mv", &coo_mv, "COO MV",
	py::arg("device_id") ,
	py::arg("hrow"),
	py::arg("hcol"),
	py::arg("hAval"),
	py::arg("x"),
	py::arg("y"),
	py::arg("halpha"),   
	py::arg("hbeta")
	);
  m.def("gemm", &gemm, "GEMM",
	py::arg("device_id") ,
	py::arg("c"),py::arg("ldc"),
	py::arg("a"),py::arg("lda"),
	py::arg("b"),py::arg("ldb"),
	py::arg("alpha"),
	py::arg("beta")
	);
	
  m.def("reset",reset);
  m.def("init", init);
  m.def("endit", endit);
}




