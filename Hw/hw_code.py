




class CPU:
    def __init__(self):
        self.MAGE ="""
        int rocblas_dgeam( 
		  char transA, 
		  char transB, 
		  int M, 
		  int N, 
		  const TT *alpha, 
		  const TT *A, 
		  int lda, 
		  const TT *beta, 
	          const TT *B, 
		  int ldb, 
		  TT *C, 
		  int ldc) {


        if (DEBUG) std::cout << "GEMA " << M << " " << N << "\\n" ;
        if (DEBUG) std::cout << "alpha " << *alpha << " beta " << *beta <<"\\n" ;
        if (DEBUG) Print(A,lda,M,N);
        if (DEBUG) Print(B,ldb,M,N);

          for (int m=0; m<M; m++)
            for (int n=0; n<N; n++) 
	      C[m*ldc+n] =  (*alpha) * A[m*lda+n] + (*beta)*B[m*ldb+n];
	  if (DEBUG) Print(C,ldc,M,N);
	  return 1;
	
	}
	int rocblas_dgemm(char transA, 
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
	    std::cout << "GEMM " << M << " " << N << " " << K << "\\n" ;
	    std::cout << "alpha " << *alpha << " beta " << *beta <<"\\n" ;
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
        """
        self.INTRO = ""
        self.TAIL = ""

        self.GEMM = """
    rocblas_status rocblas_dgemm(
        rocblas_operation transA, 
        rocblas_operation transB, 
        rocblas_int m, 
        rocblas_int n, 
        rocblas_int k, 
        const double *alpha, 
        const double *A, 
        rocblas_int lda, 
        const double *B, 
        rocblas_int ldb, 
        const double *beta, 
        double *C, 
        rocblas_int ldc) """

        ## handle, M, N, K, alpha, A,  LDA=K, B, LDB=N, beta,C, LDC=N 
        self.GEMM_i = "## handle, M, N, K, alpha, A,  LDA=K, B, LDB=N, beta,C, LDC=N "
        self.GEMM_x = "" + \
            "rocblas_dgemm( transa, transb, %d, " + \
            "%d, %d,  %s, %s, %d, %s, %d,  %s, %s, %d); "

        self.GEMA = """
    rocblas_status rocblas_dgeam(
        rocblas_operation transA, 
        rocblas_operation transB, 
        rocblas_int m, 
        rocblas_int n, 
        const double *alpha, 
        const double *A, 
        rocblas_int lda, 
        const double *beta, 
        const double *B, 
        rocblas_int ldb, 
        double *C, 
        rocblas_int ldc) """

        ## handle, M, N, K, alpha, A, beta, LDA=K, B, LDB=N, beta,C, LDC=N 
        self.GEMA_i = "##  M, N, alpha, A, lda, beta,B, LDB=N, C, LDC=N "
        
        self.GEMA_x = "" + \
            "rocblas_dgeam( transa, transb, %d, %d,  " +\
            "%s, %s, %d,   %s, %s, %d,  %s, %d); "


        self.MALLOC = "HIP_CHECK(hipMalloc(&%s, %d*%d* sizeof(%s)));"
        self.FREE   = "HIP_CHECK(hipFree(%s));" 
        
    def __str__(self):
        return self.GEMM + "\n" + \
            self.GEMA + "\n"  + \
            self.GEMM_x + "\n"  + \
            self.GEMA_x + "\n" + \
            self.MALLOC + "\n" + \
            self.FREE + "\n"
    
    def Code(self, S : str, L ) -> str :
        L1 = list(L); L1.pop(0)
        return S % tuple(L1)
        



    def compile_and_import(self,S : str, TYP : str, d : str = "JITGpu/",  filename : str = "codex.h"):

        if S is None or filename is None:
            return None
        
        import os
        import subprocess
        
        
        F = open(d+"codextype.h", "w")
        F.write("#define T 1 \n typedef %s TT; \n" %TYP)
        F.close()
        F = open(d+filename, "w")
        F.write(self.MAGE)
        F.write(S)
        F.close()
        subprocess.Popen("touch %s/*.cpp" % d,shell = True)
        cmd = 'cd %s; python3 setup.py build' % d
        subprocess.Popen(cmd,shell = True)





class GPU(CPU):
    def __init__(self):
        self.MAGE = ""

        self.INTRO =    """
        // standard column layout 
        rocblas_operation transa = rocblas_operation_none, transb = rocblas_operation_none;
        rocblas_int device_id;  // here is the gpu_ parameter
        rocblas_handle gpu  = nullptr;
        

        if (DEBUG) std::cout << "\t ROCBLAS GEMM  "  <<  gpu << std::endl;
        hipDeviceProp_t devProp;
        //HIP_CHECK(hipSetDevice(device_id));
        HIP_CHECK(hipSetDevice(gpu_));
        HIP_CHECK(hipGetDeviceProperties(&devProp, gpu_));
        ROCBLAS_CHECK(rocblas_create_handle(&gpu));
        if (DEBUG) std::cout << gpu_ <<" Device: " << devProp.name << std::endl;
        int M = %d, N = %d, K = %d;
        int size_a = M*K, size_b = K*N, size_c = M*N;

        // allocate memory on device
        TT *A, *B, *C;
        HIP_CHECK(hipMalloc(&A, M*K* sizeof(TT)));
        HIP_CHECK(hipMalloc(&B, K*N * sizeof(TT)));
        HIP_CHECK(hipMalloc(&C, M*N * sizeof(TT)));

        // copy to device
        HIP_CHECK(hipMemcpy(A, hA.data(), sizeof(TT) * size_a, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(B, hB.data(), sizeof(TT) * size_b, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(C, hC.data(), sizeof(TT) * size_c, hipMemcpyHostToDevice));
        auto start = std::chrono::high_resolution_clock::now();
        """


        
        self.GEMM = """
rocblas_status rocblas_dgemm(
        rocblas_handle handle, 
        rocblas_operation transA, 
        rocblas_operation transB, 
        rocblas_int m, 
        rocblas_int n, 
        rocblas_int k, 
        const double *alpha, 
        const double *A, 
        rocblas_int lda, 
        const double *B, 
        rocblas_int ldb, 
        const double *beta, 
        double *C, 
        rocblas_int ldc) """

        ## handle, M, N, K, alpha, A,  LDA=K, B, LDB=N, beta,C, LDC=N 
        self.GEMM_i = "## handle, M, N, K, alpha, A,  LDA=K, B, LDB=N, beta,C, LDC=N "
        self.GEMM_x = "" + \
            "rocblas_dgemm( %s, transa, transb, %d, " + "%d, %d,  %s, %s, %d, %s, %d,  %s, %s, %d); "


        self.TAIL = """
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        if (DEBUG) std::cout << "\t Time Kernel "  << duration.count()/1000000.0 << std::endl;
        HIP_CHECK(hipMemcpy(hC.data(), C, sizeof(double) * size_c, hipMemcpyDeviceToHost));
        """


        self.GEMA = """
    rocblas_status rocblas_dgeam(
        rocblas_handle handle, 
        rocblas_operation transA, 
        rocblas_operation transB, 
        rocblas_int m, 
        rocblas_int n, 
        const double *alpha, 
        const double *A, 
        rocblas_int lda, 
        const double *beta, 
        const double *B, 
        rocblas_int ldb, 
        double *C, 
        rocblas_int ldc) """

        ## handle, M, N, K, alpha, A, beta, LDA=K, B, LDB=N, beta,C, LDC=N 
        self.GEMA_i = "## handle, M, N, alpha, A, lda, B, LDB=N, beta,C, LDC=N "
        
        self.GEMA_x = "" + \
            "rocblas_dgeam( %s, transa, transb, %d, %d,  " +\
            "%s, %s, %d,   %s, %s, %d,  %s, %d); "


        self.MALLOC = "HIP_CHECK(hipMalloc(&%s, %d*%d* sizeof(%s)));"
        self.FREE   = "HIP_CHECK(hipFree(%s));" 
        
    def __str__(self):
        return self.GEMM + "\n" + \
            self.GEMA + "\n"  + \
            self.GEMM_x + "\n"  + \
            self.GEMA_x + "\n" + \
            self.MALLOC + "\n" + \
            self.FREE + "\n"


    def Code(self, S : str, L : list) -> str :
        compute =  S % tuple(L)
        sync = "HIP_CHECK(hipDeviceSynchronize()); \n" 
        
        return compute + sync
    
        


BLAS    = CPU()        
ROCBLAS = GPU()
print(ROCBLAS)

class GPUI:
    def __init__(self):


        self.GEMM = """
        std::vector<double> gemm(int device_id,
			 std::vector<double> ha, int lda, 
			 std::vector<double> hb, int ldb
			 )"""

        ## handle, M, N, K, alpha, A,  LDA=K, B, LDB=N, beta,C, LDC=N 
        self.GEMM_i = "## handle,  A,  lda, B , ldb "
        self.GEMM_x = "" + \
            "gemm( %s, %s %d %s %d) "

        self.GEMA = """
        std::vector<double> gema(int device_id,
			 std::vector<double> ha, int lda, 
			 std::vector<double> hb, int ldb
			 )
        """

        ## handle, M, N, K, alpha, A, beta, LDA=K, B, LDB=N, beta,C, LDC=N 
        self.GEMA_i = "## handle,  A,  lda, B , ldb "
        
        self.GEMA_x = "" + \
            "gema( %s, %s %d %s %d) "


    def __str__(self):
        return self.GEMM + "\n" + \
            self.GEMA + "\n"  + \
            self.GEMM_x + "\n"  + \
            self.GEMA_x + "\n"  
