




class GPU:
    def __init__(self):


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
            "rocblas_dgemm( %s, 'n', 'n', %d, " + \
            "%d, %d,  %s, %s, %d, %s, %d,  %s, %s, %d); "

        self.GEMA = """
    rocblas_status rocblas_dgema(
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

        ## handle, M, N, K, alpha, A, beta, LDA=K, B, LDB=N, beta,C, LDC=N 
        self.GEMA_i = "## handle, M, N, K, alpha, A, beta, LDA=K, B, LDB=N, beta,C, LDC=N "
        
        self.GEMA_x = "" + \
            "rocblas_dgema( %s, 'n', 'n', %d, %d, %d,  " +\
            "%s, %s, %d,  %s, %s, %s,  %s, %d); "


        self.MALLOC = "HIP_CHECK(hipMalloc(&%s, %d*%d* sizeof(%s)));"
        self.FREE   = "HIP_CHECK(hipFree(%s));" 
        
    def __str__(self):
        return self.GEMM + "\n" + \
            self.GEMA + "\n"  + \
            self.GEMM_x + "\n"  + \
            self.GEMA_x + "\n" + \
            self.MALLOC + "\n" + \
            self.FREE + "\n"
    
    def compile_and_import(self,S : str, TYP : str, d : str = "JITGpu/",  filename : str = "codex.h"):

        if S is None or filename is None:
            return None
        
        import os
        import subprocess
        

        F = open(d+"codextype.h", "w")
        F.write("#define T 1 \n typedef %s TT; \n" %TYP)
        F.close()
        F = open(d+filename, "w")
        F.write(S)
        F.close()
        cmd = 'cd %s; python3 setup.py build' % d
        subprocess.Popen(cmd,shell = True)
        
        


        
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
