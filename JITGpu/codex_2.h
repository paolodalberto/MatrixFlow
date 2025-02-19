
std::vector<double> fastgemm(int  gpu_,std::vector<double> hA, int lda, std::vector<double> hB, int ldb, std::vector<double> hC, int ldc  ){ 

        // standard column layout 
        rocblas_operation transa = rocblas_operation_none, transb = rocblas_operation_none;
        rocblas_int device_id;  // here is the gpu_ parameter
        rocblas_handle gpu  = nullptr;
        

        if (DEBUG) std::cout << "	 ROCBLAS GEMM  "  <<  gpu << std::endl;
        hipDeviceProp_t devProp;
        //HIP_CHECK(hipSetDevice(device_id));
        HIP_CHECK(hipSetDevice(gpu_));
        HIP_CHECK(hipGetDeviceProperties(&devProp, gpu_));
        ROCBLAS_CHECK(rocblas_create_handle(&gpu));
        if (DEBUG) std::cout << gpu_ <<" Device: " << devProp.name << std::endl;
        int size_a = hA.size(), size_b = hB.size(), size_c = hC.size();

        // allocate memory on device
        TT *A, *B, *C;
        HIP_CHECK(hipMalloc(&A, size_a* sizeof(TT)));
        HIP_CHECK(hipMalloc(&B, size_b * sizeof(TT)));
        HIP_CHECK(hipMalloc(&C, size_c * sizeof(TT)));

        // copy to device
        HIP_CHECK(hipMemcpy(A, hA.data(), sizeof(TT) * size_a, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(B, hB.data(), sizeof(TT) * size_b, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(C, hC.data(), sizeof(TT) * size_c, hipMemcpyHostToDevice));
        auto start = std::chrono::high_resolution_clock::now();
        // Variables declaration 
double* ADP[4]; ADP[0]= A+lda*0 +0;ADP[1]= A+lda*6 +0;ADP[2]= A+lda*0 +6;ADP[3]= A+lda*6 +6;
double* BDP[4]; BDP[0]= B+ldb*0 +0;BDP[1]= B+ldb*6 +0;BDP[2]= B+ldb*0 +6;BDP[3]= B+ldb*6 +6;
double* CDP[4]; CDP[0]= C+ldc*0 +0;CDP[1]= C+ldc*0 +6;CDP[2]= C+ldc*6 +0;CDP[3]= C+ldc*6 +6;
double* Pss[1]; HIP_CHECK(hipMalloc(&Pss[0], 6*6* sizeof(double)));
double* Ts[2]; HIP_CHECK(hipMalloc(&Ts[0], 6*6* sizeof(double)));HIP_CHECK(hipMalloc(&Ts[1], 6*6* sizeof(double)));
// Constant declaration 
double z_0=0,z_1=1,z_2=-1;
// code 
// Pss[0] << (ADP[2] - ADP[3]) * (BDP[1])
rocblas_dgeam( gpu, transa, transb, 6, 6,  &z_1, ADP[2], lda,   &z_2, ADP[3], lda,  Ts[0], lda/2); HIP_CHECK(hipDeviceSynchronize()); 
 std::cout << lda << " "  << ldb << " " << ldc << "\n";
 Print(ADP[2],lda,6,6);
 Print(ADP[3],lda,6,6);
 Print(Ts[0],lda/2,6,6);
rocblas_dgemm( gpu, transa, transb, 6, 6, 6,  &z_1, Ts[0], lda/2, BDP[1], ldb,  &z_0, Pss[0], ldc/2); HIP_CHECK(hipDeviceSynchronize()); 
 Print(BDP[1],ldb,6,6);
 Print(Pss[0],ldc/2,6,6);

 //#if (1) return hC; 

 
// CDP[3] << CDP[3] + Pss[0]
Print(CDP[3],ldc,6,6);
 rocblas_dgeam( gpu, transa, transb, 6, 6,  &z_1, CDP[3], ldc,   &z_1, Pss[0], ldc/2,  CDP[3], ldc); HIP_CHECK(hipDeviceSynchronize());
Print(CDP[3],ldc,6,6);
 Print(C,ldc,12,12);
// CDP[2] << CDP[2] - Pss[0]
rocblas_dgeam( gpu, transa, transb, 6, 6,  &z_1, CDP[2], ldc/2,   &z_2, Pss[0], ldc/2,  CDP[2], ldc); HIP_CHECK(hipDeviceSynchronize()); 

// Pss[0] << (ADP[0] + ADP[2] - ADP[3]) * (BDP[1] + BDP[2] + BDP[3])
rocblas_dgeam( gpu, transa, transb, 6, 6,  &z_1, ADP[0], lda,   &z_1, ADP[2], lda,  Ts[0], lda/2); HIP_CHECK(hipDeviceSynchronize()); 

rocblas_dgeam( gpu, transa, transb, 6, 6,  &z_1, Ts[0], lda/2,   &z_2, ADP[3], lda,  Ts[0], lda/2); HIP_CHECK(hipDeviceSynchronize()); 

rocblas_dgeam( gpu, transa, transb, 6, 6,  &z_1, BDP[1], ldb,   &z_1, BDP[2], ldb,  Ts[1], ldb/2); HIP_CHECK(hipDeviceSynchronize()); 

rocblas_dgeam( gpu, transa, transb, 6, 6,  &z_1, Ts[1], ldb/2,   &z_1, BDP[3], ldb,  Ts[1], ldb/2); HIP_CHECK(hipDeviceSynchronize()); 

rocblas_dgemm( gpu, transa, transb, 6, 6, 6,  &z_1, Ts[0], lda/2, Ts[1], ldb/2,  &z_0, Pss[0], ldc/2); HIP_CHECK(hipDeviceSynchronize()); 


// CDP[2] << CDP[2] + Pss[0]
rocblas_dgeam( gpu, transa, transb, 6, 6,  &z_1, CDP[2], ldc/2,   &z_1, Pss[0], ldc/2,  CDP[2], ldc); HIP_CHECK(hipDeviceSynchronize()); 

// CDP[1] << CDP[1] - Pss[0]
rocblas_dgeam( gpu, transa, transb, 6, 6,  &z_1, CDP[1], ldc/2,   &z_2, Pss[0], ldc/2,  CDP[1], ldc); HIP_CHECK(hipDeviceSynchronize()); 

// Pss[0] << (ADP[0] + ADP[2] - ADP[1] - ADP[3]) * (BDP[2] + BDP[3])
rocblas_dgeam( gpu, transa, transb, 6, 6,  &z_1, ADP[0], lda,   &z_1, ADP[2], lda,  Ts[0], lda/2); HIP_CHECK(hipDeviceSynchronize()); 

rocblas_dgeam( gpu, transa, transb, 6, 6,  &z_1, Ts[0], lda/2,   &z_2, ADP[1], lda,  Ts[0], lda/2); HIP_CHECK(hipDeviceSynchronize()); 

rocblas_dgeam( gpu, transa, transb, 6, 6,  &z_1, Ts[0], lda/2,   &z_2, ADP[3], lda,  Ts[0], lda/2); HIP_CHECK(hipDeviceSynchronize()); 

rocblas_dgeam( gpu, transa, transb, 6, 6,  &z_1, BDP[2], ldb,   &z_1, BDP[3], ldb,  Ts[1], ldb/2); HIP_CHECK(hipDeviceSynchronize()); 

rocblas_dgemm( gpu, transa, transb, 6, 6, 6,  &z_1, Ts[0], lda/2, Ts[1], ldb/2,  &z_0, Pss[0], ldc/2); HIP_CHECK(hipDeviceSynchronize()); 

// CDP[2] << CDP[2] - Pss[0]
rocblas_dgeam( gpu, transa, transb, 6, 6,  &z_1, CDP[2], ldc/2,   &z_2, Pss[0], ldc/2,  CDP[2], ldc); HIP_CHECK(hipDeviceSynchronize()); 

// Pss[0] << (ADP[1]) * (BDP[2])
rocblas_dgemm( gpu, transa, transb, 6, 6, 6,  &z_1, ADP[1], lda, BDP[2], ldb,  &z_0, Pss[0], ldc/2); HIP_CHECK(hipDeviceSynchronize()); 

// CDP[0] << CDP[0] + Pss[0]
rocblas_dgeam( gpu, transa, transb, 6, 6,  &z_1, CDP[0], ldc/2,   &z_1, Pss[0], ldc/2,  CDP[0], ldc); HIP_CHECK(hipDeviceSynchronize()); 

// CDP[2] << CDP[2] - Pss[0]
rocblas_dgeam( gpu, transa, transb, 6, 6,  &z_1, CDP[2], ldc/2,   &z_2, Pss[0], ldc/2,  CDP[2], ldc); HIP_CHECK(hipDeviceSynchronize()); 

// Pss[0] << (ADP[0] + ADP[2]) * (BDP[0] + BDP[1] + BDP[2] + BDP[3])
rocblas_dgeam( gpu, transa, transb, 6, 6,  &z_1, ADP[0], lda,   &z_1, ADP[2], lda,  Ts[0], lda/2); HIP_CHECK(hipDeviceSynchronize()); 

rocblas_dgeam( gpu, transa, transb, 6, 6,  &z_1, BDP[0], ldb,   &z_1, BDP[1], ldb,  Ts[1], ldb/2); HIP_CHECK(hipDeviceSynchronize()); 

rocblas_dgeam( gpu, transa, transb, 6, 6,  &z_1, Ts[1], ldb/2,   &z_1, BDP[2], ldb,  Ts[1], ldb/2); HIP_CHECK(hipDeviceSynchronize()); 

rocblas_dgeam( gpu, transa, transb, 6, 6,  &z_1, Ts[1], ldb/2,   &z_1, BDP[3], ldb,  Ts[1], ldb/2); HIP_CHECK(hipDeviceSynchronize()); 

rocblas_dgemm( gpu, transa, transb, 6, 6, 6,  &z_1, Ts[0], lda/2, Ts[1], ldb/2,  &z_0, Pss[0], ldc/2); HIP_CHECK(hipDeviceSynchronize()); 

// CDP[1] << CDP[1] + Pss[0]
rocblas_dgeam( gpu, transa, transb, 6, 6,  &z_1, CDP[1], ldc/2,   &z_1, Pss[0], ldc/2,  CDP[1], ldc); HIP_CHECK(hipDeviceSynchronize()); 

// Pss[0] << (ADP[0]) * (BDP[0])
rocblas_dgemm( gpu, transa, transb, 6, 6, 6,  &z_1, ADP[0], lda, BDP[0], ldb,  &z_0, Pss[0], ldc/2); HIP_CHECK(hipDeviceSynchronize()); 

// CDP[0] << CDP[0] + Pss[0]
rocblas_dgeam( gpu, transa, transb, 6, 6,  &z_1, CDP[0], ldc/2,   &z_1, Pss[0], ldc/2,  CDP[0], ldc); HIP_CHECK(hipDeviceSynchronize()); 

// CDP[1] << CDP[1] - Pss[0]
rocblas_dgeam( gpu, transa, transb, 6, 6,  &z_1, CDP[1], ldc/2,   &z_2, Pss[0], ldc/2,  CDP[1], ldc); HIP_CHECK(hipDeviceSynchronize()); 

// Pss[0] << (ADP[3]) * (BDP[1] + BDP[3])
rocblas_dgeam( gpu, transa, transb, 6, 6,  &z_1, BDP[1], ldb,   &z_1, BDP[3], ldb,  Ts[1], ldb/2); HIP_CHECK(hipDeviceSynchronize()); 

rocblas_dgemm( gpu, transa, transb, 6, 6, 6,  &z_1, ADP[3], lda, Ts[1], ldb/2,  &z_0, Pss[0], ldc/2); HIP_CHECK(hipDeviceSynchronize()); 

// CDP[3] << CDP[3] + Pss[0]
rocblas_dgeam( gpu, transa, transb, 6, 6,  &z_1, CDP[3], ldc/2,   &z_1, Pss[0], ldc/2,  CDP[3], ldc); HIP_CHECK(hipDeviceSynchronize()); 

// CDP[1] << CDP[1] - Pss[0]
rocblas_dgeam( gpu, transa, transb, 6, 6,  &z_1, CDP[1], ldc/2,   &z_2, Pss[0], ldc/2,  CDP[1], ldc); HIP_CHECK(hipDeviceSynchronize()); 

// free 
HIP_CHECK(hipFree(Pss[0]));
HIP_CHECK(hipFree(Ts[0]));
HIP_CHECK(hipFree(Ts[1]));

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        if (DEBUG==1) std::cout << "	 Time Kernel "  << duration.count()/1000000.0 << std::endl;
        HIP_CHECK(hipMemcpy(hC.data(), C, sizeof(double) * size_c, hipMemcpyDeviceToHost));
        HIP_CHECK(hipFree(A));
HIP_CHECK(hipFree(B));
HIP_CHECK(hipFree(C));

//Print(C, ldc, 12,12);
// Print(hC.data(), ldc, 12,12);
   
return hC;}
