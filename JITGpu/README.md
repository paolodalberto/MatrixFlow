
## Code Generation: C++ and for GPUs ?

Did you ever wanted to use GPUs. I have been trying to have a verified
computations since 10 years ago and I have always wanted to run the
code on GPUs.

Now you can use any of the example and "turn on" GPU by means of the 

```sh
export GPU=0 ## you have multiple GPU and Choose GPU=0
```

If you like a verbose output just build the Gpuinterface with DEBUG=1
(see the code).  Every GEMM will be executed on GPU=0. Ok course, this
is not efficient. 


Now there is an interesting case for play_3: we will generate C++ code
for GPU, that is just basic kernels. 


```py
    code = G3.pretty__C(python_compiler = True) 
    print(code)
    ROCBLAS.compile_and_import(
        code,
        TYP = str(Graph.numpytoC(G3.declarations[0][0].type_matrix())))

    import pdb; pdb.set_trace()
    import one
    H1 = Scalar(0)*C
    H = one.fastgemm(0,A.value().A.flatten(), B.value().A.flatten(), H1.value().A.flatten())
    R = numpy.matrix(
        H
    )
    B1 = R.reshape(C.value().shape)
```

At this time, we create a look like C++ using a interface that
resemble rocmBLAS but it is a CPU code.

```c++
std::vector<int>fastgemm(int gpu,std::vector<int> hA,  std::vector<int> hB,  std::vector<int> hC  ){ 
int *A = hA.data();int *B = hB.data();int *C = hC.data();// Variables declaration 
int* ADP[4]; ADP[0]= A+12*0 +0;ADP[1]= A+12*0 +6;ADP[2]= A+12*6 +0;ADP[3]= A+12*6 +6;
int* BDP[4]; BDP[0]= B+12*0 +0;BDP[1]= B+12*0 +6;BDP[2]= B+12*6 +0;BDP[3]= B+12*6 +6;
int* CDP[4]; CDP[0]= C+12*0 +0;CDP[1]= C+12*6 +0;CDP[2]= C+12*0 +6;CDP[3]= C+12*6 +6;
int* Pss[1]; Pss[0]= (int*)malloc (6*6*sizeof(int));
int* Ts[2]; Ts[0]= (int*)malloc (6*6*sizeof(int));Ts[1]= (int*)malloc (6*6*sizeof(int));
// Constant declaration 
int z_0=0,z_1=1,z_2=-1;
// code 
// Pss[0] << (ADP[2] - ADP[3]) * (BDP[1])
rocblas_dgema( gpu, 'n', 'n', 6, 6, 6,  &z_1, ADP[2], 12,  ADP[3], 12, &z_2,  Ts[0], 6); 
rocblas_dgemm( gpu, 'n', 'n', 6, 6, 6,  &z_1, Ts[0], 6, BDP[1], 12,  &z_0, Pss[0], 6); 
// CDP[3] << CDP[3] + Pss[0]
rocblas_dgema( gpu, 'n', 'n', 6, 6, 6,  &z_1, CDP[3], 12,  Pss[0], 6, &z_1,  CDP[3], 12); 
// CDP[2] << CDP[2] - Pss[0]
rocblas_dgema( gpu, 'n', 'n', 6, 6, 6,  &z_1, CDP[2], 12,  Pss[0], 6, &z_2,  CDP[2], 12); 
// Pss[0] << (ADP[0] + ADP[2] - ADP[3]) * (BDP[1] + BDP[2] + BDP[3])
rocblas_dgema( gpu, 'n', 'n', 6, 6, 6,  &z_1, ADP[0], 12,  ADP[2], 12, &z_1,  Ts[0], 6); 
rocblas_dgema( gpu, 'n', 'n', 6, 6, 6,  &z_1, Ts[0], 6,  ADP[3], 12, &z_2,  Ts[0], 6); 
rocblas_dgema( gpu, 'n', 'n', 6, 6, 6,  &z_1, BDP[1], 12,  BDP[2], 12, &z_1,  Ts[1], 6); 
rocblas_dgema( gpu, 'n', 'n', 6, 6, 6,  &z_1, Ts[1], 6,  BDP[3], 12, &z_1,  Ts[1], 6); 
rocblas_dgemm( gpu, 'n', 'n', 6, 6, 6,  &z_1, Ts[0], 6, Ts[1], 6,  &z_0, Pss[0], 6); 
// CDP[2] << CDP[2] + Pss[0]
rocblas_dgema( gpu, 'n', 'n', 6, 6, 6,  &z_1, CDP[2], 12,  Pss[0], 6, &z_1,  CDP[2], 12); 
// CDP[1] << CDP[1] - Pss[0]
rocblas_dgema( gpu, 'n', 'n', 6, 6, 6,  &z_1, CDP[1], 12,  Pss[0], 6, &z_2,  CDP[1], 12); 
// Pss[0] << (ADP[0] + ADP[2] - ADP[1] - ADP[3]) * (BDP[2] + BDP[3])
rocblas_dgema( gpu, 'n', 'n', 6, 6, 6,  &z_1, ADP[0], 12,  ADP[2], 12, &z_1,  Ts[0], 6); 
rocblas_dgema( gpu, 'n', 'n', 6, 6, 6,  &z_1, Ts[0], 6,  ADP[1], 12, &z_2,  Ts[0], 6); 
rocblas_dgema( gpu, 'n', 'n', 6, 6, 6,  &z_1, Ts[0], 6,  ADP[3], 12, &z_2,  Ts[0], 6); 
rocblas_dgema( gpu, 'n', 'n', 6, 6, 6,  &z_1, BDP[2], 12,  BDP[3], 12, &z_1,  Ts[1], 6); 
rocblas_dgemm( gpu, 'n', 'n', 6, 6, 6,  &z_1, Ts[0], 6, Ts[1], 6,  &z_0, Pss[0], 6); 
// CDP[2] << CDP[2] - Pss[0]
rocblas_dgema( gpu, 'n', 'n', 6, 6, 6,  &z_1, CDP[2], 12,  Pss[0], 6, &z_2,  CDP[2], 12); 
// Pss[0] << (ADP[1]) * (BDP[2])
rocblas_dgemm( gpu, 'n', 'n', 6, 6, 6,  &z_1, ADP[1], 12, BDP[2], 12,  &z_0, Pss[0], 6); 
// CDP[0] << CDP[0] + Pss[0]
rocblas_dgema( gpu, 'n', 'n', 6, 6, 6,  &z_1, CDP[0], 12,  Pss[0], 6, &z_1,  CDP[0], 12); 
// CDP[2] << CDP[2] - Pss[0]
rocblas_dgema( gpu, 'n', 'n', 6, 6, 6,  &z_1, CDP[2], 12,  Pss[0], 6, &z_2,  CDP[2], 12); 
// Pss[0] << (ADP[0] + ADP[2]) * (BDP[0] + BDP[1] + BDP[2] + BDP[3])
rocblas_dgema( gpu, 'n', 'n', 6, 6, 6,  &z_1, ADP[0], 12,  ADP[2], 12, &z_1,  Ts[0], 6); 
rocblas_dgema( gpu, 'n', 'n', 6, 6, 6,  &z_1, BDP[0], 12,  BDP[1], 12, &z_1,  Ts[1], 6); 
rocblas_dgema( gpu, 'n', 'n', 6, 6, 6,  &z_1, Ts[1], 6,  BDP[2], 12, &z_1,  Ts[1], 6); 
rocblas_dgema( gpu, 'n', 'n', 6, 6, 6,  &z_1, Ts[1], 6,  BDP[3], 12, &z_1,  Ts[1], 6); 
rocblas_dgemm( gpu, 'n', 'n', 6, 6, 6,  &z_1, Ts[0], 6, Ts[1], 6,  &z_0, Pss[0], 6); 
// CDP[1] << CDP[1] + Pss[0]
rocblas_dgema( gpu, 'n', 'n', 6, 6, 6,  &z_1, CDP[1], 12,  Pss[0], 6, &z_1,  CDP[1], 12); 
// Pss[0] << (ADP[0]) * (BDP[0])
rocblas_dgemm( gpu, 'n', 'n', 6, 6, 6,  &z_1, ADP[0], 12, BDP[0], 12,  &z_0, Pss[0], 6); 
// CDP[0] << CDP[0] + Pss[0]
rocblas_dgema( gpu, 'n', 'n', 6, 6, 6,  &z_1, CDP[0], 12,  Pss[0], 6, &z_1,  CDP[0], 12); 
// CDP[1] << CDP[1] - Pss[0]
rocblas_dgema( gpu, 'n', 'n', 6, 6, 6,  &z_1, CDP[1], 12,  Pss[0], 6, &z_2,  CDP[1], 12); 
// Pss[0] << (ADP[3]) * (BDP[1] + BDP[3])
rocblas_dgema( gpu, 'n', 'n', 6, 6, 6,  &z_1, BDP[1], 12,  BDP[3], 12, &z_1,  Ts[1], 6); 
rocblas_dgemm( gpu, 'n', 'n', 6, 6, 6,  &z_1, ADP[3], 12, Ts[1], 6,  &z_0, Pss[0], 6); 
// CDP[3] << CDP[3] + Pss[0]
rocblas_dgema( gpu, 'n', 'n', 6, 6, 6,  &z_1, CDP[3], 12,  Pss[0], 6, &z_1,  CDP[3], 12); 
// CDP[1] << CDP[1] - Pss[0]
rocblas_dgema( gpu, 'n', 'n', 6, 6, 6,  &z_1, CDP[1], 12,  Pss[0], 6, &z_2,  CDP[1], 12); 
// free 
free(Pss[0]);
free(Ts[0]);
free(Ts[1]);
return hC;}

```
