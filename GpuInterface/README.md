# Python, ROCM, and GPUs (with a s)

This is a place where I want and I can play with GPUs of the ROCM 5.6
family. I have VIIs (vega 20) and a docker tensorflow:latest. I have
installed the source for rocBLAS and rocSPARSE but I think you just
need the libraries. This is because the interface is actually a thin
C++ implementation of Sparse Mv (CSR and COO) and GEMM. The testing is
limited and the performances are just bound to the interface and the
PCI communications. 


If you have you ROCM 5.6 tensorflow, this will come with rocsparse and
rocblas and I believe is the only thing you need.
```sh
root@fastmmw:/matrixflow/GpuInterface# ls /opt/rocm   
amdgcn   hipblaslt  hipsolver  lib      oam         rocblas   rocprofiler  rocthrust
bin      hipcub     hipsparse  libexec  opencl      rocfft    rocrand      roctracer
hip      hipfft     hsa        llvm     rccl        rocm_smi  rocsolver    share
hipblas  hiprand    include    miopen   rocalution  rocprim   rocsparse
```

The building of this module is simply 
```sh
   python setup.py build
   export PYTHONPATH=$PWD/build/lib.linux-x86_64-cpython-39/
```   

I provide two python examples: temp.py and temp_2.py. The first is
simply an example of sparse computation

```py
from  scipy.io import mmread
import scipy 
import time 
import rocmgpu as example
example.init()   ## ROCM HANDLE
R = 1
A = mmread("./MTX/arrow.mtx")
A = scipy.sparse.csr_matrix(A.todense())

x =  [];
for i in range(A.shape[1]):     x.append(1)
y =  [];
for i in range(A.shape[0]):     y.append(0)
A = AA.data
A1 = A+1
IA = AA.indptr
JA = AA.indices

z = example.csr_mv(0, IA,JA, A,x,y,1,0); print(z[0])   ## GPU0
z1= example.csr_mv(1, IA,JA, A1,x,y,1,0); print(z1[0]) ## GPU1

example.endit() ## ROCMHANDLE
```

and the second is example of gemm
```py

mport numpy
import rocmgpu as example
example.init()
import time 

K = 8
A = numpy.ones((K*1024, K*1024), dtype=numpy.float64)
B = numpy.ones((K*1024, K*1024), dtype=numpy.float64)
C = numpy.ones((K*1024, K*1024), dtype=numpy.float64)

R = 1
a = time.time()
Z = numpy.matmul(A,B)
b = time.time()
print("NOINFO time", (b-a)/R)



a = time.time()
for i in range(R):
     Z1 = example.gemm(0,C.flatten(),C.shape[1],A.flatten(),A.shape[1], B.flatten(),B.shape[1], 1.0,0.0)
b = time.time()
print("NOINFO time", (b-a)/R)
example.endit()
```

Matrix Flow is a Python project and I wanted to have the opportunity
to play with accelerators. The interface is not really for performance
purpose, if you measure all the time spent in the kernel versus the
communications I think Scipy and Numpy (CPU) are by far more appealing
at least with this type of interface. You can see this as a first step
in the implementation of code for GPUs: Matrix Flow should be able
eventually to write code for accelerators and thus this a practical
first step

