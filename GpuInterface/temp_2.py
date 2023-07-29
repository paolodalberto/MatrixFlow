import numpy
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
a = time.time()
for i in range(R):
     Z2 = example.gemm(1,C.flatten(),C.shape[1],A.flatten(),A.shape[1], B.flatten(),B.shape[1], 1.0,0.0)
b = time.time()
print("NOINFO time", (b-a)/R)
a = time.time()
for i in range(R):
     Z3 = example.gemm(2,C.flatten(),C.shape[1],A.flatten(),A.shape[1], B.flatten(),B.shape[1], 1.0,0.0)
b = time.time()
print("NOINFO time", (b-a)/R)

example.endit()
