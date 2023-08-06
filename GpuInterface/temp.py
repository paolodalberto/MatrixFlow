from  scipy.io import mmread
import scipy 
import time 
import rocmgpu as example
import numpy


#example.init()
R = 2


A = mmread("/sparse/clients/samples/MTX/Groebner_id2003_aug.mtx")

A = A.astype('float64')
x =  numpy.zeros(A.shape[1]) + 0.5
y =  numpy.zeros(A.shape[0]) 


W = A.todense()
x = numpy.array(x)
y = numpy.array(y)
RT = numpy.dot(W,x)

     
if False:
     #
     Av = A.data +0.1
     IA = A.row
     JA = A.col
     print(Av[0], IA[0], JA[0])
     #pdb.set_trace()

     a = time.time()
     for i in range(R):
          z = example.coo_mv(0, IA,JA, Av,x,y,1.0,0.0); print(z[0])
          #z1= example.coo_mv(1, IA,JA, Av,x,y,1.0,0.0); print(z1[0])
     b = time.time()
     print("NOINFO time", (b-a)/R)
     #print(z[0], z1[0])
     import pdb; pdb.set_trace()
     z = example.coo_mv(1, IA,JA, Av,x,y,1.0,0.0); print(z[0])
     t = A@x
     print(numpy.sum(z-t))
     



import pdb; pdb.set_trace()#

AA = scipy.sparse.csr_matrix(A)

A = AA.data
A1 = A
IA = AA.indptr
JA = AA.indices

z = example.csr_mv(0, IA,JA, A,x,y,1.0,0.0);  print(z[0])
t = AA@x
print(numpy.sum(numpy.array(z)-t))
     
pdb.set_trace()

print(R)
a = time.time()
for i in range(R):
     
     z1= example.csr_mv(1, IA,JA, A1,x,y,1,0); #print(z1[0])
    #z2= example.csr_mv(2, IA,JA, A1,x,y,1,0); #print(z2[0])
b = time.time()
print("NOINFO time", (b-a)/R)
#print(z[0], z1[0])


#import pdb; 

#pdb.set_trace()
print(numpy.sum(RT-z))
K = 100;
A = numpy.ones((K,K), dtype = numpy.float64)
B = A*1.0

C = numpy.matmul(A,B)
CT = C *0; 
for i in range(R):
     RG =  example.gemm(0,
                        C.flatten(), C.shape[1],
                        A.flatten(), A.shape[1],
                        B.flatten(), B.shape[1],
                        1.0, 0.0)
     A = A+0.0001;

import pdb;
pdb.set_trace()


#example.endit()
