from  scipy.io import mmread
import scipy 
import time 
import rocmgpu as example
import numpy


#example.init()
R = 2


A = mmread("/sparse/clients/samples/MTX/Groebner_id2003_aug.mtx")
A = mmread('/sparse/A/sparse-computations/AD08/training/model/ad08_0.9finSpar/q_dense_batchnorm_weights.mtx').transpose()

#A = A.astype('float64')
x =  numpy.zeros(A.shape[1]) + 0.5
y =  numpy.zeros(A.shape[0]) 


W = A.todense()
x = numpy.array(x)
y = numpy.array(y)
RT = numpy.dot(W,x)

     
if True:

     a = time.time()
     for i in range(R):
          z = example.coo_mv(0, A.row,A.col, A.data,x,y,1.0,0.0); print(z[0])
          #z1= example.coo_mv(1, IA,JA, Av,x,y,1.0,0.0); print(z1[0])
     b = time.time()
     print("NOINFO time", (b-a)/R)
     #print(z[0], z1[0])
     import pdb; pdb.set_trace()
     z1 = example.coo_mv(0, A.row,A.col, A.data,x,y,1.0,0.0); print(z[0])
     t = A@x
     t = A.todense()@x
     print(numpy.sum(z1-t))
     



import pdb; pdb.set_trace()#

AA = scipy.sparse.csr_matrix(A)



z = example.csr_mv(0, AA.indptr,AA.indices, AA.data,x,y,1.0,0.0);  print(z[0])
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
