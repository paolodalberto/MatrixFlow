from  scipy.io import mmread
import scipy 
import time 
import rocmgpu as example



example.init()
R = 1
A = mmread("./MTX/arrow.mtx")
x =  []
for i in range(A.shape[1]):
     x.append(1)
y =  []
for i in range(A.shape[0]):
     y.append(0)

if False :
     import pdb; pdb.set_trace()
     Av = A.data
     A1 = A.data 
     IA = A.row
     JA = A.col

     pdb.set_trace()

     a = time.time()
     for i in range(R):
          z = example.coo_mv(0, IA,JA, Av,x,y,1,0); print(z[0])
          z1= example.coo_mv(1, IA,JA, A1,x,y,1,0); print(z1[0])
     b = time.time()
     print("NOINFO time", (b-a)/R)
     print(z[0], z1[0])




import pdb; #
AA = scipy.sparse.csr_matrix(A.todense())

A = AA.data
A1 = A+1
IA = AA.indptr
JA = AA.indices



print(R)
z= example.csr_mv(0, IA,JA, A,x,y,1,0)
a = time.time()
for i in range(R):
    z = example.csr_mv(0, IA,JA, A,x,y,1,0); print(z[0])
    z1= example.csr_mv(1, IA,JA, A1,x,y,1,0); print(z1[0])
    z2= example.csr_mv(2, IA,JA, A1,x,y,1,0); print(z2[0])
b = time.time()
print("NOINFO time", (b-a)/R)
print(z[0], z1[0])


import pdb; #





example.endit()
