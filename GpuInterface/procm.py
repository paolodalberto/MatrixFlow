import rocmgpu
import numpy
import os
from scipy.sparse import csr_matrix

VERIFY = True if "VERIFY" in os.environ else False

###
## rocBLAS style:
##    C has to be in column major layout
##    x is a dense vector
##    b may be not dense 
## z = 1*Cx + 1*y  
###
def fromsparse_dgemv(device, C, x, y) :
    
     W = rocmgpu.gemv(device,
                   C.toarray().flatten(order='F'),C.shape[0],
                   x,
                   y.toarray().flatten(),
                   1.0,1.0);
     ## from array (shapeless) to vector 
     W = numpy.array(W).reshape(y.shape)

     if VERIFY:
         Q = C.toarray()@x + y.toarray() 
         print("DIFF", numpy.sum(Q-W))

     
     return W
###
## rocBLAS style:
##    C has to be in column major layout
##    x is a dense vector
##    b may be not dense 
## z = 1*Cx + 1*y  
###
def dgemv(device, C, x, y) :
    
    W = rocmgpu.gemv(device,
                     C.flatten(order='F'),C.shape[0],
                     x,
                     y,
                     1.0,1.0);
    ## from array (shapeless) to vector 
    W = numpy.array(W).reshape(y.shape)
    
    if VERIFY:
        Q = C@x + y 
        print("DIFF", numpy.sum(Q-W))
        
     
    return W


###
## rocSPARSE style:
##    C has to be in column major layout
##    x is NOT a dense vector
##    b may be not dense 
## z = 1*Cx + 1*y  
###
def dgemv_csr(device, C, x, y) :

    W = rocmgpu.csr_mv(device, C.indptr.flatten(),C.indices.flatten(), C.data.flatten(),x.toarray().flatten(),y.toarray().flatten(),1.0,1.0); 
    W = csr_matrix(numpy.array(W).reshape(y.shape))
    
    if VERIFY:
        Q =  C@x + y
        print("DIFF", numpy.sum(Q-W))
        
        
        
    return W


###
## rocBLAS style:
##    C has to be in column major layout
##    A is a dense vector
##    b may be not dense 
## z = 1*Cx + 1*y  
###
def dgemm_(device, L, R) :
     #import pdb; pdb.set_trace()
     try :
          V =  rocmgpu.gemm(
               device,
               L.A.flatten('F'), L.shape[0],
               R.A.flatten('F'), R.shape[0]
               
          )
     except Exception as e:
          print(e)
          print(type(L))
          print(type(R))
          V =  rocmgpu.gemm(
               device,
               L.flatten('F'), L.shape[0],
               R.flatten('F'), R.shape[0])
          
          import pdb; pdb.set_trace()
          
     B = numpy.matrix(
          V
     )
     B = B.reshape((L.shape[0],R.shape[1]), order='F')
     
     if VERIFY:
          Q =  L@R
          print("DIFF", numpy.sum(Q-B))
          import pdb; pdb.set_trace()
     return B

def dgemm(device, L, R) :
     #import pdb; pdb.set_trace()
     try :
          V = numpy.zeros((L.shape[0],R.shape[1]), dtype=L.dtype)
          V =  rocmgpu.gemm_(
               device,
               V, V.shape[0],
               L, L.shape[0],
               R, R.shape[0]
               
          )
     except Exception as e:
          print(e)
          print(type(L))
          print(type(R))
          import pdb; pdb.set_trace()
          

     #V = V.transpose()
     B = numpy.matrix(
          V.data
     )
     #B = B.reshape((L.shape[0],R.shape[1]), order='C')

     
     if   VERIFY:
          
          Q =  L@R
          print("DIFF", numpy.sum(Q-B))
          import pdb; pdb.set_trace()
          
     return B

###
## rocBLAS style:
##    C has to be in column major layout
##    A is a dense vector
##    b may be not dense 
## z = 1*Cx + 1*y  
###
def dgema_(device, L, R) :
     #import pdb; pdb.set_trace()
     try :
          V =  rocmgpu.gema(
               device,
               L.A.flatten('F'), L.shape[0],
               R.A.flatten('F'), R.shape[0]
          )
     except Exception as e:
          print(e)
          print(type(L))
          print(type(R))
          V =  rocmgpu.gema(
               device,
               L.flatten('F'), L.shape[0],
               R.flatten('F'), R.shape[0])
          
          import pdb; pdb.set_trace()
          
     B = numpy.matrix(
          V
     )
     
     B = B.reshape((L.shape[0],R.shape[1]), order='F')
     
     if VERIFY:
          Q =  L+R
          print("DIFF", numpy.sum(Q-B))
          
     return B
def dgema(device, L, R) :
     #import pdb; pdb.set_trace()
     try :
          V = numpy.zeros((L.shape[0], L.shape[1]), dtype=L.dtype)
          V =  rocmgpu.gema_(
               device,
               V, V.shape[0],
               L, L.shape[0],
               R, R.shape[0]
          )
     except Exception as e:
          print(e)
          print(type(L))
          print(type(R))
          import pdb; pdb.set_trace()
          V =  rocmgpu.gema(
               device,
               L.flatten('F'), L.shape[0],
               R.flatten('F'), R.shape[0])
          
          import pdb; pdb.set_trace()
          
     
     B = numpy.matrix(
          V.data
     )
     #B = B.reshape((L.shape[0],L.shape[1]), order='F')
     
     if VERIFY:
          Q =  L+R
          print(Q.shape, B.shape)
          print("DIFFA", Q-B)
          import pdb; pdb.set_trace()
                
     return B
