from tnorm import Norm, Nmul
from matrix import Vector, Matrix, Tiling
import numpy
import scipy
import pdb

## SoftMax Projection : Max and partial sums 
def Pplus(A : Matrix, axis = -1) -> Matrix:

    T = A.value()
    M = numpy.max(T, axis=-1)
    X = T - M[:,None]
    E = numpy.exp(X)
    P = numpy.sum(E,axis=-1)
    
    A.set_value(numpy.exp(T))
    GB = numpy.zeros((2,A.shape()[0]))
    GB[0,:] = M
    GB[1,:] = P
    #pdb.set_trace()
    return Matrix(GB)

## softmax partial projection 
def PRplusM(A : Matrix, PQ : Matrix) -> Matrix:
    #print(A)
    #print(PQ)
    M = A.value()[0,:]
    P = A.value()[1,:]


    PM = PQ.value()[0,:]
    PP = PQ.value()[1,:]
    #print(M, PM)
    #pdb.set_trace()
    M1 = numpy.maximum(M,PM)
    
    S = numpy.exp(M1-M)
    PS = numpy.exp(M1-PM)
    
    P = P/S + PP/PS 
    A.value()[0,:] = M1
    A.value()[1,:] = P
    return A



###
## This is to emphasize that SoftMax basic computation is exaclty like
## Euclidean Norm once we define the proper projection and the
## normalization is the same. 
##
## Tiling algorithm and computation is the same
###


class SoftMax(Norm):
    
    def __init__(
            self,
            P      = Pplus,
            R      = PRplusM,
            N      = Nmul,
            G      = None):
        
        Norm.__init__(self,P,R,N,G)
        
    def t_dim(self, A : Matrix): return (2, A.shape()[0])
    def T_dim(self, A : Matrix): return Matrix(numpy.finfo(A.matrix.dtype).min  * numpy.ones((2,A.shape()[0])))
    def pass_two(self,
                 A      : Matrix,
                 T      : Matrix
                 ):
        #pdb.set_trace()
        A.color +=1
        T.color +=1
        M = T.value()[0,:]
        P = T.value()[1,:]

        S = numpy.exp(M)
        P = Vector(1/(P*S))
        self.N(A,P)
        #A.set_value(A.value()/P[:,None])

        return A
    

        
if __name__ == "__main__":


    #import pdb
    shape =  (512,4096)


    if True:
        A = numpy.random.rand(*shape)
        A1 = A*1.0
        pdb.set_trace()
        
        N = SoftMax()
        N.comp(Matrix(A))
        
        
        R1 = scipy.special.softmax(A1,1)

        print("MAX ERROR A", numpy.max(numpy.fabs(R1-A)))
        #pdb.set_trace()

        A2 = A1 *1.0 + 0.0
        pdb.set_trace()
        N.comp_uni(Matrix(A2))
    
    
        print("MAX ERROR B", numpy.max(numpy.fabs(R1-A2)))
        pdb.set_trace()
