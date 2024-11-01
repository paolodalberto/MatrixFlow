from tnorm import Norm, Nmul, LayerNorm,PplusE,PRplus, Nplus
from matrix import Vector, Matrix, Tiling
import numpy
import scipy
import pdb

## SoftMax Projection : Max and partial sums 
def PplusMax(A : Matrix, axis = -1) -> Matrix:

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
def PRplusMax(A : Matrix, PQ : Matrix) -> Matrix:
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
## normalization. 
##
## Tiling algorithm and computation is the same :)
###




class SoftMax(Norm):
    
    def __init__(
            self,
            P      = PplusMax,
            R      = PRplusMax,
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
        




###
## This is to emphasize that SoftMax basic computation is exaclty like
## Euclidean Norm once we define the proper projection and the
## normalization. 
##
## Tiling algorithm and computation is the same :)
###




class RMSNorm(LayerNorm):
    
    def __init__(
            self,
            P      = PplusE,
            R      = PRplus,
            N      = Nmul,
            G      = None):
        
        LayerNorm.__init__(self,P,R,N,G)
        
    def t_dim(self, A : Matrix) : return A.shape()[0]
    def T_dim(self, A : Matrix): return Vector(numpy.zeros(A.shape()[0]))

    ## base projection of a matrix (Kernel computation)
    def pass_one(self, A  : Matrix):
        return self.P(A)

    ## (Kernel computation)
    def pass_two(self,
                 A      : Matrix,
                 GB     : Matrix,
                 CSUM    :Vector
                 ):

        A.color +=1
        GB.color +=1
        
        # sequential and sync with pass one
        N = max(A.shape()[1],self.A.shape()[1] if self.A else 0)
        M = A.shape()[0]

        ## one pass computation of the mu and sigma
        s  = CSUM.value()/N
        try:
            s = 1/numpy.sqrt(s)
        except:
            pdb.set_trace()
        s = Vector(s)
        #pdb.set_trace()
        B = self.N(A,s)
        B = self.N(B,Vector(GB.value()[0,:]), row=False)
        B = Nplus(B, Vector(GB.value()[1,:]) , row=False)
        return B

        
if __name__ == "__main__":


    #import pdb
    shape =  (512,4096)


    if False:
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

    if True:
        A = numpy.random.rand(*shape)
        Gamma = numpy.ones(shape[1])
        Beta = numpy.zeros(shape[1])
        GB = numpy.zeros((2,shape[1]))
        GB[0,:] = Gamma
        GB[1,:] = Beta
        
        A1 = A*1.0
        pdb.set_trace()
        
        N = RMSNorm()
        AA = Matrix(A1)
        GGB = Matrix(GB)
        N.comp(AA,GGB)
        
        var = numpy.average(A**2,axis=-1)
        s = 1/numpy.sqrt(var)
        
        
        R1 = A*s[:,None]*Gamma + Beta

        print("MAX ERROR A", numpy.max(numpy.fabs(R1-A1)))
        #pdb.set_trace()

        A2 = A *1.0 + 0.0
        pdb.set_trace()
        N.comp_uni(Matrix(A2),GGB)
    
    
        print("MAX ERROR B", numpy.max(numpy.fabs(R1-A2)))
        pdb.set_trace()
