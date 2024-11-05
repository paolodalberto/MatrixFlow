from tnorm import Norm, Nmul, LayerNorm,PplusE,PRplus,PRplusM, Nplus, Gg, square
from matrix import Vector, Matrix, Tiling
from splitting import *
import numpy
import scipy
import pdb
import copy

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
    def T_dim(self, A : Matrix): return Matrix(
            numpy.finfo(A.matrix.dtype).min  *
            numpy.ones((2,A.shape()[0]),dtype=A.matrix.dtype)
    )
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
## This is to emphasize that RMS Norm basic computation is exaclty
## like Euclidean Norm for the projection and exaclty like Layernorm
## for normalization.
##
## Tiling algorithm and computation is the same of Layer Norm:)
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
    def T_dim(self, A : Matrix): return Vector(
            numpy.zeros(A.shape()[0]).astype(A.matrix.dtype)
    )

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
            s = 1/numpy.sqrt(numpy.finfo(A.matrix.dtype).resolution +s)
        except:
            pdb.set_trace()
        s = Vector(s)
        #pdb.set_trace()
        B = self.N(A,s)
        B = self.N(B,Vector(GB.value()[0,:]), row=False)
        B = Nplus(B, Vector(GB.value()[1,:]) , row=False)
        return B


###
## to be filled
###



## Layer norm projections are sums (to compute mu and sigma)
def PplusInstance(A : Matrix, axis = 0) -> Vector:
    return Vector(numpy.sum(A.value(),axis=axis))



class InstanceNorm(LayerNorm):
    
    def __init__(
            self,
            P      = PplusInstance,
            R      = PRplusM,
            N      = Nmul,
            G      = Gg):
        
        LayerNorm.__init__(self,P,R,N,G)
        self.Qr = Qc
        self.Qr_ = Qc_
        self.Qc = Qr
        self.Qc_ = Qr_
        self.Qrc_ = Qcr_

    def t_dim(self, A : Matrix) : return (2,A.shape()[1])
    def T_dim(self, A : Matrix): return Vector(
            numpy.zeros((2,A.shape()[1])).astype(A.matrix.dtype)
    )

    def direction(self): return 1
        
    ## (Kernel computation)
    def pass_one(self,A      : Matrix) -> Matrix:
        A.color +=1
        #pdb.set_trace()
        SUM  = self.P(A)
        SUMQ = self.P(self.G(A,square))
        CSUM= numpy.zeros((2,A.shape()[1]))
        CSUM[0,:] = SUM.value()
        CSUM[1,:] = SUMQ.value()
        return Matrix(CSUM)
    
    ## (Kernel computation)
    def pass_two(self,
                 A      : Matrix,
                 GB     : Matrix,
                 CSUM    :Matrix
                 ):

        A.color +=1
        GB.color +=1
        
        # sequential and sync with pass one
        N = max(A.shape()[1],self.A.shape()[1] if self.A else 0)
        M = max(A.shape()[0],self.A.shape()[0] if self.A else 0)

        ## one pass computation of the mu and sigma
        mu  = CSUM.value()[0,:]/M
        mu2 = (CSUM.value()[0,:]/numpy.sqrt(M))**2
        #pdb.set_trace()
        #print(mu)
        
        try:
            s = 1/numpy.sqrt(numpy.finfo(A.matrix.dtype).resolution +(CSUM.value()[1,:] - mu2)/M)
        except:
            pdb.set_trace()
        #print(s)
        mu = Vector(-mu*s)
        s = Vector(s)
        
        #print(A.value())
        B = self.N(A,s, row=False)
        #print(B.value())
        B = Nplus(B,mu,row=False)
        #print(B.value())
        B = self.N(B,Vector(GB.value()[0,:]),row=False)
        B = Nplus(B, Vector(GB.value()[1,:]),row=False)
        #print(B.value())
        return B


    
  
        
if __name__ == "__main__":


    #import pdb
    shape =  (512,4096)
    dt = numpy.float32

    if True:
        A = numpy.random.rand(*shape).astype(dt)
        A1 = A*1.0
        #pdb.set_trace()
        
        N = SoftMax()
        N.comp(Matrix(A))
        
        
        R1 = scipy.special.softmax(A1,1)

        print("SM MAX ERROR A", numpy.max(numpy.fabs(R1-A)))
        #pdb.set_trace()

        A2 = A1 *1.0 + 0.0
        #pdb.set_trace()
        N.comp_uni(Matrix(A2))
    
    
        print("SM MAX ERROR B", numpy.max(numpy.fabs(R1-A2)))
        pdb.set_trace()

    if False:
        A = numpy.random.rand(*shape).astype(dt)
        Gamma = numpy.ones(shape[1]).astype(dt)
        Beta = numpy.zeros(shape[1]).astype(dt)
        GB = numpy.zeros((2,shape[1]))
        GB[0,:] = Gamma
        GB[1,:] = Beta
        
        A1 = A*1.0
        #pdb.set_trace()
        
        N = RMSNorm()
        AA = Matrix(A1)
        GGB = Matrix(GB)
        N.comp(AA,GGB)
        
        var = numpy.average(A**2,axis=-1)
        s = 1/numpy.sqrt(numpy.finfo(A.dtype).resolution +var)
        
        
        R1 = A*s[:,None]*Gamma + Beta

        print("RMS MAX ERROR A", numpy.max(numpy.fabs(R1-A1)))
        #pdb.set_trace()

        A2 = A *1.0 + 0.0
        #pdb.set_trace()
        N.comp_uni(Matrix(A2),GGB)
    
    
        print("RMS MAX ERROR B", numpy.max(numpy.fabs(R1-A2)))
        pdb.set_trace()
    if True:
        ## instance Norm

        shape =  (512,4096)
        A = numpy.random.rand(*shape).astype(dt).transpose()
        shape =  A.shape
        A1 = A*1.0 + 0.0
        if False:
            Gamma = numpy.random.rand(shape[1]).astype(dt)
            Beta = numpy.random.rand(shape[1]).astype(dt)
        else:
            Gamma = numpy.ones(shape[1]).astype(dt)
            Beta = numpy.zeros(shape[1]).astype(dt)
            
        GB = numpy.zeros((2,shape[1])).astype(dt)
        GB[0,:] = Gamma
        GB[1,:] = Beta
        
        
        #Computation using numpy
        mu = numpy.average(A,axis=0)
        var = numpy.var(A,axis=0)
        s = 1/numpy.sqrt(numpy.finfo(A.dtype).resolution +var)
        
        mu = mu*s
        R1 = (A*s-mu)*Gamma + Beta

        N = InstanceNorm()
        AA = Matrix(copy.deepcopy(A))
        GGB = Matrix(GB)

        ## computation as matrix
        N.comp(AA,GGB)
        print("IN MAX ERROR A", numpy.max(numpy.fabs(R1-AA.value())))
        #pdb.set_trace()

        BB = Matrix(copy.deepcopy(A))
        #pdb.set_trace()
        N.comp_uni(BB,GGB)
        print("IN MAX ERROR B", numpy.max(numpy.fabs(R1-BB.value())))
        pdb.set_trace()
