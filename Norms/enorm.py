from tnorm import Norm
from matrix import Vector, Matrix, Tiling
import numpy
import scipy
import pdb

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
    
    return Matrix(GB)

def PRplusM(A : Matrix, P : Matrix) -> Matrix:

    M = A.value()[0,:]
    P = A.value()[1,:]

    PM = P.value()[0,:]
    PP = P.value()[1,:]
        
    M1 = numpy.max(M,PM)
    
    S = numpy.exp(M1-M)
    PS = numpy.exp(M1-PM)
    
    P = P/S + PP/PS 
    
    return Matrix(GB)




class SoftMax(Norm):
    
    def __init__(
            self,
            P      = Pplus,
            R      = PRplusM,
            N      = None,
            G      = None):
        
        Norm.__init__(self,P,R,N,G)
        
                
    
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
        P = P*S
        
        A.set_value(A.value()/P[:,None])

        return A
    

        
if __name__ == "__main__":


    #import pdb
    shape =  (128,512)


    if True:
        A = numpy.random.rand(*shape)
        A1 = A*1.0
        
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
