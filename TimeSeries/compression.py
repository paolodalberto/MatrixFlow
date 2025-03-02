import zlib
import numpy
import scipy
import matplotlib.pyplot as plt

import pdb

def compression_measure(A : numpy.ndarray, B : numpy.ndarray) -> float :

    #pdb.set_trace()
    

    AB = numpy.concatenate((A,B),axis=0)

    CAB = len(zlib.compress(AB))
    CB  = len(zlib.compress(B))
    CA  = len(zlib.compress(A))

    return (CAB -min(CB,CA))/max(CA,CB) 

def build_h0(A : numpy.ndarray) -> numpy.ndarray :

    res = numpy.zeros(A.shape[0]//2)
    for i in range(2, A.shape[0]-2,2):
        res[i//2] = compression_measure(A[0:i,:], A[i:,:])

    return numpy.sort(res)


def build_h1(A : numpy.ndarray, B: numpy.ndarray) -> numpy.ndarray :

    res = numpy.zeros(min(100,A.shape[0]))
    for i in range(1, min(100,A.shape[0])):
        res[i] = compression_measure(A[0:i,:], B[:j,:])

    return numpy.sort(res)

def pval(d : float, dis = numpy.ndarray):

        
    return sum(d>dis)/dis.shape[0]

        
        

    


if __name__ == '__main__':

    import matplotlib.pyplot as plt


    if True:
        print("Testing equality")
        QQ = 500
        M = 10
        #pdb.set_trace()
        X = numpy.zeros((QQ,M))
        Y = numpy.zeros((QQ,M))

        
        for i in range(QQ):
            X[i,:] = scipy.stats.norm().rvs(M)
            Y[i,:] = scipy.stats.norm().rvs(M)
            #Y.append(numpy.random.rand(M)-0/2)
        pdb.set_trace()
        boot = build_h0(numpy.concatenate((X,X)))
        dis  = compression_measure(X,Y)
        print(dis, pval(dis,boot))
        pdb.set_trace()
        print("Testing inequality")
        QQ = 500
        M = 10
        X = numpy.zeros((QQ,M))
        Y = numpy.zeros((QQ,M))
        for i in range(QQ):
            X[i,:] = scipy.stats.norm().rvs(M)
            Y[i,:] = numpy.random.rand(M)-0/2
            #Y.append(numpy.random.rand(M)-0/2)
            
        dis  = compression_measure(X,Y)
        print(dis, pval(dis,boot))

