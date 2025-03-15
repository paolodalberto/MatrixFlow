import zlib
import numpy
import scipy
import matplotlib.pyplot as plt

import pdb

###
## This is a general measure where the compression is by bytes (the
## binary representation of the sequence). 
##
## The idea is about the same of |A /\ B|/|A U B|: a relative measure.
## if A and B are similar compressing A, B, and AB they should all
## take the same size, or very close to it.
##
## CAB - min(CA,CB) ... approximate the XOR ... what is not share by the longer sequence
## if A !=B  then this difference <= MAX(CA,CB)
## if A==B, the difference =0 and the ratio is 
## OF course this is an approximaton and we are aming to something close to 1
###
def compression_measure(A : numpy.ndarray, B : numpy.ndarray) -> float :

    ## literally a longer sequence, but we could shuffle, sort them
    ## and compress ... but then we need to do something similar to A
    ## and B
    AB = numpy.concatenate((A,B),axis=0)

    CAB = len(zlib.compress(AB))
    CB  = len(zlib.compress(B))
    CA  = len(zlib.compress(A))

    ## you may have negative measures !
    return (CAB -min(CB,CA))/max(CA,CB) 


###
## The idea is to have a distribution of the measure when the
## assumption H0 is true, A and B are similar. We take the same
## sequence and create a bootstrap sequence and collect the measures
##
## The average measure should be small and a large sample will mean a
## probable difference
###

def build_h0(A : numpy.ndarray) -> numpy.ndarray :

    res = numpy.zeros(A.shape[0]//2)
    for i in range(2, A.shape[0]-2,2):
        res[i//2] = compression_measure(A[0:i,:], A[i:,:])

    return numpy.sort(res)

###
## The idea is to have a distribution of the measure when the
## assumption H1 is true, A and B are NOT similar. We take the same
## sequence and create a bootstrap sequence and collect the measures
##
###

def build_h1(A : numpy.ndarray, B: numpy.ndarray) -> numpy.ndarray :

    res = numpy.zeros(min(100,A.shape[0]))
    for i in range(1, min(100,A.shape[0])):
        res[i] = compression_measure(A[0:i,:], B[:j,:])

    return numpy.sort(res)


###
## The simple assumption is that a larger measure out in the
## distribution, is associate with a possible difference , this ratio
## provide the probability of error considering H0 true with this
## measure
def pval(d : float, dis = numpy.ndarray):
        
    return sum(d>dis)/dis.shape[0]
###
## The simple assumption is that a small measure out in the
## distribution, is associate with a possible difference , this ratio
## provide the probability of error considering H0 true with this
## measure
def pval_h1(d : float, dis = numpy.ndarray):
        
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

