
import math
import numpy
from matrix import Vector, Matrix, Tiling
import copy

## Authors notes: This is based on norm computation and thus Rows are
## a little special. There are three way to split a matrix
##
## By row, By column, Row-column
##


def Identity(A: Matrix): return [A, 'i']




## Horizontal partition/views 
def Qr(A : Matrix,
       r: int =4  ## horizontal number of parts
       ) -> list:
    ret = []
    M = math.ceil(A.shape()[0]/r)
    
    for i in range(r):
        v = [[i*M,min((i+1)*M,A.shape()[0])],  [0,A.shape()[1]]]
        ret.append(A.part(v))
    ret.append("r")
    return ret


def Qr_(A : Matrix,
        M: int =4 ## horizontal maximum elements 
        ) -> list:
    ret = []
    r = math.ceil(A.shape()[0]/M)
    for i in range(r):
        v = [[i*M,min((i+1)*M,A.shape()[0])],[0,A.shape()[1]]]
        ret.append(A.part(v))
    ret.append("r") 
    return ret


## copies (always parallel because they are copies)
def Cr(A : Matrix, r: int =4 ) -> list:
    ret = []
    for i in range(r):
        v = copy.deepcopy(A)
        ret.append(v)
    ret.append("r+")
    return ret

    

## vertical 
def Qc(A : Matrix,
       r : int =4 ## vertical number of parts
       ) -> list:
    ret = []
    M = math.ceil(A.shape()[1]/r)
    for i in range(r):
        v = [[0,A.shape()[0]],[i*M,min((i+1)*M,A.shape()[1])]]
        ret.append(A.part(v))
    ret.append("c")
    return ret


def Qc_(A : Matrix,
        M : int =4 ## vertical maximum elements
        ) -> list:
    ret = []
    r = math.ceil(A.shape()[1]/M)
    for i in range(r):
        v = [[0,A.shape()[0]],[i*M,min((i+1)*M,A.shape()[1])]]
        ret.append(A.part(v))

    ret.append("c")
    return ret



## split by row and then by column it is not really spatial temporal
## it is only temporal, I need to revisit this especially for the
## corner cases
def Qrc_(A : Matrix, M: int =4, N: int =2) -> Tiling:

    T = Tiling(A)
    def qr(a : Matrix) : Qr_(a,M)
    def qc(a : Matrix) : Qc_(a,N)

    ## is it nice ?
    T.spatial_temporal(qr,qc)
    T.partition[-1] = 'rc'

    return T

## split by row and then by column it is not really spatial temporal
## it is only temporal, I need to revisit this especially for the
## corner cases
def Qcr_(A : Matrix, M: int =4, N: int =2) -> Tiling:

    T = Tiling(A)
    def qr(a : Matrix) : Qr_(a,M)
    def qc(a : Matrix) : Qc_(a,N)

    ## is it nice ?
    T.spatial_temporal(qc,qr)
    T.partition[-1] = 'rc'

    return T

## Assume we have a space constraint (we use L) and if we use this
## space for a ping pong (2 ping pong 1 only ping).  We have a spatial
## partition and we wonder what is the largest partition that
## satisfies the constraints 
## 
## This is a splitting problem but it will be used only by an
## algorithm and the computation of a norm

def fit(spatial : Tiling,
        L, ## level size 
        Q, ## splitting function 
        mult, ## 2 ping pong 1 no ping
        gran  ## granulatiry of the computation 
        ) -> Tiling:
    DDRs = spatial.partition # list of matrices 
    #pdb.set_trace()
    for c in range(len(DDRs)-1):
        DDR_tf = None
        t = 1
        q = True
        while q:
            # can we partitions it 
            if (Q is Qr_ and t*gran>DDRs[c].shape()[0]) or \
               (Q is Qc_ and t*gran>DDRs[c].shape()[1]):
                q = False
                continue
            if (Q is Qr_ and  DDRs[c].shape()[0]%t*gran!=0) or \
               (Q is Qc_ and  DDRs[c].shape()[1]%t*gran!=0):
                t+=1
                ## The best partition is without remainder
                continue
            
            
            DDRt = Q(DDRs[c],t*gran)
            A = DDRt[0]
            if A is None:
                q = False
            elif mult*A.space()>L :
                
                if DDR_tf is None:
                    #print("No L
                    #computation",mult*size(DDRt[0].shape)>L, Q is Qr_
                    #, Q is Qc_,t*gran,DDRs[c].shape,DDRt[0].shape)
                    return None
                q = False
            else:
                # maximum
                DDR_tf = DDRt
                t+=1
        if DDR_tf is None: return None
        DDRs[c] = Tiling(DDRs[c], DDR_tf) 
        DDRs[c].pred = spatial
        
    return spatial

## As above but the parition willbe row and column together, in my
## head is row-column
##
##
def fit_qrc(spatial : Tiling, L, Q, mult, gran_1,gran_2) -> Tiling:
    
    DDRs = spatial.partition
    #pdb.set_trace()
    for c in range(len(DDRs)-1):
        DDR_tf = None
        shape = DDRs[c].shape()
        M = None
        for g1 in range(gran_1,shape[0]+1,gran_1):
            #pdb.set_trace()
            for g2 in range(gran_1,shape[0]+1,gran_2):
                if (DDRs[c].shape()[0]%g1!=0) or \
                   (DDRs[c].shape()[1]%g2!=0):
                    ## The best partition is without remainder
                    continue
                
                DDRt = Q(DDRs[c],g1,g2)
                #print(g1,g2)
                A = DDRt.get_tile()
                #print(A,mult*A.space(),L)
                if A is None or  mult*A.space()>L :
                    if M is None:
                        #print("No L
                        #computation",mult*size(DDRt[0].shape)>L, Q is Qr_
                        #, Q is Qc_,t*gran,DDRs[c].shape,DDRt[0].shape)
                        return None
                    break
                else:

                    if   M is None:  M = DDRt
                    elif A.space()> M.get_tile().space():
                        M = DDRt
                    #print(M[0][1][0])
        DDRs[c] = M
        DDRs[c].pred = spatial

    return spatial

