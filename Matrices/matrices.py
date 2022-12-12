import numpy 
import math


###
## A 2 dimensional matrix represented by a numpy matrix 
###
class Matrix:
    def __init__(self,
                 A : numpy.matrix
    ) :

        self.matrix = A
        self.min = (0,0)
        self.max = A.shape
        self.logicalshape = A.shape

        
    def value(self): return self.matrix
    def set_value(self, A):
        self.matrix[...] = A;
        return self.matrix 
        
    def __add__( self, A ):
        L = self.matrix
        R = A.value()

        if False:
            ls = L.shape
            rs = R.shape
            
            mx  = A.logicalshape
            if mx>ls:
                L = numpy.pad(L, [ (0, mx[0]-ls[0]),(0, mx[1]-ls[1])]) 
            if mx>rs:
                L = numpy.pad(R, [ (0, mx[0]-rs[0]),(0, mx[1]-rs[1])]) 
        
        return Matrix(L+R)
    def __sub__( self, A ):
        L = self.matrix
        R = A.value()

        if False:
            ls = L.shape
            rs = R.shape
            
            mx  = A.logicalshape
            if mx>ls:
                L = numpy.pad(L, [ (0, mx[0]-ls[0]),(0, mx[1]-ls[1])]) 
            if mx>rs:
                L = numpy.pad(R, [ (0, mx[0]-rs[0]),(0, mx[1]-rs[1])]) 
        
        return Matrix(L-R)

    def __mul__( self, A ):

        if type(A) is Scalar:
            ## B = alpha A
            return Matrix(numpy.matmul(self.value(), A.value()))
        elif type(A) is Matrix :
            ## SELF  * A (multiplication)
            L = self.value()
            R = A.value()
            return Matrix(numpy.matmul(L,R))#[:,0:k],R[0:k,:] ))
        elif type(A) is Vector:
            ## A*v = w
            return Vector(numpy.matmul(self.value(), A.value()))

    def __str__(self) :
        return str(self.value().shape)

    ## this makes sense only in a partition.
    def disjoint(self, A, shape):
        a = numpy.dot(self.min,(shape[1],1)) <= \
            numpy.dot(A.max,(shape[1],1))
        a = a or numpy.dot(self.max,(shape[1],1)) <= \
            numpy.dot(A.min,(shape[1],1))
                
        return  a
    def space(self):
        return self.matrix.size*self.matrix.dtype.itemsize

    
###
## A partitions A_ij of A so that A_i /\ A_j = 0 and \/A_i = A. The
## main goal is to create disjoint matrices covering the original
## submatrices. This is like a concat from the combination of matrices
## to a larger one and viceversa.
###
    
class PartitionMatrix:
    def __init__(self,
                 A : Matrix,
                 logicalShape : list = None
                 
    ) :
        self.original = A
        self.logicalshape = logicalShape

        if logicalShape is None:
            self.logicalshape =  tuple(
                [int(math.ceil(i/2)) for i in A.value().shape]
            )
            logicalShape = self.logicalshape

        
        #print(A, self.logicalshape)
        #import pdb; pdb.set_trace()
        
        # we partition the matrix A in disjoint sub-matrices with up
        # to logical shape size and the union is the original matrix
        self.l = []
        
        
        matrix = A.value()
        shape  = matrix.shape
        
        m= [ (0,shape[0] %  self.logicalshape[0]),
             (0,shape[1] %  self.logicalshape[1])]


        ## yep, we pad the matrix to make sure that the fast
        ## algorithms can be applied without worries. There are better
        ## solution and Paolo should be alble to do better

        matrix = numpy.pad(matrix, m)
        shape  = matrix.shape
        
        for i in range(math.ceil(shape[0]/logicalShape[0])):
            row = []
            for j in range(math.ceil(shape[1]/logicalShape[1])):
                A = Matrix(matrix[
                    i*logicalShape[0]:min((i+1)*logicalShape[0],shape[0]),
                    j*logicalShape[1]:min((j+1)*logicalShape[1],shape[1])
                ])
                A.min = (i*logicalShape[0],j*logicalShape[1])
                A.max = (
                    min((i+1)*logicalShape[0],shape[0]),
                    min((j+1)*logicalShape[1],shape[1]))
                row.append(A)
                A.logicalshape =  self.logicalshape
            self.l.append(row)

        #for row in self.l:
        #    print([a.logicalshape for a in row ])
        #import pdb; pdb.set_trace()


    ###
    ## Because DeepMind store the "gamma" matrix in a transpose format
    ## and the computation for C_12 is actually for C_21 ... do not
    ## ask.
    ###
    
    def transpose(self):
        
        L = [ ]
        for j in range(len(self.l[0])):
            W = []
            for i in range(len(self.l)):
                W.append(self.l[i][j])
            L.append(W)
            
        self.l = L
        return self
            
    def value(self): return self.l

    ###
    ## would you like to consider if this partion is a real parition ?
    ## Disjoint set of matrices covering the original ?
    ## Ask.
    ###
    def partition(self):
        MM = self.value()

        cover = self.original.min == MM[0][0].min and \
                self.original.max == MM[-1][-1].max
        disjoint = True
        MM = [ item for sublist in MM for item in sublist]
        while len(MM)>1:
            m = MM.pop()
            for n in MM:
                if not m.disjoint(n,self.original.value().shape):
                    return False
        return cover

###
## A vector 1D
###
    
class Vector:
    def __init__(self,
                 A : numpy.array
    ) :
        self.vector = A
        
    def value(self): return self.vector

    def __add__(self, A ) :
        return self.value() + A.value()

    def __mul__(self, A) :

        
        if A is Matrix: ## a^t * A = v
            return Vector(numpy.matmul(self.value(), A.value()))
        else:
            ## either a^t * a = scalar
            ## of     a  *  a = matrix  
            Q = self.value() * A.value()
            if Q.shape[0] == Q.shape[1]  and Q.shape == 1: return Scala(Q[0][0])
            else: return Matrix(Q ) 
    def __str__(self) :
        return str(self.value().shape)

    def space(self):
        return self.vector.size*self.vector.dtype.itemsize

###
## Scalar. 
###
    
class Scalar:
    def __init__(self,
                 A : numpy.float64
    ) :
        self.scalar = A
        self.type = numpy.float64
        
    def value(self): return self.scalar

    def __str__(self) :
        return str(self.value())
    def __mul__(self, A ) :
        if type(A) is Matrix:
            return Matrix(self.value() * A.value())
        elif type(A) is Matrix:
            return Vector(self.value() * A.value())
        else:
            return Scalar(self.value() * A.value())
    def space(self):
        return 8


## This is an algorithm because it suggests a computation order by a
## permutation we know the order of the Gamma and our way to assign to
## a PE. It is still in the Matrix box. It may move in the future.
##
## inputs : tensor sequence for A (alpha)
##        : tensor for B (beta)
##        : tensor for C  (gamma)

class Algorithm:
    def __init__(
            self,
            alpha : numpy.ndarray,
            beta  : numpy.ndarray,
            gamma : numpy.ndarray
    ) :
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

    
        
    ### group tells me the number of different computation we want to
    ### have separated: 2 PEs ... 2 independent computations This is a
    ### bilinear then we compute the number of temporary P_i we need
    ### for each output and take the min_max. We return a permutation
    ### of the Gamma tensor so that we can split and we introduce the
    ### bound (we avoid any further comparison if any is larger than M
    def partition_by_output(self, group):

        sums = [ 0 for i in range(group)]        


        ## minimum of the maximum number of temporary needed y
        ## splitting the computation by k contiguous group
        def min_max(l, # permutation 
                    k, # cons group 
                    G, # gamma
                    M  # min_max so far
        ):
            for j in range(0, len(l), k):

                # we could do this faster
                temp = G[l[j:min((j+k), len(l))],:]
                q = sum(temp)
                qs = numpy.count_nonzero(q)
                if qs > M :
                    # no better, we quit 
                    return M
                sums[math.ceil(j/k)] = qs

            M1 = max(sums)
            return M1
        
        ## here come the combinatorial thing: we take every schedule
        ## of the C computation and  [ group chosen over n] partitions 
        from itertools import permutations
        G = self.gamma*1
        G[G!=0] = 1
        
        
        #import pdb; pdb.set_trace()
        M=G.shape[1]
        ## this many permutations
        N = numpy.math.factorial(self.gamma.shape[0])
        K = None
        count =  0
        for l in permutations(range(self.gamma.shape[0])):

            ## you will get bored at watching nothing happening
            count+=1
            if count %1000000 ==0:
                print(count, N)
            #if count % 2049 < 100: continue

            ## take every permutation
            MM = min_max(l,math.ceil(len(l)/group),G,M)
            if MM<M:
                ## if the max is smaller than before  
                M = MM
                K = l
                print(M,l)
        

        ## this is a permutation of indexes
        return K
            
        
        
def read_alpha(filename, ty):
    data = []
    with open(filename,'r') as F:
        data = F.read().split("\n")
        F.close()

    alpha = []
    beta  = []
    gamma = []
    """
product Gamma                        Alpha                        Beta
    1 ;   0 -1 -1  0  0  0  0  0  0 ;  1  0  0  1  0  0  0  0  0 ;  0 -1  0  0 -1  0  0 -1  0
    """
    print(data)
    for l in data:
        if l.find(";")<0 : continue
        print(l)
        l, g,a,b = l.split(";")
        gamma.append([ float(i) for i in g.strip().split()])
        alpha.append([ float(i) for i in a.strip().split()])
        beta.append([ float(i) for i in b.strip().split()])


    A =  numpy.ndarray(
        shape =(len(alpha), len(alpha[0])),
        dtype = ty,
        buffer = numpy.array(alpha)
    ).transpose()
    B =  numpy.ndarray(
        shape =(len(beta), len(beta[0])),
        dtype = ty,
        buffer = numpy.array(beta)
    ).transpose()
    C =  numpy.ndarray(
        shape =(len(gamma), len(gamma[0])),
        dtype = ty,
        buffer = numpy.array(gamma)
    ).transpose()

    return A,B,C
