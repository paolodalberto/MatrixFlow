import numpy 
import pdb

####
## Authors notes, Paolo D'Alberto
## 
## We are going to work on matrices mostly and in here we provide the
## basic definition and tools to describes the operands and partitions
## of them. A Partition of a matrix is the logical and phisical
## splitting such that each part is a non overlapping part and full
## coverage. We introduce here Vectors, Matrices, and Tiling
## Partitions for AIE 4x4 



###
## A vector 1D or a part of it between min and max
###
    
class Vector:
    def __init__(self,
                 A : numpy.array ## we must have data to partition
    ) :
        
        self.vector = A
        self.min = 0  
        self.max = A.shape[0]

    
    def value(self): return self.vector[self.min:self.max]
    def set_value(self,A):
        self.value()[...] = A;
        return self

    def shape(self) :
        return  [self.max -self.min] 
    def __str__(self) :
        return str(self.value().shape)
    # A = B + C
    def __add__(self, A ) :
        return Vector(self.value() + A.value())
    # A += B
    def __iadd__(self, A ) :
        return self.set_value(self.value() + A.value())
    # Space in number of elements
    def space(self):
        return self.value().size#*self.vector.dtype.itemsize


###
## A 2 dimensional matrix: in this way we can take the projection of
## the projection
###
class Matrix:
    def __init__(self,
                 A : numpy.ndarray
    ) :
        self.matrix = A ## every body will have a copy 
        self.min = (0,0)
        if A is None :   self.max = (0,0)
        else:            self.max = A.shape

        self.color   = 0
        self.sub = None

    def space(self):
        return self.value().size#*self.matrix.dtype.itemsize//8
    # A = B+C
    def __add__( self, A ):
        L = self.value()
        R = A.value()
        return Matrix(L+R)
    # A +=B
    def __iadd__(self, A ) :
        return self.set_value(self.value() + A.value())

    # A = B*C Matrix multiplication 
    def __mul__( self, A ):
        if type(A) is Scalar:
            ## B = alpha A
            return Matrix(self.value()*A.value())
        elif type(A) in [int,float]:
            ## B = alpha A
            return Matrix(A*self.value())
        elif type(A) is Matrix :
            ## SELF  * A (multiplication)
            L = self.value()
            R = A.value()
            B= numpy.matmul(L,R)
            C = Matrix(B)
            C.gpu = True
            return C
        elif type(A) is Vector:
            ## A*v = w
            return Vector(numpy.matmul(self.value(), A.value()))
    # A *= B 
    def __imul__( self, A ):

        if type(A) in [int,float]:
            ## B = alpha A
            return self.set_value(A*self.value())
        elif type(A) is Matrix :
            ## SELF  * A (multiplication)
            
            L = self.value()
            R = A.value()
            return  self.set_value(numpy.matmul(L,R))
        elif type(A) is Vector:
            ## A*v = w
            return Vector(numpy.matmul(self.value(), A.value()))

    
    
    def shapes(self):
        return  self.matrix.shape, self.min, self.max
    def shape(self):
        return   [ self.max[i] -self.min[i] for i in range(2)]
    def value(self):
        return self.matrix[self.min[0]:self.max[0],self.min[1]:self.max[1]]
    def set_value(self, A):
        self.value()[...] = A;
        return self
    def __str__(self) :
        return str(self.value().shape)

    ## partitioning relative to the current min max partition
    def part(self, v : list):
        ## [m,M],[n,N] = v
        ## A[m*M : (m+1)*M, n*N : (n+1)*N]
        shape = self.shape()
        ## this is a logical sub partition ...
        if v[0][1]>shape[0] or v[1][1] > shape[1]  or  \
           v[0][1]<v[0][0] or v[1][1] < v[1][0] :
            print(v)
            print(shape)
            print(self.matrix.shape)
            pdb.set_trace()
            return None

        m = [ self.min[i]+v[i][0] for i in range(2) ]  
        M = [
            min ( self.min[i]+v[i][1],
                  min(
                      self.max[i],
                      self.matrix.shape[i]
                  )
                 )
            for i in range(2)
        ]
        A =  Matrix(self.matrix)
        A.min = m
        A.max = M
        A.sub = True
        return A

## Non overlapping Partitioning of a Matrix this is a data structure
## we can visit as we like
##
## Given a Matrix, Tiling is a list where each element is either a
## Tiling or a Matrix. The leaf of this tiling makes a complete
## partition and coverage of the original Matrix.
## Tiling -: [
##             M0, M1, M2
##           ] and M0 | M1 | M2 = M
## or 
## Tiling -: [
##              T0, T1, T2
##           ] 
## 
##
##     
class Tiling:

    ## MAtrix, pre-partition, not used
    def __init__(self, buffer : Matrix, partition : list = [], pred  = None ):
        self.partition = partition
        self.buffer_   = buffer
        self.tile_     = None
        if len(partition)!=0:   self.tile_ = self.partition[0]
        self.pred = pred
        self.properties = {}
        self.depth =  1
        
    def getlist(self): return self.partition

    def leaf_count(self):
        L = len(self.partition)
        for j in range(L-1):
            if type(d) in  [Matrix,str] :
                return d.color
            else:
                return leaf_count(d)
        return 1
    
    ## Q is a splitting function: A -> list [ A0, A1 ... ]
    def traversal(self, Q):
        self.partition = Q(self.buffer_)
        self.tile_ = self.partition[0]

    ## Q is a list of splitting functions this is a little abstract
    ## but it is a useful routine when we know how the partition works
    ## at different level of the memory hierarchy. 
    def rec_traversal(self, Q : list ) -> int :

        if len(Q)>0 :
            self.partition = Q[0](self.buffer_)
            self.tile_ = self.partition[0]
            #print("T", Q[0],self.buffer_,self.partition)
        
            
            if len(Q)==1 : return 0
            
            L = len(self.partition)-1
            for i in range(L):
                self.partition[i] = Tiling(self.partition[i])
                self.partition[i].rec_traversal(Q[1:])
            #pdb.set_trace()
            #print("R",self)
        return 0
        
    def get_tile(self)     -> Matrix: return self.tile_
    def get_buffer(self)   -> Matrix: return self.buffer_
    def get_partition(self)-> list: return self.partition

    
    ## Traversal = Spatial * Temporal
    ##
    ## This is my revelation how MLADF describe the tiling: 1) there
    ## is a spatial partition and after the spatial we need to choose
    ## a tile and how to traverse each spatial by a temporal visit
    ## L3 -> L2 : spatial L3 to 4 buffers in memtile
    ##            Take a tile and traverse in time the spatial buffer
    ## L2 -> L1 : spatial L2 to 4 buffers in L1 core
    ##            Take a tile and traverse in time the spatial buffer
    def spatial_temporal(self, Q_s, Q_t):
        self.partition = Q_s(self.buffer_)
        L = len(self.partition)-1
        for i in  range(L):
            self.partition[i] = Tiling(self.partition[i])
            self.partition[i].traversal(Q_t)
        self.tile_ = self.partition[0].buffer_

    def len(self):
        return len(self.partition)

    ## this is a template how to visit the Tiling data structure and
    ## will be used for the norm computation. Here is used mostly to
    ## represent as string the data structure.
    def visit(self, level : int =0):
             
        ## description of the head shape, type, level
        ident =  "\n"+"\t".join(["" for i in range(level+1) ])
        res = ident+"Level %s %s %d " % (
            str(self.buffer_.shape() if type(self.buffer_) is Matrix else self.buffer_.shape),
            self.partition[-1],level
        )
        # the last element in the partition is a str describing
        # concisely how we split the matrix
        L = len(self.partition)-1

        for j in range(min(self.depth,L)):
            d = self.partition[j]
            if type(d) is Matrix:
                ashape = str(d.shape())
                a = "%s-%d/%d I:%d" % ( ashape,j,L-1,d.color)
                res +=  ident+"a "+a
                continue
            else:
                # Left buffer
                #import pdb; pdb.set_trace()
                ashape = str((d.get_buffer()).shape()  )
                a = "%s+ %d/%d" % ( ashape,j,L-1)
                b = d.visit(level+1)
            res+= ident+"a "+a
            res+= ident+"b "+b
            
        return res
    ## We are interested in norms and the computation is asymmetric
    ## the columns have a particular flavor to it
    def max_col(self) :
        res = [self.tile_.shape()[1]]
        l = min(1, len(self.partition))
        for i in range(l):
            d = self.partition[i]
            if type(d) is Tiling:
                res += d.max_col()
        return res
        
    ## Collecting all the leaves of the partition
    def flat(self):
        res = []
        L = len(self.partition)
        if L<=1 : return []
        res.append([self.partition[L-1],L-1])
        for j in range(L-1):
            d = self.partition[j]
            if type(d) in  [Matrix,str] :
                res.append(d)
            else:
                res += d.flat()
        return res

    ## Collecting all the leaves of the partition
    def stream(self):
        res = []
        L = len(self.partition)
        if L<=1 : return []
        res.append([self.partition[L-1],L-1])
        ty = self.partition[L-1]
        for j in range(L-1):
            d = self.partition[j]
            if type(d) in  [Matrix,str] :
                res.append(d)
            else:
                C = d.flat()
                F = d.leaf_count()
                if ty[0] in ["r"] and d.partition[-1] in ['c']:
                    for i in range(F):
                        T +=  C
                else:
                    T = C

                res += T
                
        return res

    ## we like to visualize the Data Structure 
    def __str__(self):
        #import pdb; pdb.set_trace()
        return self.visit()
