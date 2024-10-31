import numpy 
import pdb

###
## A vector 1D
###
    
class Vector:
    def __init__(self,
                 A : numpy.array
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

    def __add__(self, A ) :
        return self.value() + A.value()
    def __iadd__(self, A ) :
        return self.set_value(self.value() + A.value())

    def space(self):
        return self.value().size#*self.vector.dtype.itemsize


###
## A 2 dimensional matrix represented by a numpy matrix projection of
## projection create a copy and for the evaluation of the computation
## I would like to make it an inplace computation
## 
###
class Matrix:
    def __init__(self,
                 A : numpy.ndarray
    ) :
        self.matrix = A ## every body will have a copy 
        self.min = (0,0)
        if A is None :
            self.max = (0,0)
        else: 
            self.max = A.shape

        self.padded = False
        self.pointer = None
        self.gpu     = None
        self.color   = 0
        self.sub = None

    def space(self):
        return self.value().size#*self.matrix.dtype.itemsize//8

    def __add__( self, A ):
        L = self.value()
        R = A.value()
        return Matrix(L+R)
    def __iadd__(self, A ) :
        
        return self.set_value(self.value() + A.value())

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
class Tiling:
    def __init__(self, buffer : Matrix, partition : list = [], pred  = None ):
        self.partition = partition
        self.buffer_   = buffer
        self.tile_     = None
        if len(partition)!=0: 
            self.tile_ = self.partition[0]
        self.pred = pred
        self.properties = {}
        
    def getlist(self): return self.partition
        
    def traversal(self, Q):
        self.partition = Q(self.buffer_)
        self.tile_ = self.partition[0]

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
        
    def get_tile(self) -> Matrix: return self.tile_
    def get_buffer(self)->Matrix: return self.buffer_
    def get_partition(self)-> list: return self.partition
     
    ## Traversal = Spatial * Temporal 
    def spatial_temporal(self, Q_s, Q_t):
        self.partition = Q_s(self.buffer_)
        L = len(self.partition)-1
        for i in  range(L):
            self.partition[i] = Tiling(self.partition[i])
            self.partition[i].traversal(Q_t)
        self.tile_ = self.partition[0].buffer_

    def len(self):
        return len(self.partition)
        
    def visit(self, level : int =0):
        #import pdb; pdb.set_trace()
        
        ## description of the head shape, type, level
        ident =  "\n"+"\t".join(["" for i in range(level+1) ])
        res = ident+"Level %s %s %d " % (
            str(self.buffer_.shape() if type(self.buffer_) is Matrix else self.buffer_.shape),
            self.partition[-1],level
        )

        L = len(self.partition)-1
        for j in range(min(4,L)):
            d = self.partition[j]
            if type(d) is Matrix:
                ashape = str(d.shape())
                a = "%s-%d/%d " % ( ashape,j,L-1)
                res +=  ident+"a "+a
                continue
            else:
                # Left buffer
                #import pdb; pdb.set_trace()
                ashape = str((d.get_buffer()).shape()  )
                a = "%s+ %d/%d " % ( ashape,j,L-1)
                b = d.visit(level+1)
            res+= ident+"a "+a
            res+= ident+"b "+b
            
        return res

    def max_col(self) :
        res = [self.tile_.shape()[1]]
        l = min(1, len(self.partition))
        for i in range(l):
            d = self.partition[i]
            if type(d) is Tiling:
                res += d.max_col()
        return res
        
    
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

    def __str__(self):
        #import pdb; pdb.set_trace()
        return self.visit()
