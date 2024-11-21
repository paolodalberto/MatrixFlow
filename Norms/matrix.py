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
    def mladf_shape(self):
        return   list(reversed(self.shape()))
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
        self.properties = {'temporal' : True }
        self.depth =  1
        
    def getlist(self): return self.partition

    def leaf_count(self):
        L = len(self.partition)
        for j in range(L-1):
            d = self.partition[j]
            if type(d) in  [Matrix,str] :
                return d.color
            else:
                return d.leaf_count()
        return 1
    
    ## Q is a splitting function: A -> list [ A0, A1 ... ]
    def traversal(self, Q):
        self.partition = Q(self.buffer_)
        self.tile_ = self.partition[0]

    def core_spatial(self, Q):
        self.properties['ofm'] = 1
        L = len(self.partition)-1
        for i in range(L):
            d = self.partition[i]
            if type(d) is Matrix:
                T = Tiling(self.partition[i])
                T.traversal(Q)
                T.properties['temporal'] = False
                self.partition[i] = T
            else:
                d.core_spatial(Q)

        
    ## Q is a list of splitting functions this is a little abstract
    ## but it is a useful routine when we know how the partition works
    ## at different level of the memory hierarchy. 
    def rec_traversal(self, Q : list ,S : list ) -> int :

        if len(Q)>0 :
            
            self.partition = Q[0](self.buffer_)
            self.tile_ = self.partition[0]
            self.properties['temporal'] =S[0]=='t'
            #print("T", Q[0],self.buffer_,self.partition)
        
            
            if len(Q)==1 : return 0
            
            L = len(self.partition)-1
            for i in range(L):
                self.partition[i] = Tiling(self.partition[i])
                self.partition[i].rec_traversal(Q[1:],S[1:])
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

    ## this is a template how to visit the Tiling data structure and
    ## will be used for the norm computation. Here is used mostly to
    ## represent as string the data structure.


    """
    from GELU
    
    Paolo Notes:
    Split transfer per column, there are 4 and there is an offset for
    gelu every thing is 1D
  
// IFM L3, 
std::vector<uint32_t> ifm_L3_dim = {L3_SZ};
std::vector<adf::access_pattern> ifm_L3_pattern_out = {
    adf::tiling({.buffer_dimension = ifm_L3_dim,
            .tiling_dimension = ifm_L2_dim,
            .offset = {0 * (L3_SZ / COLS)},
            .tile_traversal = {{.dimension = 0, .stride = ifm_L2_dim[0], .wrap = ifm_L3_dim[0]/ifm_L2_dim[0]/COLS}}}),
    adf::tiling({.buffer_dimension = ifm_L3_dim,
            .tiling_dimension = ifm_L2_dim,
            .offset = {1 * (L3_SZ / COLS)},
            .tile_traversal = {{.dimension = 0, .stride = ifm_L2_dim[0], .wrap = ifm_L3_dim[0]/ifm_L2_dim[0]/COLS}}}),
    adf::tiling({.buffer_dimension = ifm_L3_dim,
            .tiling_dimension = ifm_L2_dim,
            .offset = {2 * (L3_SZ / COLS)},
            .tile_traversal = {{.dimension = 0, .stride = ifm_L2_dim[0], .wrap = ifm_L3_dim[0]/ifm_L2_dim[0]/COLS}}}),
    adf::tiling({.buffer_dimension = ifm_L3_dim,
            .tiling_dimension = ifm_L2_dim,
            .offset = {3 * (L3_SZ / COLS)},
            .tile_traversal = {{.dimension = 0, .stride = ifm_L2_dim[0], .wrap = ifm_L3_dim[0]/ifm_L2_dim[0]/COLS}}})
};
    Paolo Notes:
    There are 4 cores and the communication is split spatially. 

// IFM L2 
std::vector<uint32_t> ifm_L2_dim = {L2_SZ};
uint32_t ifm_L2_ping_addr = 0;
uint32_t ifm_L2_pong_addr = 262144;

std::vector<adf::access_pattern> ifm_L2_pattern_out = {
    adf::tiling({.buffer_dimension = ifm_L2_dim,
            .tiling_dimension = {L1_SZ},
            .offset = {0},
            .tile_traversal = {{.dimension = 0,
                                .stride = L1_SZ * ROWS,
                                .wrap = ifm_L2_dim[0]/L1_SZ/ROWS}}}),

    adf::tiling({.buffer_dimension = ifm_L2_dim,
            .tiling_dimension = {L1_SZ},
            .offset = {L1_SZ},
            .tile_traversal = {{.dimension = 0,
                                .stride = L1_SZ * ROWS,
                                .wrap = ifm_L2_dim[0]/L1_SZ/ROWS}}}),

    adf::tiling({.buffer_dimension = ifm_L2_dim,
            .tiling_dimension = {L1_SZ},
            .offset = {2 * L1_SZ},
            .tile_traversal = {{.dimension = 0,
                                .stride = L1_SZ * ROWS,
                                .wrap = ifm_L2_dim[0]/L1_SZ/ROWS}}}),

    adf::tiling({.buffer_dimension = ifm_L2_dim,
            .tiling_dimension = {L1_SZ},
            .offset = {3 * L1_SZ},
            .tile_traversal = {{.dimension = 0,
                                .stride = L1_SZ * ROWS,
                                .wrap = ifm_L2_dim[0]/L1_SZ/ROWS}}})
};
std::vector<adf::access_pattern> ifm_L2_pattern_in = {
    adf::tiling({.buffer_dimension = ifm_L2_dim,
            .tiling_dimension = ifm_L2_dim,
            .offset = {0}})};



    """

    """
    from layer Norm
    
    From L3 -> L2 the IFM is a single memory of size 3x512KB
    So actually the tiling is temporal first 

    // L3 ifm
    std::vector<adf::access_pattern> input_ddr_read_pattern = {
        tiling({.buffer_dimension = {INNER_DIM, OUTER_DIM},
                .tiling_dimension = {INNER_DIM, OUTER_DIM / OUTER_ITER_P2},
                .offset = {0, 0},
                .tile_traversal = {
                    {.dimension = 0, .stride = INNER_DIM, .wrap = 1},
                    {.dimension = 1, .stride = OUTER_DIM / OUTER_ITER_P2, .wrap = OUTER_ITER_P2}}})};

    std::vector<adf::access_pattern> input_ddr_write_pattern = {
        tiling({.buffer_dimension = {INNER_DIM, OUTER_DIM / OUTER_ITER_P2},
                .tiling_dimension = {INNER_DIM, OUTER_DIM / OUTER_ITER_P2},
                .offset = {0, 0},
                .tile_traversal = {
                    {.dimension = 0, .stride = 0, .wrap = 1},
                    {.dimension = 1, .stride = 0, .wrap = 1}}})};


    // L2 ifm
    Now the L2 tile above is now split spatially and then by temporal
    

    std::vector<adf::access_pattern> ifm_pattern = {
        tiling({
            .buffer_dimension = {INNER_DIM, OUTER_DIM / OUTER_ITER_P2},
            .tiling_dimension = {INNER_SV_DIM, OUTER_DIM / OUTER_ITER_P2 / COLS}, // FIXME: inner dim may need 64 align
            .offset = {0, OUTER_DIM / OUTER_ITER_P2 / COLS * 0},
            .tile_traversal = {{.dimension = 0, .stride = INNER_SV_DIM, .wrap = INNER_DIM / INNER_SV_DIM}, //
                               {.dimension = 1, .stride = OUTER_DIM / OUTER_ITER_P2 / COLS, .wrap = 1}},
            .packet_port_id = -1,
            .repetition = 1,
        }),
        tiling({
            .buffer_dimension = {INNER_DIM, OUTER_DIM / OUTER_ITER_P2},
            .tiling_dimension = {INNER_SV_DIM, OUTER_DIM / OUTER_ITER_P2 / COLS}, // FIXME: inner dim may need 64 align
            .offset = {0, OUTER_DIM / OUTER_ITER_P2 / COLS * 1},
            .tile_traversal = {{.dimension = 0, .stride = INNER_SV_DIM, .wrap = INNER_DIM / INNER_SV_DIM}, //
                               {.dimension = 1, .stride = OUTER_DIM / OUTER_ITER_P2 / COLS, .wrap = 1}},
            .packet_port_id = -1,
            .repetition = 1,
        }),
        tiling({
            .buffer_dimension = {INNER_DIM, OUTER_DIM / OUTER_ITER_P2},
            .tiling_dimension = {INNER_SV_DIM, OUTER_DIM / OUTER_ITER_P2 / COLS}, // FIXME: inner dim may need 64 align
            .offset = {0, OUTER_DIM / OUTER_ITER_P2 / COLS * 2},
            .tile_traversal = {{.dimension = 0, .stride = INNER_SV_DIM, .wrap = INNER_DIM / INNER_SV_DIM}, //
                               {.dimension = 1, .stride = OUTER_DIM / OUTER_ITER_P2 / COLS, .wrap = 1}},
            .packet_port_id = -1,
            .repetition = 1,
        }),
        tiling({
            .buffer_dimension = {INNER_DIM, OUTER_DIM / OUTER_ITER_P2},
            .tiling_dimension = {INNER_SV_DIM, OUTER_DIM / OUTER_ITER_P2 / COLS}, // FIXME: inner dim may need 64 align
            .offset = {0, OUTER_DIM / OUTER_ITER_P2 / COLS * 3},
            .tile_traversal = {{.dimension = 0, .stride = INNER_SV_DIM, .wrap = INNER_DIM / INNER_SV_DIM},
                               {.dimension = 1, .stride = OUTER_DIM / OUTER_ITER_P2 / COLS, .wrap = 1}},
            .packet_port_id = -1,
            .repetition = 1,
        })};
    """

    ## L3 -> L2  spatial + temporal 
    ## L2 -> L1  temporal (because of the broadcast) 
    def traversal_mladf(self, level : int =0, at : int = 0, parallel = 'r'):
             
        ## description of the head shape, type, level
        ident =  "\n"+"\t".join(["" for i in range(level+1) ])
        # the last element in the partition is a str describing
        # concisely how we split the matrix

        Time = self.properties['temporal']
        Spatial = not Time

        typ = self.partition[-1]
        L = len(self.partition)-1 if Spatial else 1
        
        if level == at:
            
            print(self.properties)
            Count = self.leaf_count()
            
            B = self.get_buffer()

            R = []
            self.properties['temporal']
            for p in range(L):
                if Spatial and type(self.partition[p]) is Tiling:
                    ## spatial tile + temporal tile
                    #pdb.set_trace()
                    P = self.partition[p].partition[0]
                else:
                    ## temporal tile
                    
                    P = self.partition[p]
                #Count = self.partition[p].leaf_count()
                                

                T = P.get_buffer() if not type(P) is Matrix else P 
                tiling = {
                    '.buffer_dimension' : B.mladf_shape(),
                    '.tiling_dimension' : T.mladf_shape(),
                    '.offset'           : list(reversed([ T.min[i]-B.min[i] for i in range(len(T.min))])),
                    '.tile_traversal'   : [
                        {
                            '.dimension' : i,
                            '.stride' : T.mladf_shape()[i],
                            '.wrap'   : int(B.mladf_shape()[i]/ T.mladf_shape()[i])
                        }
                        for i in range(len(T.min))
                    ],
                    '.packet_port_id'   : -1,
                    '.repetition'       : 1 if typ == parallel else max(Count,1)
                }
                #print(tiling)
                R.append(tiling)
                if self.partition[L] == 'c':
                    ## this is only temporal 
                    return R
            
            return R
                    
            
        else:
            if Time:
                j =0
                d = self.partition[j]
                if not type(d) is Matrix and level<at :
                    return   d.traversal_mladf(level+1,at)
            else:
                R = []
                for j in range(L):
                    d = self.partition[j]
                    if not type(d) is Matrix and level<at :
                        R.append(d.traversal_mladf(level+1,at))
                return R
                
        print(level, at)
        return None

    def str_traversal(self,R,name, pk = -1):
        S = " { %s " % (name)
        T = """
        adf::tiling({.buffer_dimension = { %d, %d },
            .tiling_dimension = {%d , %d },
            .offset = {%d %d },
            .tile_traversal = {
                 {.dimension = %d, .stride = %d, .wrap = %d},
                 {.dimension = %d, .stride = %d, .wrap = %d}
             },
            .packet_port_id = %d,
            .repetition     = %d
        })
        """
        
        count = -1
        for l in R[:-1]:
            count +=1
            if type(l) is list:
                S += self.str_traversal(l, "// COL %d \n" % count, pk)
                continue

            
            trav =  l['.tile_traversal']
            S += T %( l['.buffer_dimension'][0],  l['.buffer_dimension'][1],
                      l['.tiling_dimension'][0],  l['.tiling_dimension'][1],
                      l['.offset'][0],  l['.offset'][1],
                      trav[0]['.dimension'],trav[0]['.stride'],trav[0]['.wrap'],
                      trav[1]['.dimension'],trav[1]['.stride'],trav[1]['.wrap'],
                      (pk if pk<0 else count), l['.repetition']) + ","
            
        count +=1
        l = R[-1]
        if type(l) is list:
            S += self.str_traversal(l, "// COL %d \n" % count,pk)
        else:
            trav =  l['.tile_traversal']
            S += T %( l['.buffer_dimension'][0],  l['.buffer_dimension'][1],
                      l['.tiling_dimension'][0],  l['.tiling_dimension'][1],
                      l['.offset'][0],  l['.offset'][1],
                      trav[0]['.dimension'],trav[0]['.stride'],trav[0]['.wrap'],
                      trav[1]['.dimension'],trav[1]['.stride'],trav[1]['.wrap'],
                      (pk if pk<0 else count), l['.repetition'])

        S += "};"
            
        return S
    
    def full_traversal(self,parallel='r') :
        L3 = self.traversal_mladf(0,0,parallel)
        print("L3 -> L2\n")
        print(self.str_traversal(L3,"// L3 ->L2\n"))
        
        L2 = self.traversal_mladf(0,1,parallel)
        print("L2 -> L1\n")
        print(self.str_traversal(L2,"L2 -> L1\n"))

        if 'ofm' in self.properties and self.properties['ofm']== 1:
            pdb.set_trace()
            L1 = self.traversal_mladf(0,3,parallel)
            print("L1 -> L1\n")
            print(self.str_traversal(L1,"L1 -> L2\n",pk=1))

            L2 += L1
        
        print(self)
        pdb.set_trace()
        #L1 = self.traversal_mladf(0,2)
        #print(L1)
        
        return L3 + L2  
        
    ## we like to visualize the Data Structure 
    def __str__(self):
        #import pdb; pdb.set_trace()
        return self.visit()
