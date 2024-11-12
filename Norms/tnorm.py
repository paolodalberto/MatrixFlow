import numpy 
import math
import os
import pdb
import copy

from matrix import Vector, Matrix, Tiling
from splitting import Qr, Qr_, Qc, Qc_, Qrc_,Cr, fit, fit_qrc, Identity


def square(x) : return x*x



###
##  A norm has two basic computation: Projections and Normalization
##  
##  Projections are function P(A[i,:]) -> Real or in other words from
##  a matrix KxL to a vector K
##
##  Normalization is taking the projection + a little extra and then
##  broad cast to the elements N(A[i,:], alpha) -> A[i,:]  
###



## Example of Projections 

## Layer norm projections are sums (to compute mu and sigma)
def PplusE(A : Matrix, axis = -1) -> Vector:
    return Vector(numpy.sum(A.value()**2,axis=axis))
## Layer norm projections are sums (to compute mu and sigma)
def Pplus(A : Matrix, axis = -1) -> Vector:
    return Vector(numpy.sum(A.value(),axis=axis))
## Partial sums combination: projections in parts
def PRplus(A : Vector, B : Vector) -> Vector:
    A += B
    return A
def PRplusM(A : Matrix, B : Matrix) -> Matrix:
    A += B
    return A

## softmax use max and partial sums
def Pmax(A : Matrix, axis = -1) -> Vector:
    return Vector(numpy.max(A.value(),axis=axis))

## Example of Normalizations: these are clumsy and can be done better
def Nmul(A: Matrix, B : Vector, row = True) -> Matrix:

    ashape = A.shape()
    bshape = B.shape()

    if bshape[0] == ashape[1] and not row :
        A.matrix[A.min[0]:A.max[0],A.min[1]:A.max[1]] *= B.value()[None,:]
    else:
        A.matrix[A.min[0]:A.max[0],A.min[1]:A.max[1]] *= B.value()[:,None]
    return A

## example of product norm
def Nplus(A : Matrix, B : Matrix, row = True) -> Matrix:
    ashape = A.shape()
    bshape = B.shape()

    if bshape[0] == ashape[1] and not row:
        A.matrix[A.min[0]:A.max[0],A.min[1]:A.max[1]] += B.value()[None,:]
    else:
        A.matrix[A.min[0]:A.max[0],A.min[1]:A.max[1]] += B.value()[:,None]
    return A



## example of Scalar operation to a matrix
def Gg(A : Matrix, g = square) -> Matrix:
    func = numpy.vectorize(g)
    B = Matrix((func(A.value())))
    return B


    
## This is the Euclidean norm x/||x||^2 This is the minimum
## computation

class Norm :
    def __init__(
            self,
            P      = PplusE, # Matrix reduction
            R      = PRplus, # Blocked reduction
            N      = Nmul,  # matrix broadcast
            G      = Gg     # internal 
    ):
        self.R     = R
        self.P     = P
        self.N     = N
        self.G     = G
        self.A     = None
        self.comp_IFM = self.comp_IFM_unified_overlay
        
        ## parallel
        self.Qr = Qr
        self.Qr_ = Qr_ 
        ## Reduction by row (across columns paritions)
        self.Qc = Qc
        self.Qc_ = Qc_
        ## this is not parallel but the reductions are expressed as a list of "c"
        self.Qrc_ = Qrc_

    def t_dim(self, A : Matrix) : return A.shape()[0]
    def T_dim(self, A : Matrix): return Vector(
            numpy.zeros(A.shape()[0]).astype(A.matrix.dtype)
    )

    def direction(self): return 0
    
    ## base projection of a matrix (Kernel computation)
    def pass_one(self, A  : Matrix):
        return self.P(A)

    ## base normalization of a matrix + factor T  (Kernel computation)
    def pass_two(self, A:Matrix, T:Matrix):
        L = self.A.shape()[1] if self.A else A.shape()[1]
        TT = Vector(1/numpy.sqrt(
            numpy.finfo(A.matrix.dtype).resolution + T.vector/L)) # average and sqrt
        self.N(A, TT)
        return A

    ## based computation on matrix 
    def comp(self,A:Matrix):   
        T = self.pass_one(A) 
        return self.pass_two(A,T)

    ## Tiling and computation AIE emulation, in place
    def comp_uni(self,A:Matrix):
        self.A = A

        ## we create the tiling 
        DDR = self.comp_IFM(self.A)
        print(DDR)
        ## we compute using the tiling 
        self.comp_visit(DDR)
        print("---------------------------------------\n With Colors")
        print(DDR)

    def reduction(self,s :str):
        if self.Qr == Qr and s =="r": return False
        if self.Qr == Qc and s =="c": return False
        return True

    def parallel(self,s :str):
        if self.Qr == Qr and s =="r": return True
        if self.Qr == Qc and s =="c": return True
        return False
        
    ## Basically: row tiles 'r' are parallel computations and 
    ##
    ##            columns   'c' we compute the projection and then the normalization
    ##            the algorithm will re-read the input for the two parts
    ##
    ## I love recursion. 
    ##
    def comp_visit(self,
                   Ti : Tiling,
                   level : int = 0,
                   T :Vector  = None) -> Vector  :

        DDRs = Ti.get_partition()
        ty = DDRs[-1]
        if self.parallel(ty[0]) :
            
            for j in range(len(DDRs)-1):
                d = DDRs[j]
                if type(d) is Matrix: self.comp(d)
                else:    self.comp_visit(d)
        else: #elif ty == 'c' :
            
            if T is None and level ==0 :
                ## first 'c'
                ## this should fit the core space
                
                T = self.T_dim(Ti.get_buffer())
                #T = Vector(numpy.zeros(Ti.get_buffer().shape()[0]))
                
                for j in range(len(DDRs)-1):
                    d = DDRs[j]
                    if type(d) is Matrix:    self.R(T,self.pass_one(d))
                    else: self.R(T,self.comp_visit(d, level = level+1))
                        
                for j in range(len(DDRs)-1):
                    d = DDRs[j]
                    if type(d) is Matrix:  self.pass_two(d,T)
                    else:            self.comp_visit(d,T = T)

            elif T is None and level>0 :
                ## we are deep into the projections
                
                j = 0
                #print(self.A.value())
                d = DDRs[0]
                T = self.T_dim(Ti.get_buffer())
                
                #T = Vector(numpy.zeros(Ti.get_buffer().shape()[0]))
                for j in range(len(DDRs)-1):
                    d = DDRs[j]
                    if type(d) is Matrix:    self.R(T,self.pass_one(d))
                    else: self.R(T,self.comp_visit(d, level = level+1))
                return T
            else:
                ## we are into the normalization
                for j in range(len(DDRs)-1):
                    d = DDRs[j]
                    if type(d) is Matrix:  self.pass_two(d,T)
                    else:              self.comp_visit(d,T=T)
                        
            #pdb.set_trace()
        ##print(DDRs[0][0])
        return None
      
    ## An exercise in building a tiling for norm
    ##
    ## A traversal is composed by a spatial division then by a
    ## temporal division (traversal of the spatial by a tile and order
    def comp_IFM_(self,
                A    :  Matrix, # original matrix
                rows : int =4,   # cores per column
                cols : int =4,   # cores per row
                L2   : int = 128*1024,  # size in elements L2 IFM
                L1   : int = 4*1024,    # size in elements Ping/Bank
                H    : int = 16,
                V    : int = 32
                ) -> list:

        Repetition = {'L3' : -1, 'L2' : -1, 'L1' : -1 }

        T = Tiling(A)

        ## we split A into 4 parts spatial
        ## L3 -> L2  spatial 4 By Columns

        def qr(A) : return self.Qr(A, cols)
        T.traversal(qr)

        pdb.set_trace()
        ## we try to determine the largest tile in L2 for which we
        ## split by rows and we use ping pong 
        Ts = fit(T,
                 L2,  ## L2 size (tile for L2)
                 self.Qr_, ## split by row
                 2,   ## double buffering 
                 rows ## minimum granularity
                 )

        if Ts:
            ## we ca traverse by row
            Repetition['L3'] =1
            #print(T.visit())
            ## L3 -> L2 temporal for every spatial (we literally build
            ## each traversal but it is not really needed because the
            ## traversal has to be the same)

            DDRs = Ts.getlist()
            ## L2 -> L1 if there is core splitting there will be
            ## spatial and then temporal,
            for s in range(len(DDRs)-1):
                
                ## parallel split // well broad cast ... 
                Q = fit( DDRs[s],
                         L1,  ## L1 size 
                         self.Qr_, ## split by row 
                         1,   ## ping pong
                         rows ## to feed each core 
                        )
                #print(Q)
                if Q is None:

                    ## One row no fit in L1 but we do reduction in L1
                    Q = fit( DDRs[s],
                             L1,  ## L1 size 
                             self.Qc_, ## split by column
                             1,   ## this is ping 
                             V*2   ## 32*2 
                            )  
                    if Q is None:
                        
                        ## we cannot have all rows at once
                        Q = fit_qrc( DDRs[s],
                                     L1,
                                     self.Qrc_,
                                     1, ## this is ping
                                     rows,
                                     V*2
                                )
                        if Q is None:
                            ## ARGH, we cannot do the computation
                            #print("No L computation")
                            return None
                        else:
                            Repetition['L1'] =1
                            Repetition['L2'] =2
                            
                    else:
                        ## we split the computation by 64 columns
                        ## we compute all rows, hurrah
                        Repetition['L1'] =1
                        Repetition['L2'] =2
                        #print("L2 repetition =2 ")
                else:
                    ## One row will fit in L1, thus there is no
                    ## repetition in L2 (only in L1)
                    Repetition['L2'] =1
                    Repetition['L1'] =2

                DDRs[s] = Q
        else:
            
            ## we need to change strategy and we have to read L3 twice
            ## the order is important because 
            #print("L3 repetition =2 ")
            Repetition['L3'] =2

            ## we read the whole matrix into columns 
            Ts = fit(T, L2, self.Qc_, 2,1)
            DDRs = Ts.getlist()
            ## L2 -> L1 if there is core splitting there will be spatial
            ## and then temporal, 
            for s in range(len(DDRs)-1):
                
                Q = fit( DDRs[s], L1, self.Qc_, 1, 2)
                if Q is None:
                    print(" We should not be here! ########")
                    pdb.set_trace()
                    Q = fit_qrc( DDRs[s], L1, self.Qrc_, 1, rows,2)
                    if Q is None:
                        
                        return None

                DDRs[s] = Q


        #print(Repetition)
        return T
    ## An exercise in building a tiling for norm
    ##
    ## A traversal is composed by a spatial division then by a
    ## temporal division (traversal of the spatial by a tile and order
    def comp_IFM_unified_overlay(
            self,
            A    :  Matrix, # original matrix
            rows : int =4,   # cores per column
            cols : int =4,   # cores per row
            L2   : int = 128*1024,  # size in elements L2 IFM
            L1   : int = 4*1024,    # size in elements Ping/Bank
            H    : int = 16,
            V    : int = 32
    ) -> list:

        Repetition = {'L3' : -1, 'L2' : -1, 'L1' : -1 }

        T = Tiling(A)

        ## there is no spatial split so we make a identity partition

        T.traversal(Identity)

        pdb.set_trace()
        ## we try to determine the largest tile in L2 for which we
        ## split by rows and we use ping pong 
        Ts = fit(T,
                 L2,       ## L2 size (tile for L2)
                 self.Qr_, ## split by row (parallel)
                 2,        ## double buffering 
                 rows*cols ## minimum granularity
                 )

        if Ts:
            # T.partition[0] is the temporal tiling L3 ->L2
            T = T.partition[0]

            # Now we do spatial and then temporal
            TP = T.partition
            L = len(TP)-1
            for i in range(L):
                ## for each temporal tile, we do tiling
                Q = Tiling(TP[i])

                #spatial split
                def qr(A) : return self.Qr(A, cols)
                Q.traversal(qr)

                ## temporal row ?
                Ts = fit(Q,
                         L1,       ## L2 size (tile for L2)
                         self.Qr_, ## split by row (parallel)
                         1,        ## double buffering 
                         rows      ## minimum granularity
                         )

                if not Ts is None: TP[i] = Q; continue
                # temporal column ? 
                Ts = fit(Q,
                         L1,       ## L2 size (tile for L2)
                         self.Qc_, ## split by row (parallel)
                         1,        ## double buffering 
                         2*V       ## minimum granularity
                        )

                if not Ts is None: TP[i] = Q;continue
                ## temporal rw x col
                Ts = fit(Q,
                         L1,       ## L2 size (tile for L2)
                         self.Qrc_, ## split by row (parallel)
                         1,        ## double buffering 
                         rows,
                         2*V       ## minimum granularity
                         )
                if Ts is None:
                    return None

                pdb.set_trace()
                TP[i] = Q
                
            
        else:
            
            ## we need to change strategy and we have to read L3 twice
            ## the order is important because 
            #print("L3 repetition =2 ")
            Repetition['L3'] =2

            ## we read the whole matrix into columns 
            Ts = fit(T, L2, self.Qc_,
                     2, # double buffering 
                     2  # 2 columns (32B read) 
                     )

            if Ts is None: return None

            TP = T.partition
            L = len(TP)-1
            for i in range(L):
                ## for each temporal tile, we do tiling
                Q = Tiling(TP[i])
                Ts = fit( Q, L1, self.Qc_, 1, 2)
                if not Ts is None: continue
                
                Ts = fit_qrc( Q, L1, self.Qrc_, 1, rows,2)
                if Ts is None: return None
            
        pdb.set_trace()
        print(Repetition)
        return T



###
##  Layer Norm normalize by mu and sigma but it has a "column"
##  normalization as well by gamma and beta.
##
##  Because the computation of mu and sigma actually drives the space
##  constraints there is a tiling computation of A and the (gamma,
##  beta) tiling is subservient to the one of A.
##
##  This is true for the broad cast / unified overlay where each core
##  need to store 4 rows while gamma and beta takes 2
##
###
    

class LayerNorm(Norm):
    def __init__(
            self,
            P      = Pplus,
            R      = PRplusM,
            N      = Nmul,
            G      = Gg):

        Norm.__init__(self,P,R,N,G)
        self.GB = None

    ## the partial results will need temporary tensors these are also
    ## the accumulators and we may use different precisions
    def t_dim(self, A : Matrix) : return (2,A.shape()[0])
    def T_dim(self, A : Matrix, prec = 0): return Vector(
            numpy.zeros((2,A.shape()[0])).astype(
                numpy.float32 if prec==0 else A.matrix.dtype
            )
    )

    ## An exercise in building a tiling for norm: the weight tiling 
    def comp_wts(self,
                 A    :  Matrix, # original matrix
                 rows : int =4,   # cores per column
                 cols : int =4,   # cores per row
                 L2   : int = 128*1024, # size in elements L2 IFM
                 L1   : int = 4*1024,    # size in elements Ping/Bank
                 V    : list = [],
                 ) -> list:

        Repetition = {'L3' : -1, 'L2' : -1, 'L1' : -1 }
        T = Tiling(A)

        ## Tiling by column if the tiling takes all the columns, thus
        ## we want to highlight that the single block is by row, so we
        ## have a marker that the comptution as an asymmetric feel and
        ## the r -> c switch is caught during the comptuation.

        
        def qc3(A) : return Qc_(A,V[0])   #self.
        def qc3_(A) : return Cr(A)   #self.
        def qc2(A) : return Qc_(A,V[1]) #self.
        def qc2_(A) : return Qr(A,1) #self.
        def qc1(A) : return Qc_(A,V[2]) #self.
        def qc1_(A) : return Qr(A,1) #self.
        import pdb; pdb.set_trace()
        print(V,A.shape())
        Q = [qc3 if A.shape()[1]>V[0] else qc3_,
             qc2 if A.shape()[1]>V[1] else qc2_,
             qc1 if A.shape()[1]>V[2] else qc1_]
        print(Q)
        T.rec_traversal(Q)
       
        return T

    ## (Kernel computation)
    def pass_one(self,A      : Matrix) -> Matrix:
        A.color +=1
        #pdb.set_trace()
        SUM  = self.P(A)
        SUMQ = self.P(self.G(A,square))
        CSUM= numpy.zeros((2,A.shape()[0]))
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
        M = A.shape()[0]

        ## one pass computation of the mu and sigma
        mu  = CSUM.value()[0,:]/N
        mu2 = (CSUM.value()[0,:]/numpy.sqrt(N))**2

        try:
            eps = numpy.finfo(A.matrix.dtype).resolution
            s = 1/numpy.sqrt(eps +(CSUM.value()[1,:] - mu2)/N)
        except:
            pdb.set_trace()
        mu = Vector(-mu*s)
        s = Vector(s)
        #pdb.set_trace()
        B = self.N(A,s)
        B = Nplus(B,mu)
        B = self.N(B,Vector(GB.value()[0,:]), row=False)
        B = Nplus(B, Vector(GB.value()[1,:]) , row=False)
        return B

    ## matrix computation of the layer norm
    def comp(self,
             A: Matrix,
             GB : Matrix):
        T = self.pass_one(A)
        return self.pass_two(A,GB,T)

    ## tiling computation of the layer norm
    def comp_uni(self,
               A  : Matrix, # matrix MxN
               GB : Matrix # matrix 2xN
            ):

        self.A = A
        ## compute tiling for IFM
        Pace = self.comp_IFM(A)
        print("---------------------------------------\nPartition input")
        print(Pace)
                
        self.GB = GB
        ## compute tiling for the gamma and beta
        V = Pace.max_col()
        #pdb.set_trace()
        Wts  = self.comp_wts(GB, V=V )
        print("---------------------------------------\nPartition weights")
        print(Wts)

        self.comp_visit(Pace,Wts)
        print("---------------------------------------\n With Colors")
        print(Pace)
        print(Wts)
        pdb.set_trace()
        print(Pace.full_traversal())
        print(Wts.full_traversal())

        
    ###
    ##  You can see how the recursive computation follows the same
    ##  comptutation of the Norm but now there is a "wts".  The tiling
    ##  for IFM and WTS is different but the "levels" are the same and
    ##  the tiling is done so that is it consistent
    ##
    ###
    def comp_visit(self,
                   Ti : Tiling,
                   Wt : Tiling,
                   level : int = 0,
                   T : Matrix = None)  :
        DDRs = Ti.get_partition()
        WDDR = Wt.get_partition()

        ty = DDRs[-1]
        if self.parallel(ty[0]) : # ty.find('r')==0  :
            ## paralle computation 
            for j in range(len(DDRs)-1):
                di = DDRs[j]
                dw = WDDR[j] if len(WDDR) == len(DDRs) else WDDR[0]
                if not type(di) is Matrix: self.comp_visit(di,dw,level=level,T = None)
                else:                      self.comp(di,dw)

        else: #elif ty == 'c' :
            ## boom recursive projections 
            if T is None and level ==0 :

                di = DDRs[0]
                ## this si a core local space use for accumulation
                #T = Vector(numpy.zeros((2,Ti.get_buffer().shape()[0])))
                T = self.T_dim(Ti.get_buffer())
                #T = Vector(numpy.zeros(self.t_dim(Ti.get_buffer())))
                for j in range(len(DDRs)-1):
                    di = DDRs[j];  dw = WDDR[j] if len(WDDR) == len(DDRs) else WDDR[0]
                                     
                    if type(di) is Matrix:              self.R(T,self.pass_one(di))
                    else: self.R(T,self.comp_visit(di,dw,level = level+1,T = None))

                # T is computed
                # Pass two
                for j in range(len(DDRs)-1):
                    di = DDRs[j];  dw = WDDR[j] if len(WDDR) == len(DDRs) else WDDR[0]
                           
                    if type(di) is Matrix: self.pass_two(di,dw,T)
                    else:  self.comp_visit(di,dw,level =level+1,T = T)

            elif T is None and level>0 :

                #print(self.A.value())
                di = DDRs[0]
                dw = WDDR[0]
                #T = Vector(numpy.zeros((2,Ti.get_buffer().shape()[0])))
                T = self.T_dim(Ti.get_buffer()) #T = Vector(numpy.zeros(self.t_dim(Ti.get_buffer())))
                for j in range(len(DDRs)-1):
                    di = DDRs[j];   dw = WDDR[j] if len(WDDR) == len(DDRs) else WDDR[0]

                    if type(di) is Matrix:  self.R(T,self.pass_one(di))
                    else: self.R(T,self.comp_visit(di,dw,level = level+1,T = None ))
                    
                        
                return T
            else:
                for j in range(len(DDRs)-1):
                    di = DDRs[j]
                    dw = WDDR[j] if len(WDDR) == len(DDRs) else WDDR[0]
                    if type(di) is Matrix: self.pass_two(di,dw,T)
                    else:  self.comp_visit(di,dw,level=level,T=T)
                    
                        
            #pdb.set_trace()
        ##print(DDRs[0][0])
        return None



    
if __name__ == "__main__":


    #import pdb
    shape =  (512,2048)
    dt = numpy.float16

    if False:
        ## Euclidean Norm !
        
        A = numpy.random.rand(*shape).astype(dt)
        A1 = A*1.0

        ## computation as single matrix 
        N = Norm()
        N.comp(Matrix(A))
        
        ## computation using numpy
        M = 1/numpy.sqrt(numpy.finfo(A.dtype).resolution + numpy.sum(A1**2, axis=-1)/A1.shape[1])
        R1 = A1*M[:,None]
        print("MAX ERROR NORM", numpy.max(numpy.fabs(R1-A)))
        pdb.set_trace()

        ## computation using tiling 
        A2 = A1 *1.0 + 0.0
        N.comp_uni(Matrix(A2))
        print("MAX ERROR NORM", numpy.max(numpy.fabs(R1-A2)))
        pdb.set_trace()

        
    if True:
        ## Layer Norm
        
        A = numpy.random.rand(*shape).astype(dt)
        A1 = A*1.0 + 0.0
        if True:
            Gamma = numpy.random.rand(shape[1]).astype(dt)
            Beta  = numpy.random.rand(shape[1]).astype(dt)
        else:
            Gamma = numpy.ones(shape[1]).astype(dt)
            Beta  = numpy.zeros(shape[1]).astype(dt)
            
        GB = numpy.random.rand(2,A.shape[1]).astype(dt)
        GB[0,:] = Gamma
        GB[1,:] = Beta


        ## computation using numpy 
        mu  = numpy.average(A1,axis=-1)
        var = numpy.var(A1,axis=-1)
        s = 1/numpy.sqrt(numpy.finfo(A.dtype).resolution +var)
        
        mu = mu*s
        R1 = (A1*s[:,None]-mu[:, None])*Gamma + Beta

        LN = LayerNorm()
        AA = Matrix(copy.deepcopy(A))
        GGB = Matrix(GB)

        ## computation as matrix
        R = LN.comp(AA,GGB)
        print("MAX ERROR LN", numpy.max(numpy.fabs(R1-R.value())))

        ## computation using tiling
        BB = Matrix(copy.deepcopy(A))
        LN.comp_uni(BB, GGB)
        print("MAX ERROR LN ",numpy.max(numpy.fabs(R1-BB.value())))
        pdb.set_trace()
        
