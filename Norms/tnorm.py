import numpy 
import math
import os
import pdb
import copy

from matrix import Vector, Matrix, Tiling
from splitting import Qr, Qr_, Qc, Qc_, Qrc_,Cr, fit, fit_qrc, Identity, copy_tiling


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
        if True : return 0
        print("---------------------------------------\n With Colors")
        print(DDR)
        print(
            DDR.full_traversal(
                parallel='r' if self.parallel('r') else 'c'
            )
        )
        #pdb.set_trace()

        O = Matrix(A.value()*1.0)
        T = Tiling(O)
        T = copy_tiling(T, DDR)
        #pdb.set_trace()
        T.core_spatial(self.Qr)
        print("---------------------------------------\n Output")
        print(T)
        print(
            T.full_traversal(
                parallel='r' if self.parallel('r') else 'c'
            )
        )
        

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
    def comp_visit(self,Ti : Tiling,level : int = 0, T :Vector  = None) -> Vector  :

        DDRs = Ti.get_partition() 
        ty = DDRs[-1]
        
        if not self.parallel(ty[0]) and T is None : 
            # PARTIAL RESULTS
            T = self.T_dim(Ti.get_buffer())
            # PASS ONE 
            for j in range(len(DDRs)-1):
                di = DDRs[j];  

                ## The AIE will compute literally this in the core, so
                ## the update does not requires a second operand to be
                ## correct
                if type(di) is Matrix:  self.R(T,self.pass_one(di))
                else:                   self.comp_visit(di,level=1,T=T)

            # PASS TWO
            for j in range(len(DDRs)-1):
                di = DDRs[j];  
                
                if type(di) is Matrix: self.pass_two(di,T)
                else:  self.comp_visit(di,level =2,T = T)

            #delete T
            del T

        elif not self.parallel(ty[0]) and T :
            # reduction but we have already temporary
            ti = T
            for j in range(len(DDRs)-1):
                di = DDRs[j];  

                ## in the else in the AIE we do not need a second operand
                if type(di) is Matrix:
                    if level ==1 : self.R(ti, self.pass_one(di))  ## we are still computing T
                    if level ==2 : self.pass_two(di,ti)        ## we are distributing    T

                else: self.comp_visit(di,level=level,T = T)

            if level == 1:  return T  ## we must return the partial 
            else:           return None

        elif self.parallel(ty[0]) and T :
            ## we split the T by 2xR by row R
            Ts = self.Qc(T,len(DDRs)-1)
            for j in range(len(DDRs)-1):
                ti = Ts[j]; di = DDRs[j];

                if type(di) is Matrix:
                    if level ==1 : ti.set_value(self.pass_one(di)) ## we compute each part
                    if level ==2 : self.pass_two(di,ti)         ## distribute each part
                else: self.comp_visit(di,level=level,T = ti)    ## paralell do not reduce, they copy
            return T 

        elif self.parallel(ty[0]) and T is None:
            ## parallel and no temporary business as usual
            for j in range(len(DDRs)-1):
                di = DDRs[j]; 

                if type(di) is Matrix: self.comp(di)
                else:                  self.comp_visit(di,level=level,T =None)
                
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

        T = Tiling(A)

        ## we split A into 4 parts spatial
        ## L3 -> L2  spatial 4 By Columns

        def qr(A) : return self.Qr(A, cols)
        T.traversal(qr)
        T.properties['temporal'] = False
        
        ## we try to determine the largest tile in L2 for which we
        ## split by rows and we use ping pong 
        Ts = fit(T,
                 L2,  ## L2 size (tile for L2)
                 self.Qr_, ## split by row
                 2,   ## double buffering 
                 rows ## minimum granularity
                 )
        
        if Ts:
            DDRs = Ts.getlist()
            for s in range(len(DDRs)-1):

                ## parallel split // well broad cast ... 
                Q = fit( DDRs[s], L1,self.Qr_,1,rows)
                if not Q is None: DDRs[s] = Q; continue

                ## One row no fit in L1 but we do reduction in L1
                Q = fit( DDRs[s],L1,self.Qc_,1,V*2)
                if not Q is None:  DDRs[s] = Q; continue
                        
                ## we cannot have all rows at once
                Q = fit_qrc( DDRs[s],L1,self.Qrc_,1,rows,V*2)
                if Q is None: return None

                DDRs[s] = Q
        else:
            ## we read the whole matrix into columns 
            Ts = fit(T, L2, self.Qc_, 2,1)
            DDRs = Ts.getlist()
            ## L2 -> L1 if there is core splitting there will be spatial
            ## and then temporal, 
            for s in range(len(DDRs)-1):
                
                Q = fit( DDRs[s], L1, self.Qc_, 1, 2)
                if not Q is None: DDRs[s] = Q; continue 

                print(" We should not be here! ########")
                pdb.set_trace()
                Q = fit_qrc( DDRs[s], L1, self.Qrc_, 1, rows,2)
                if Q is None:return None
                    
                DDRs[s] = Q

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

        T = Tiling(A)

        ## there is no spatial split so we make a identity partition
        T.traversal(Identity)

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

                #spatial split into 4 column memtiles 
                def qr(A) : return self.Qr(A, cols)
                Q.traversal(qr)
                Q.properties['temporal'] = False

                ## temporal row ?
                Ts = fit(Q, L1, self.Qr_, 1, rows)
                if not Ts is None: TP[i] = Q; continue

                # temporal column ? 
                Ts = fit(Q,L1,self.Qc_,1,2*V)
                if not Ts is None: TP[i] = Q;continue

                ## temporal rw x col
                Ts = fit(Q,L1,self.Qrc_,1,rows, 2*V)

                if Ts is None:
                    ## we should not be here 
                    pdb.set_trace()
                    return None

                TP[i] = Q
                
            
        else:
            
            ## we need to change strategy and we have to read L3 twice
            ## the order is important because
            
            ## we read the whole matrix into columns 
            Ts = fit(T, L2, self.Qc_,2,8)

            if Ts is None: return None
            T = T.partition[0]
            TP = T.partition
            L = len(TP)-1
            for i in range(L):
                ## for each temporal tile, we do tiling
                Q = Tiling(TP[i])

                #spatial split
                def qr(A) : return self.Qr(A, cols)
                Q.traversal(qr)
                Q.properties['temporal'] = False

                # time split 
                Ts = fit( Q, L1, self.Qc_, 1, 8)
                if Ts is None: return None

                ## this should not work at all 
                #  Ts = fit_qrc( Q, L1, self.Qrc_, 1, rows,2)
                #  if Ts is None: return None
                TP[i] = Q;

        #print(T)
        #pdb.set_trace()
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

        self.comp_wts = self.comp_wts_unified

        
    ## the partial results will need temporary tensors these are also
    ## the accumulators and we may use different precisions
    def t_dim(self, A : Matrix) : return (2,A.shape()[0])
    def T_dim(self, A : Matrix, prec = 0): return Matrix(
            numpy.zeros((2,A.shape()[0])).astype(
                numpy.float32 if prec==0 else A.matrix.dtype
            )
    )

    ## An exercise in building a tiling for norm: the weight tiling 
    def comp_wts_(self,
                 A    :  Matrix, # original matrix
                 rows : int =4,   # cores per column
                 cols : int =4,   # cores per row
                 L2   : int = 128*1024, # size in elements L2 IFM
                 L1   : int = 4*1024,    # size in elements Ping/Bank
                 V    : list = [],
                 ) -> list:

        
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
        #import pdb; pdb.set_trace()
        #print(V,A.shape())
        Q = [qc3 if A.shape()[1]>V[0] else qc3_,
             qc2 if A.shape()[1]>V[1] else qc2_,
             qc1 if A.shape()[1]>V[2] else qc1_]
        print(Q)
        T.rec_traversal(Q)
       
        return T


    ## An exercise in building a tiling for norm: the weight tiling 
    def comp_ofm_unified(
            self,
            T   :  Tiling, Matrix, # original matrix
            rows : int =4,   # cores per column
            cols : int =4,   # cores per row
            L2   : int = 128*1024, # size in elements L2 IFM
            L1   : int = 4*1024,   # size in elements Ping/Bank
            V    : list = [],
    ) -> list:

        
        T = Tiling(A)

        ## Tiling by column if the tiling takes all the columns, thus
        ## we want to highlight that the single block is by row, so we
        ## have a marker that the comptution as an asymmetric feel and
        ## the r -> c switch is caught during the comptuation.

        
        def qc3(A) : return Qc_(A,V[0])   #self.
        def qc3_(A) : return Qr(A,1)   #self.
        def qc2(A) : return Qc_(A,V[1]) #self.
        def qc2_(A) : return Cr(A) #self.
        def qc1(A) : return Qc_(A,V[2]) #self.
        def qc1_(A) : return Qr(A,1) #self.
        #import pdb; pdb.set_trace()
        #print(V,A.shape())
        Q = [qc3 if A.shape()[1]>V[0] else qc3_,
             qc2 if A.shape()[1]>V[1] else qc2_,
             qc1 if A.shape()[1]>V[2] else qc1_]
        print(Q)
        T.rec_traversal(Q,['t','s','t'])
       
        return T


    ## An exercise in building a tiling for norm: the weight tiling 
    def comp_wts_unified(self,
                 A    :  Matrix, # original matrix
                 rows : int =4,   # cores per column
                 cols : int =4,   # cores per row
                 L2   : int = 128*1024, # size in elements L2 IFM
                 L1   : int = 4*1024,   # size in elements Ping/Bank
                 V    : list = [],
                 ) -> list:


        T = Tiling(A)

        ## Tiling by column if the tiling takes all the columns, thus
        ## we want to highlight that the single block is by row, so we
        ## have a marker that the comptution as an asymmetric feel and
        ## the r -> c switch is caught during the comptuation.

        
        def qc3(A) : return Qc_(A,V[0])   #self.
        def qc3_(A) : return Qr(A,1)   #self.
        def qc2(A) : return Qc_(A,V[1]) #self.
        def qc2_(A) : return Cr(A) #self.
        def qc1(A) : return Qc_(A,V[2]) #self.
        def qc1_(A) : return Qr(A,1) #self.
        #import pdb; pdb.set_trace()
        #print(V,A.shape())
        Q = [qc3 if A.shape()[1]>V[0] else qc3_,
             qc2 if A.shape()[1]>V[1] else qc2_,
             qc1 if A.shape()[1]>V[2] else qc1_]
        print(Q)
        T.rec_traversal(Q,['t','s','t'])
       
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

        pdb.set_trace()
        self.comp_visit(Pace,Wts)

        if True: return 0

        
        print("---------------------------------------\n With Colors")
        print(Pace)
        print(Wts)
        #pdb.set_trace()
        print(Pace.full_traversal(parallel='r' if self.parallel('r') else 'c'))
        print(Wts.full_traversal( parallel='r' if self.parallel('r') else 'c'))
        
        print("---------------------------------------\n With Colors and Core")
        O = Matrix(A.value()*1.0)
        T = Tiling(O)
        T = copy_tiling(T,Pace)
        #pdb.set_trace()
        T.core_spatial(self.Qr)
        print(
            T.full_traversal(
                parallel='r' if self.parallel('r') else 'c'
            )
        )

 

    ###
    ##  You can see how the recursive computation follows the same
    ##  comptutation of the Norm but now there is a "wts".  The tiling
    ##  for IFM and WTS is different but the "levels" are the same and
    ##  the tiling is done so that is it consistent
    ##
    ###
    def comp_visit(self,Ti : Tiling,Wt : Tiling, level : int = 0,T : Matrix = None) :  
        DDRs = Ti.get_partition();  WDDR = Wt.get_partition()

        # pdb.set_trace()
        ty = DDRs[-1]
        
        if not self.parallel(ty[0]) and T is None : 
            # PARTIAL RESULTS
            T = self.T_dim(Ti.get_buffer())
            # PASS ONE 
            for j in range(len(DDRs)-1):
                di = DDRs[j];  dw = WDDR[j] if len(WDDR) == len(DDRs) else WDDR[0]

                ## The AIE will compute literally this in the core, so
                ## the update does not requires a second operand to be
                ## correct
                if type(di) is Matrix:  self.R(T,self.pass_one(di))
                else:                   self.comp_visit(di,dw,level=1,T=T)

            # PASS TWO
            for j in range(len(DDRs)-1):
                di = DDRs[j];  dw = WDDR[j] if len(WDDR) == len(DDRs) else WDDR[0]
                
                if type(di) is Matrix: self.pass_two(di,dw,T)
                else:  self.comp_visit(di,dw,level =2,T = T)

            #delete T
            del T

        elif not self.parallel(ty[0]) and T :
            # reduction but we have already temporary
            ti = T
            for j in range(len(DDRs)-1):
                di = DDRs[j];  dw = WDDR[j] if len(WDDR) == len(DDRs) else WDDR[0]

                ## in the else in the AIE we do not need a second operand
                if type(di) is Matrix:
                    if level ==1 : self.R(ti, self.pass_one(di))  ## we are still computing T
                    if level ==2 : self.pass_two(di,dw,ti)        ## we are distributing    T

                else: T,self.comp_visit(di,dw,level=level,T = T)

            if level == 1:  return T  ## we must return the partial 
            else:           return None

        elif self.parallel(ty[0]) and T :
            ## we split the T by 2xR by row R
            Ts = self.Qc(T,len(DDRs)-1)
            for j in range(len(DDRs)-1):
                ti = Ts[j]; di = DDRs[j]; dw = WDDR[j] if len(WDDR) == len(DDRs) else WDDR[0]

                if type(di) is Matrix:
                    if level ==1 : ti.set_value(self.pass_one(di)) ## we compute each part
                    if level ==2 : self.pass_two(di,dw,ti)         ## distribute each part
                else: self.comp_visit(di,dw,level=level,T = ti)    ## paralell do not reduce, they copy
            return T 

        elif self.parallel(ty[0]) and T is None:
            ## parallel and no temporary business as usual
            for j in range(len(DDRs)-1):
                di = DDRs[j]; dw = WDDR[j] if len(WDDR) == len(DDRs) else WDDR[0]

                if type(di) is Matrix: self.comp(di,dw)
                else:                  self.comp_visit(di,dw,level=level,T =None)
                
        # pdb.set_trace()
        return None



    
if __name__ == "__main__":


    #import pdb
    
    shape =  (1024,4096*4)
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
        A1 = A1.astype(numpy.float32)
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
        pdb.set_trace()
        
        ## computation using tiling
        BB = Matrix(copy.deepcopy(A))
        LN.comp_uni(BB, GGB)
        print("MAX ERROR LN ",numpy.max(numpy.fabs(R1-BB.value())))
        pdb.set_trace()
        
