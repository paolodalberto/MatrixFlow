import numpy 
import math
import os
import pdb
import copy

from matrix import Vector, Matrix, Tiling
from splitting import Qr, Qr_, Qc, Qc_, Qrc_,Cr, fit, fit_qrc

def square(x) : return x*x
## Example of projective sums or reductions (this is associative)
def Pplus(A : Matrix, axis = -1) -> Vector:
    return Vector(numpy.sum(A.value(),axis=axis))

def Pmax(A : Matrix, axis = -1) -> Vector:
    return Vector(numpy.max(A.value(),axis=axis))

## example of product norm 
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


    


class Norm :
    def __init__(
            self,
            P      = Pplus,
            N      = Nmul,
            G      = Gg):
        self.P     = P
        self.N     = N
        self.G     = G
        self.A     = None
        

    def pass_one(self, A  : Matrix):
        return self.P(A)
    def pass_two(self, A:Matrix,T:Matrix):
        #pdb.set_trace()
        L = self.A.shape()[1] if self.A else A.shape()[1]
        TT = Vector(numpy.sqrt(T.vector/L))
        self.N(A, TT)
        return A

    def comp(self,A:Matrix):
        
        #pdb.set_trace()
        T = self.pass_one(A)

        return self.pass_two(A,T)

    def comp_uni(self,A:Matrix):
        self.A = A
        DDR = self.comp_IFM(self.A)
        print(DDR)
        self.comp_visit(DDR)
        
    def comp_visit(self,
                   Ti : Tiling,
                   level : int = 0,
                   T :Vector  = None)  :
        #pdb.set_trace()

        ##  [ d, ... r/c]
        ##  d -> [d0] [list] as above
        ##  or d = matrix 

        #print(type( DDRs[0]))
        DDRs = Ti.get_partition()
        ty = DDRs[-1]
        if ty.find('r')==0  :
            
            for j in range(len(DDRs)-1):
                d = DDRs[j]
                if type(d) is Matrix: self.comp(d)
                else:    self.comp_visit(d)
        elif ty == 'c' :
            
            if T is None and level ==0 :
                #print(self.A.value())
                d = DDRs[0]
                T = Vector(numpy.zeros(Ti.get_buffer().shape()[0]))
                for j in range(len(DDRs)-1):
                    d = DDRs[j]
                    if type(d) is Matrix:    T += self.pass_one(d)
                    else: T += self.comp_visit(d, level = level+1)
                        
                for j in range(len(DDRs)-1):
                    d = DDRs[j]
                    if type(d) is Matrix:
                        self.pass_two(d,T)
                    else: self.comp_visit(d,T = T)

            elif T is None and level>0 :
                j = 0
                #print(self.A.value())
                d = DDRs[0]
                T = Vector(numpy.zeros(Ti.get_buffer().shape()[0]))
                for j in range(len(DDRs)-1):
                    d = DDRs[j]
                    if type(d) is Matrix:
                        T += self.pass_one(d)
                    else:T += self.comp_visit(d, level = level+1)
                return T
            else:
                for j in range(len(DDRs)-1):
                    d = DDRs[j]
                    if type(d) is Matrix:
                        self.pass_two(d,T)
                    else: self.comp_visit(d,T=T)
                        
            #pdb.set_trace()
        ##print(DDRs[0][0])
        return None
    
            
        
        
    
    
    ## An exercise in building a tiling for norm
    ##
    ## A traversal is composed by a spatial division then by a
    ## temporal division (traversal of the spatial by a tile and order
    def comp_IFM(self,
                A    :  Matrix, # original matrix
                rows : int =4,   # cores per column
                cols : int =4,   # cores per row
                L2   : int = 128*1024, # size in elements L2 IFM
                L1   : int = 4*1024,    # size in elements Ping/Bank
                H    : int = 16,
                V    : int = 32
                ) -> list:

        Repetition = {'L3' : -1, 'L2' : -1, 'L1' : -1 }

        T = Tiling(A)

        ## we split A into 4 parts spatial
        ## L3 -> L2  spatial 4 By Columns

        def qr(A) : return Qr(A, cols)
        T.traversal(qr)
        print(T)
        #pdb.set_trace()

    
        ## we try to determine the largest tile in L2 for which we
        ## split by rows and we use ping pong 
        Ts = fit(T,
                 L2,  ## L2 size (tile for L2)
                 Qr_, ## split by row
                 2,   ## double buffering 
                 rows ## minimum granularity
                 )
        print(T)
        if Ts:
            ## we ca traverse, we do need repetition at L3, and thus
            ## repetition in L2 at a minimum
            #print("L3 repetition =1 ")
            Repetition['L3'] =1
            #print(T.visit())
            ## L3 -> L2 temporal for every spatial (we literally build
            ## each traversal but it is not really needed because the
            ## traversal has to be the same)

            DDRs = Ts.getlist()
            ## L2 -> L1 if there is core splitting there will be
            ## spatial and then temporal,
            for s in range(len(DDRs)-1):
                
                ## parallel split 
                Q = fit( DDRs[s],
                         L1,  ## L1 size 
                         Qr_, ## split by row 
                         1,   ## ping pong
                         rows ## to feed each core 
                        )
                print(Q)
                if Q is None:

                    ## One row no fit in L1 but we do reduction in L1
                    Q = fit( DDRs[s],
                             L1,  ## L1 size 
                             Qc_, ## split by column
                             1,   ## this is ping 
                             V*2   ## 32*2 
                            )  
                    if Q is None:
                        #pdb.set_trace()
                        ## we cannot have all rows at once
                        Q = fit_qrc( DDRs[s],
                                     L1,
                                     Qrc_,
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
                        print("L2 repetition =2 ")
                else:
                    ## One row will fit in L1, thus there is no
                    ## repetition in L2 (only in L1)
                    Repetition['L2'] =1
                    Repetition['L1'] =2
                    #print("L2 repetition =1 ")
                    #print("L1 repetition =2 ")
                        

                DDRs[s] = Q
        else:

            pdb.set_trace()
            ## we need to change strategy and we have to read L3 twice
            ## the order is important because 
            print("L3 repetition =2 ")
            Repetition['L3'] =2

            ## we read the whole matrix into columns 
            Ts = fit(T, L2, Qc_, 2,1)
            DDRs = Ts.getlist()
            ## L2 -> L1 if there is core splitting there will be spatial
            ## and then temporal, 
            for s in range(len(DDRs)-1):
                
                Q = fit( DDRs[s], L1, Qc_, 1, 1)
                if Q is None:
                    Q = fit_qrc( DDRs[s], L1, Qrc_, 1, rows,1)
                    if Q is None:
                        return None

                DDRs[s] = Q

        #pdb.set_trace()
        #print(DDRs)    
        #print(A.shape())
        print(Repetition)
        #pdb.set_trace()
        
       
        return T

    

class LayerNorm(Norm):
    def __init__(
            self,
            P      = Pplus,
            N      = Nmul,
            G      = Gg):

        Norm.__init__(self,P,N,G)
        self.GB = None
        
    ## An exercise in building a tiling for norm
    ##
    ## A traversal is composed by a spatial division then by a
    ## temporal division (traversal of the spatial by a tile and order
    def comp_wts(self,
                 A    :  Matrix, # original matrix
                 rows : int =4,   # cores per column
                 cols : int =4,   # cores per row
                 L2   : int = 128*1024, # size in elements L2 IFM
                 L1   : int = 4*1024,    # size in elements Ping/Bank
                 H    : list = [],
                 V    : int = 32
                 ) -> list:

        Repetition = {'L3' : -1, 'L2' : -1, 'L1' : -1 }
        
        
        T = Tiling(A)
        print(V)
        def qc3(A) : return Qc_(A,V[0])
        def qc2(A) : return Qc_(A,V[1])
        def qc1(A) : return Qc_(A,V[2])

        ## we split A into 4 copies
        ## we tile in L3
        ## I know is missing L2
        ## We tile in L1 
        Q = [qc3,qc2,qc1]
        T.rec_traversal(Q)
       
        return T
        
    def pass_one(self,A      : Matrix) -> Matrix:
        A.color +=1
        #pdb.set_trace()
        SUM  = self.P(A)
        SUMQ = self.P(self.G(A,square))
        GB = numpy.zeros((2,A.shape()[0]))
        GB[0,:] = SUM.value()
        GB[1,:] = SUMQ.value()
        return Matrix(GB)
    

    def pass_two(self,
                 A      : Matrix,
                 GB     : Matrix,
                 CSUM    :Matrix
                 ):
        #pdb.set_trace()
        A.color +=1
        GB.color +=1
        
        # sequential and sync with pass one
        N = max(A.shape()[1],self.A.shape()[1] if self.A else 0)
        M = A.shape()[0]
        
        mu  = CSUM.value()[0,:]/N
        mu2 = CSUM.value()[0,:]**2/N
        try:
            s = 1/numpy.sqrt((CSUM.value()[1,:] - mu2)/N)
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
    
    def comp(self,
             A: Matrix,
             GB : Matrix):
        T = self.pass_one(A)
        return self.pass_two(A,GB,T)
    
    def comp_uni(self,
               A  : Matrix, # matrix MxN
               GB : Matrix # matrix 2xN
            ):
        #pdb.set_trace()
        self.A = A
        Pace = self.comp_IFM(A)
        print(Pace)
        
        self.GB = GB
        V = Pace.max_col()
        pdb.set_trace()
        Wts  = self.comp_wts(GB, V=V )
        print(Wts)
        pdb.set_trace()
        self.comp_visit(Pace,Wts)

    def comp_visit(self,
                   Ti : Tiling,
                   Wt : Tiling,
                   level : int = 0,
                   T : Matrix = None)  :
        #pdb.set_trace()

        ##  [ d, ... r/c]
        ##  d -> [d0] [list] as above
        ##  or d = matrix 

        #print("---------------------------------------------------------\n",
        #      level, Ti, Wt)
        DDRs = Ti.get_partition()
        WDDR = Wt.get_partition()
        #print(level, len(WDDR),len(DDRs))
        ty = DDRs[-1]
        if ty.find('r')==0  :
            
            for j in range(len(DDRs)-1):
                #print("### J ####", j)
                #print(level, len(WDDR),len(DDRs))
                #print("---------------------------------------------------------\n",
                #      level, Ti, Wt)
                
                
                di = DDRs[j]
                ## spatial ?
                dw = WDDR[j] if len(WDDR) == len(DDRs) else WDDR[0]
                if not type(di) is Matrix:
                    self.comp_visit(di,dw,
                                    level=level,
                                    T = None)
                else:  self.comp(d)

        elif ty == 'c' :
            #pdb.set_trace()
            
            if T is None and level ==0 :

                di = DDRs[0]
                ## this si a core local space use for accumulation
                T = Vector(numpy.zeros((2,Ti.get_buffer().shape()[0])))

                for j in range(len(DDRs)-1):
                    di = DDRs[j];  dw = WDDR[j]
                                     
                    if type(di) is Matrix: T += self.pass_one(di)
                    else: T += self.comp_visit(di,dw,level = level+1,T = None)

                # T is computed
                # Pass two
                #pdb.set_trace()
                for j in range(len(DDRs)-1):
                    di = DDRs[j];  dw = WDDR[j]
                           
                    if type(di) is Matrix: self.pass_two(di,dw,T)
                    else:  self.comp_visit(di,dw,level =level+1,T = T)

            elif T is None and level>0 :

                #print(self.A.value())
                di = DDRs[0]
                dw = WDDR[0]
                T = Vector(numpy.zeros((2,Ti.get_buffer().shape()[0])))

                for j in range(len(DDRs)-1):
                    di = DDRs[j];   dw = WDDR[j]

                    if type(di) is Matrix:  T += self.pass_one(di)
                    else: T += self.comp_visit(di,dw,level = level+1,T = None )
                    
                        
                return T
            else:
                for j in range(len(DDRs)-1):
                    di = DDRs[j]
                    dw = WDDR[j]
                    if type(di) is Matrix:
                        self.pass_two(di,dw,T)
                    else: self.comp_visit(di,dw,level=level,T=T)
                    
                        
            #pdb.set_trace()
        ##print(DDRs[0][0])
        return None
                
if __name__ == "__main__":


    #import pdb
    shape =  (512,4096*8)


    if False:
        A = numpy.random.rand(*shape)
        A1 = A*1.0
        
        N = Norm()
        N.comp(Matrix(A))
        
        
        M = numpy.sqrt(numpy.sum(A1, axis=-1)/A1.shape[1])
        
        
        R1 = A1*M[:,None]
        #print(R1)
        print("MAX ERROR A", numpy.max(numpy.fabs(R1-A)))
        #pdb.set_trace()

        A2 = A1 *1.0 + 0.0
        pdb.set_trace()
        N.comp_uni(Matrix(A2))
    
    
        print("MAX ERROR B", numpy.max(numpy.fabs(R1-A2)))
        pdb.set_trace()

        
    if True:
        A = numpy.random.rand(*shape)
        A1 = A*1.0 + 0.0
        Gamma = numpy.ones(shape[1])
        Beta = numpy.zeros(shape[1])
        
        mu  = numpy.average(A1,axis=-1)
        var = numpy.var(A1,axis=-1)
        print(mu)
        print(var)
        s = 1/numpy.sqrt(var)
        
        mu = mu*s
        R1 = (A1*s[:,None]-mu[:, None])*Gamma + Beta

        GB = numpy.zeros((2,A.shape[1]))
        GB[0,:] = Gamma
        GB[1,:] = Beta

        pdb.set_trace()
        LN = LayerNorm()
        AA = Matrix(copy.deepcopy(A))
        GGB = Matrix(GB)
        R = LN.comp(AA,GGB)
        pdb.set_trace()
        

        print("MAX ERROR", numpy.max(numpy.fabs(R1-R.value())))
        pdb.set_trace()

        
        BB = Matrix(copy.deepcopy(A))
        LN.comp_uni(BB, Matrix(GB))
        print(numpy.max(numpy.fabs(R1-BB.value())))
        pdb.set_trace()
