####
##
##  Assume we have a time series of execution time or efficiency or
##  cost in general. We believe we are smarty pants and we have built
##  a model of the function that take a set of parameters and an input.
##
##  Here the cost function is about matrix multiplication: the input
##  is the problem shapes and the parameter describe the complexity of
##  the computation with respect to a three level memory hierarchy. 
##
##  The goal here is to show the basic mechanics to find out the
##  parameters in the model that will allows us to fir better the
##  model for estimates and forecasts using scipy only, old fashion.
###

import numpy
import math
import scipy 


##
# Architecture in a few lines 
##

Byte = 2**3     # Bytes 
G = (2**10)**3  # Giga
GB = G*Byte     # GigaBytes

GHz = 10**9     # Giga Hertz, 
Clock = 2*GHz   # the clock of a core, one tick,  many operations

ROWS = 4        # Rows of cores 
COLS = 4        # Columns of cores 
CORES = ROWS * COLS  

## Small cost functions x is a problem shape and we specify the
## operand in bits. 

def cycles(x):
    m,n,k,atype,btype,ctype = x
    B = 256
    if   ctype == 8 :         B = 256   # int8 = 256 operations/cycle
    elif ctype == 16:         B = 64    # int16/fp16 = 64 
    elif ctype == 32 :        B = 16    # float32  
    
    return math.ceil(m*n*k/B)    #  cycles to compute m*n*k operation

##  execution time for a problem x 
def core_gemm(x):
    return cycle(x)/Clock
    
## Size of the operands A_ik and B_kj in bits
def comm_AB(x):
    m,n,k,atype,btype,ctype = x
    return m*k*atype + n*k*btype

## Size of the operands C_ij in bits
def comm_C(x):
    m,n,k,atype,btype,ctype = x
    return m*n*ctype

##  Memory Hierarchy L3, L2, L1
##  L3 = DDR where every thing begins and ends 
##  L2 is an intermediary that communicate to columns
##  L1 is the memory per core.


ddr_bandwidth = 2 * 4 * GB # two channels of 4GBs each

## there are 1 channel per column and one channel per row and each
## channel has bandwidth

MC_bandwidth_channel =   4 * GB  


###
## Time estimate: 
##     p = set of parameter describing the features of the architecture
##     x = set of parameters describing the algorithms and problem size, see inside the code 
##
##     C = A*B
##     L3 -> L2 
##     C_i,j = sum_k A_i,k B_k,j 
##          C_ii,jj  = sum_k  A_ii,kk B_kk,jj  
###
def f_model(
        p : numpy.array = numpy.array([
            ddr_bandwidth,
            MC_bandwidth_channel,   
            Clock] ),
        x  : list = None,
):

    if x is None or (type(x) is list and len(x)!=3): return 0

    ## Algorithm properties 
    P, Msub, Csub = x   
    M,N,K   ,atype,   btype,  ctype = P     # problem size and bytes
    MM,MN,MK,Matype, Mbtype, Mctype = Msub  # L2/Memtile problem size
    CM,CN,CK,Catype, Cbtype, Cctype = Csub  # L1/Core problem size   

    ddrbandwidth,MCbandwidth,Clock = list(p.flatten()) # Arch parameters 


    ## This is a C = A*B L2 - L1 computation with ping pong
    ## 
    P1  = comm_AB(Csub)/MCbandwidth ## prologue prefetch  A_ik and B_kj
    P2  = comm_C(Csub)/MCbandwidth  ## epilogue  send the C_ij 
    C1  = cycles(Csub)/Clock        ## compute 
    B1  = max(C1, max(P1, P2))      ## body: max( compute and comm)

    ### L2 -> L1
    ## A_i0, B_0j
    ## C_ij = sum_k=1^(n-1) A_i(k-1) *B_(k-1) |  A_ik, B_kj
    ## write C_ij
    
    T1  = P1+B1*math.ceil(MK/CK) + P2 ## K split computation


    ## M - core 
    ## MM, MN,MK is the problem in L2 
    SP = math.ceil((MM* MN)/(ROWS*COLS)) ## output points a core will
                                         ## compute overall 

    ## number of calls to a single core computation * the execution time 
    T1  = math.ceil(SP/(CM*CN)) * T1     

    ## L3 -> L2 
    P1 =  comm_AB(Msub)/ddrbandwidth ## prologue
    P2  = comm_C(Msub)/ddrbandwidth  ## epilogue
    C1  = T1                         ## compute There are 16 of these
    B1  = max(C1, max(P1, P2))       ## body
    T2  = P1+B1*math.ceil(M/MK) + P2 

    SP = math.ceil((M* M)/(MM*MN)) ## sub problem solved by a MemTile
    T2  = math.ceil(SP/(CM*CN)) * T2

    return T2

###
## We have
##    x : architecture parameters (the handle to minimize M_)
##    a : samples/time series
##    b : algorithm information 
###   

def cost_function(x , a, b  ):
    
    T = 0
    e = f_model(x,b)
    for r in a:
        T += (r-e)**2
    return T/len(a)



if __name__ == '__main__':

    
    
    P = [1024, 1024,1024, 16,16,16]
    M = [256,   256, 256, 16,16,16]
    C = [16,     16, 64, 16,16,16]

    targ =  [P,M,C]
    p =   numpy.array([  GB,  GB,   Clock])


    ## we create a reference time 
    T = f_model(p=p,x=targ)
    print("T", T)

    ## we create noise
    r = numpy.random.rand(10)**8
    print("r",r)

    
    Z = numpy.array([ T for i in range(10)])
    print("Z", Z)
    Z +=r 
    print("Z", Z)

    #Z is our series of times 
    
    print("M_",
        cost_function(
            p,
            Z,targ
        )
    )

    ## now we look for the p minimizing f_model so that our algorithm with
    ## this architecture parameter will provide the best approximation
    res = scipy.optimize.minimize(cost_function,
                                  p,
                                  args=(Z,targ),
                                  method='powell',
                                  bounds=((GB/10,2*GB),(GB/10,2*GB),(1/100*Clock,2*Clock)),
                                  options={ 'disp': True})


    print(res)
    print(res.x/10**9)
    print(p/10**9)
    print("PP", p-res.x)
    T1 = f_model(p=res.x,x=targ)
    print(sum(Z)/len(Z))
    print(T1)
