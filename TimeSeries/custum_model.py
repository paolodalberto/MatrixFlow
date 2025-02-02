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
##          L2 -> L1 16 times in parallel  
##          C_ii,jj  = sum_k  A_ii,kk B_kk,jj
##
##    There is no description how the tiles are computed and there is no
###
def f_model(
        p : numpy.array = numpy.array([
            ddr_bandwidth,
            MC_bandwidth_channel,   
            Clock] ),
        x  : list = None,
):

    if x is None or (type(x) is list and len(x)!=3): return 0

    ## Algorithm properties: original problem and tiling for the
    ## memory a 3 level memory hierarchy. In summary, thi describe
    ## completely the algorithm but not completely the architecture

    P, Msub, Csub                   = x   
    M,N,K   ,atype,   btype,  ctype = P     # problem size and bytes
    MM,MN,MK,Matype, Mbtype, Mctype = Msub  # L2/Memtile problem size
    CM,CN,CK,Catype, Cbtype, Cctype = Csub  # L1/Core problem size   

    ddrbandwidth,MCbandwidth,Clock = list(p.flatten()) # Arch parameters 


    ## This is a C = A*B from L2 - L1 computation with ping pong/ pipelining
    ## In short notation  L2 -> L1
    ## Read A_i0, B_0j
    ## C_ij = sum_k=1^(n-1) A_i(k-1) *B_(k-1) ||  read A_ik, B_kj
    ## write C_ij

    P1  = comm_AB(Csub)/MCbandwidth ## prologue prefetch  A_ik and B_kj
    P2  = comm_C(Csub)/MCbandwidth  ## epilogue  send the C_ij 
    C1  = cycles(Csub)/Clock        ## compute 
    B1  = max(C1, max(P1, P2))      ## body: max( compute and comm)

    # Core/L1 computation time 
    T1  = P1+B1*math.ceil(MK/CK) + P2 ## K split computation
    
    ## M - core 
    ## MM,MN,MK is the problem in L2 and we split evenly across cores  
    SP = math.ceil((MM* MN)/(ROWS*COLS)) ## output points a core will
                                         ## compute overall 
    ## number of calls to a single core computation * the execution time 
    T1  = math.ceil(SP/(CM*CN)) * T1     

    
    ## L3 -> L2, now we feed the same algorithm to L2 (K split)
    P1 =  comm_AB(Msub)/ddrbandwidth ## prologue
    P2  = comm_C(Msub)/ddrbandwidth  ## epilogue
    C1  = T1                         ## compute There are 16 of these
    B1  = max(C1, max(P1, P2))       ## body
    T2  = P1+B1*math.ceil(M/MK) + P2 ## K split 

    SP = math.ceil((M* M)/(MM*MN)) ## sub problem solved by a MemTile
    T2  = math.ceil(SP/(CM*CN)) * T2

    return T2

###
## We have
##    x : architecture parameters (the handle to minimize M_)
##    a : samples/time series
##    b : algorithm information
##    L2 .... but the model produce a single output
###   

def cost_function(x , a, b  ):
    
    T = 0
    e = f_model(x,b)
    for r in a:
        T += (r-e)**2
    return T/len(a)


def case_or_use(Z, p,targ, f_model, cost_function):

    #Z is our series (but time does not matter)
    
    print("Cost function to minimize ",
        cost_function(
            p,
            Z,targ
        )
    )

    ## now we look for the p minimizing f_model so that our algorithm
    ## with this architecture parameter will provide the best
    ## approximation
    res = scipy.optimize.minimize(cost_function,
                                  p,
                                  args=(Z,targ),
                                  method='powell',
                                  bounds=((p[0]/10,p[0]),(p[1]/10,p[1]),(p[2]/10,p[2])),
                                  options={ 'disp': True})

    R = res.x.flatten()
    print("optimal  p",R/G)
    print("original p",p/G)
    print("PP", (p-R)/G)

    return R
    

    
    


if __name__ == '__main__':


    P = [1024, 1024,1024, 16,16,16]  ## problem size 
    M = [256,   256, 256, 16,16,16]  ## L2 tiles 
    C = [16,     16, 64, 16,16,16]   ## L1 tiles 

    ## target algorithm
    targ =  [P,M,C]  

    # Architecture parameters 
    p =   numpy.array([  GB,  GB,   Clock])

    ## we create a reference time 
    T = f_model(p=p,x=targ)
    print("Time ", T)

    ## we create noise: positive and this represent the a percentage
    ## of the signal
    r = numpy.random.rand(10)**2
    print("r",r)

    Z = numpy.array([ T for i in range(10)])
    print("Z", Z)
    
    Z +=  Z*r/100 
    print("Z", Z)

    

    ## this is the p we estimate from the data 
    R = case_or_use(Z[:-1], p,targ, f_model,cost_function)

    
    ## Example of using the tuned model  

    
    T1 = f_model(p=R,x=targ)
    print(T1)  # fitted
    print(T)   # golden 

    mu = numpy.mean(Z[:-1])
    s  = math.sqrt(numpy.var(Z[:-1]))

    print(Z[-1], T1)
    if numpy.abs(T1-Z[-1])>3*s:
        
        print("Warning > 3sigma")
    else:
        print("All is well")
