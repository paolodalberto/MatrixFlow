
import numpy
import math
import scipy 

#import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
#import statsmodels.api as sm
#from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.arima.model import ARIMA
import numpy

###
##  Time series classic y_i = alpha y_(i-1) + x_i + beta x_(i-1)
##  ARIMA
##
##  create a model
##  forecast
##  example
###

def fit_arima(df, p : int = 1,d : int = 1 ,q : int =1) :
    # Fit ARIMA model
    model = ARIMA(df, order=(p, d, q))
    model_fit = model.fit()

    # Summary of the model
    print(model_fit.summary())

    return model_fit, model

def forecast_arima(model, step : int =2):
    cast = model.forecast(step)
    print(cast)
    return cast

def arima_use(
        y :numpy.array = numpy.random.rand(10),
        p : int = 1,d : int = 1 ,q : int =1
): 
    mu = numpy.mean(y)
    var = numpy.var(y)
    std = numpy.sqrt(var)

    print(mu,std)
    
    model = ARIMA(y[:-1], order=(p, d, q))
    model_fit = model.fit()
    
    ## we forecast the last two points 
    f=list(model_fit.forecast(1).flatten())
    ym2 = y[:-1] + f

    print(f, y[-1])
    if numpy.abs(f[0]-y[-1])>2*std:
        
        print("Warning > 2sigma")
    else:
        print("Warning < 2sigma")

####
##
## how to build and fit  a design model
##
##
##
##
###


Byte = 2**3
G = (2**10)**3
GB = G*Byte

GHz = 10**9
Clock = 2*GHz

ROWS = 4
COLS = 4 
CORES = ROWS * COLS  

## Small cost functions

def cycles(x):
    m,n,k,atype,btype,ctype = x
    B = 256
    if   ctype == 8 :         B = 256
    elif ctype == 16:         B = 64
    elif ctype == 32 :        B = 16
    
    return math.ceil(m*n*k/B)

def core_gemm(x):
    return cycle(x)/Clock
    

def comm_AB(x):
    m,n,k,atype,btype,ctype = x
    return m*k*atype + n*k*btype

def comm_C(x):
    m,n,k,atype,btype,ctype = x
    return m*n*ctype

###
## Time estimate: 
##     p = set of parameter describing the features of the architecture
##     x = set of parameters describing the algorithms and problem size, see inside the code 
##
###
def time_estimates(
        p : numpy.array = numpy.array([  2* 4*GB,  4*GB,   Clock] ),
        x  : list = None,
):

    if x is None or (type(x) is list and len(x)!=3): return 0

    ## Algorithm properties 
    P, Msub, Csub = x   
    M,N,K   ,atype,   btype,  ctype = P     # problem size and bytes
    MM,MN,MK,Matype, Mbtype, Mctype = Msub  # L2/Memtile problem size
    CM,CN,CK,Catype, Cbtype, Cctype = Csub  # L1/Core problem size   

    ddrbandwidth,MCbandwidth,Clock = list(p.flatten()) # Arch parameters 

    
    P1  = comm_AB(Csub)/MCbandwidth ## prologue
    P2  = comm_C(Csub)/MCbandwidth  ## epilogue
    C1  = cycles(Csub)/Clock        ## compute 
    B1  = max(C1, max(P1, P2))      ## body: max( compute and comm)

    ##      
    T1  = P1+B1*math.ceil(MK/CK) + P2 ## K split computation


    ## M - core
 
    SP = math.ceil((MM* MN)/(ROWS*COLS)) ## sub problem solved by a core
    T1  = math.ceil(SP/(CM*CN)) * T1

    P1 =  comm_AB(Msub)/ddrbandwidth ## prologue
    P2  = comm_C(Msub)/ddrbandwidth  ## epilogue
    C1  = T1                         ## compute 
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

def M_(x , a, b  ):
    
    T = 0
    e = time_estimates(x,b)
    for r in a:
        T += (r-e)**2
    return T



if __name__ == '__main__':

    import pdb; pdb.set_trace()
    arima_use()
    
    
    P = [1024, 1024,1024, 16,16,16]
    M = [256,   256, 256, 16,16,16]
    C = [16,     16, 64, 16,16,16]

    targ =  [P,M,C]
    p =   numpy.array([  GB,  GB,   Clock])


    ## we create a reference time 
    T = time_estimates(p=p,x=targ)
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
        M_(
            p,
            Z,targ
        )
    )

    ## now we look for the p minimizing M_ so that our algorithm with
    ## this architecture parameter will provide the best approximation
    res = scipy.optimize.minimize(M_,p,args=(Z,targ),
                                  method='powell',
                                  bounds=((GB/10,2*GB),(GB/10,2*GB),(1/100*Clock,2*Clock)),
                                  options={ 'disp': True})


    print(res)
    print(res.x/10**9)
    print(p/10**9)
    print("PP", p-res.x)
    T1 = time_estimates(p=res.x,x=targ)
    print(sum(Z)/len(Z))
    print(T1)
