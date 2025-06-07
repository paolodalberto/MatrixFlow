###
## Assume the following
##
##  T = \sum_i^K A_i + \sum_k G_k
##
## The second part is a constant, K is fixed and the A_i are 'active'
## that is a function of a few parameters. T is a total count and we
## observing a single computation but repeated several time, say L
## times.
##
## \sum_k G_k is the start up cost and is independent from L
##
## A_i represents the contribution overall and at steady state is
## independent from the number of iterations L, we may consider the
## contribution per iteration:
##
## B_i,t=0 def is the contribution per iteration and we assume B_i,t ~
## B_i,0 : A_i/L and A_i ~ B_i,0*L A_i ~ \Sum_t=1^L B_i,t
##
## Note we do not have the contribution per iteration only the total
## and be happy that is split into K parts. Again B_i are independent
## (or partition without pairwise intersection)
##
## Now, Each iteration n in [1,L] works on a problem of know sizes P = [M,N ..]
##
## We estimate each component as linear regression
##
## B_i ~ alpha_i^t * P: no intercept cause it is in G_k we can
##       consider w.t.l.o.g. alpha and P the same length and introduce
##       zeros into alphas so we have a consistent representation
##
## So T ~ T(N,P, C_0)
##
## We model the computation by T = \sum_i^K A_i + \sum_k G_k
## \sum_i^K A_i = \sum_i^K B_i*N = N \sum_i^K alpha_i*P
## 
## of for at steady state observation (general step j)
##
## T_j = \sum_i^K alpha_i*P ... this is the linear regression
##
##
import pandas as pd
import numpy 
import math
import scipy 
import pdb

import matplotlib.pyplot as plt

def linear(x,p):
    return sum(x*p)

def cost_function(p,x):
    err = numpy.mean((x[:,0]-numpy.dot(x[:,1:],p))**2)
    return err


def fit(df,
        parameters = ['N', 'MH', 'NH'],
        active     = ['main', 'part1', 'part2', 'invsqrt']
        ):

    Q = []

    pdb.set_trace()
    for i in active:
        p = [ 1.0 for i in parameters]
        data = df[[i]+parameters]
        X = data.to_numpy()
        res = scipy.optimize.minimize(
            cost_function,
            p,
            args=(X),
            method='powell',
            options={ 'disp': True}
        )
        R = res.x.flatten()


        print(X[:,0],numpy.dot(X[:,1:],R))
        print("optimal  p",R)
        Q.append(res.x.flatten())
        
    return Q
    
if __name__ == '__main__':

  
    data_set  = [ # set of A_i 
        [1000, 1000, 1000, 1000, 1000, 10, 10,2 ], # t=1
        [1000, 1000, 1000, 1000, 1000, 5,  10, 4], # t=2
        [1000, 1000, 1000, 1000, 1000, 10, 10,2 ], # t=3
    ] 
    df = pd.DataFrame(data_set)
    df.columns = [
        'main', 'part1', 'part2', 'invsqrt', 'K0', ## data
        'N', 'MH', 'NH'                            ## parameters 
    ]
    print(df)

    fit(df)
