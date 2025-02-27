import matplotlib.pyplot as plt
import numpy as np
from scipy import special
import math


def erf(x):
    
    Q1 =  np.float16(1.128379167095476)
    Q3 =  np.float16(-0.3761263890318252)
    Q5 =  np.float16( 0.1128379167095475)
 
    one = np.float16( 1.0)
    minus_one =  np.float16(-1.0)
                      
    y = np.ndarray.astype(x,np.float16)
    r = y*(Q1 +Q3*y**2 +Q5*y**4)
    r = np.minimum(r,1.0)
    r = np.maximum(r,-1.0)
    
    return r

def gelu(x) :
    return x*(1+special.erf(x/math.sqrt(2)))/2


def s(x)    :
    y = np.ndarray.astype(x,np.float16)
    return 1/(1 + np.exp(-y))

def t(x)    :

    y = np.ndarray.astype(x,np.float16)
    z = np.float16(np.sqrt(2/math.pi))
    
    return 0.5*y*(
        1+ np.tanh(z*(y+0.044715*y**3))
    )

def A(x) :
    y2 = y * y
    a = y * (1.0 + y2 * (17325.0/135135.0 + y2 * (378.0 + y2)/135135.0       ))
    return a
def B(x) :
    y2 = y * y
    b =      1.0 + y2 * (62370.0/135135.0 + y2 * (3150.0 + y2 * 28.0)/135135.0)
    return b

def ft( x):


    y = np.ndarray.astype(x,np.float16)
    y2 = y * y

    k0 = np.float16(17325.0/135135.0)
    k1 = np.float16(62370.0/135135.0)
    k2 = np.float16(3150.0/135135.0)
    k3 = np.float16(28.0/135135.0)
    
    #a = y * (135135.0 + y2 * (17325.0 + y2 * (378.0 + y2)))
    a = y * (1.0 + y2 * (k0 + y2 * (378.0 + y2)/135135.0))
    #b =      135135.0 + y2 * (62370.0 + y2 * (3150.0 + y2 * 28.0))
    b =      1.0 + y2 * (k1 + y2 * (k2 + y2 * k3))

    r = a/b


    

    return r

# This is the erf approximation
def erf_p(x):
    #constants
    a1 = 0.254
    a2 = -0.284
    a3 = 1.4214
    a4 = -1.453
    a5 = 1.061
    p = 0.327

    # Save the sign of x
    sign = np.sign(x)
    x = np.fabs(x);

    # A&S formula 7.1.26
    t = 1.0 / (1.0 + p * x);
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x);

    return sign * y

def gelu_p(x) :
    return x*(1+erf_p(x/math.sqrt(2)))/2

def t2(x):

    
    y = np.ndarray.astype(x,np.float16)
    z = np.float16(np.sqrt(2/math.pi))

    
    g = 1+ ft(z*(y+0.044715*y**3))

    return 0.5*y*g
           
def t3(x):
    y = np.ndarray.astype(x,np.float16)
    q = np.float16(1/np.sqrt(2))
    z = y*0.5*(1+erf(y*q))
    return z

def q(x):
    p1 = 1.702
    q0 = 1/2
    q1 = 0.25
    q2 = 0.03125


    mul1=p1*x

    ## sigmoid 
    mul1 = np.minimum(mul1,4)
    mul1 = np.maximum(mul1,-4)
    mul1_abs = np.fabs(mul1)

    
    mul0 = q2*mul1_abs
    mac0 = q0+mul1*q1
    msc = mac0 - mul0*mul1
    mul2 = msc*x
    return mul2
    
if __name__ == "__main__":
    

    #
    x = np.linspace(-4, 10,1000)
    y = gelu(x)

    x = np.ndarray.astype(x,np.float16)

    y1= x*s(1.702*x)
    y2= t(x)
    y3= q(x)

    y4= t2(x)
    y5= t3(x)
    y6= gelu_p(x)

    print(np.max(A(x)))
    print(np.max(B(x)))
    import pdb; pdb.set_trace()
    #print(x)
    #print(y)
    #print(y1)
    #print(y2)
    #print(y2)
    #print(y4)

    
    
    #plt.plot(x, y, label = "gelu")
    #plt.plot(x, y1, label="xs(x)")
    #plt.plot(x, y2, label="t(x)") 

    #plt.plot(x, y3, label="q(x)") 
    #plt.plot(x, y4, label="t2(x)") 
    #plt.legend()
    
    
    #plt.show()
    #plt.plot(x, y-y1, label = "gelu-xs")
    plt.plot(x, y-y2, label = "gelu-t " + str(np.max(np.fabs(y-y2))))
    plt.plot(x, y-y3, label = "gelu-q "+ str(np.max(np.fabs(y-y3))))
    #plt.plot(x, y-y4, label = "gelu-t2 "+ str(np.max(np.fabs(y-y4))))
    #plt.plot(x, y-(np.maximum(y3,y4)), label = "gelu-tq2")
    #plt.plot(x, y-y5, label = "gelu-erf")
    plt.plot(x, y-y6, label = "gelu_p" + str(np.max(np.fabs(y-y6))))
    
    plt.legend()
    plt.title("float 16")
    plt.xlabel("x")
    plt.ylabel("signed error")
    plt.savefig("test.png")
    plt.show()
