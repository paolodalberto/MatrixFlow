import matplotlib.pyplot as plt
import numpy as np
from scipy import special
import math

def s(x)    :
    return 1/(1 + np.exp(-x))
def gelu(x) :
    return x*(1+special.erf(x/math.sqrt(2)))/2
def t(x)    :
    return 0.5*x*(
        1+ np.tanh(math.sqrt(2/math.pi)*(x+0.044715*x**3))
    )

def ft( x):
    x2 = x * x
    a = x * (135135.0 + x2 * (17325.0 + x2 * (378.0 + x2)))
    b = 135135.0 + x2 * (62370.0 + x2 * (3150.0 + x2 * 28.0))
    return a / b


def t2(x):
    
    return 0.5*x*(1+ ft(math.sqrt(2/math.pi)*(x+0.044715*x**3)))
           


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
    

    #import pdb; pdb.set_trace()
    x = np.linspace(-5, 5,1000)
    y = gelu(x)


    y1= x*s(1.702*x)
    y2= t(x)
    y3= q(x)
    y4= t2(x)
    
    
                            #print(x)
    print(y)
    print(y1)
    print(y2)
    print(y2)
    print(y4)

    
    
    plt.plot(x, y, label = "gelu")
    plt.plot(x, y1, label="xs(x)")
    plt.plot(x, y2, label="t(x)") 
    plt.plot(x, y3, label="q(x)") 
    plt.plot(x, y4, label="t2(x)") 
    plt.legend()
   #plt.savefig("test.png")
    
    plt.show()
    plt.plot(x, y-y1, label = "gelu-xs")
    plt.plot(x, y-y2, label = "gelu-t")
    plt.plot(x, y-y3, label = "gelu-q")
    plt.plot(x, y-y4, label = "gelu-t2")
    
    plt.legend()

    plt.show()
