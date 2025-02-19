import numpy
import math


def the_five_range(x) : return [x-x*5/100, x+x*5/100]

if __name__ == '__main__':


    R  = numpy.random.rand(10)
    N  = numpy.array([i for i in range(10)])

    y=N+R

    print("series", y)
    
    B =  the_five_range(y[-2])
    print("bound",B)
    if B[0]<y[-1] and y[-1]<B[1]:
        
        print("All is well ")
    else:
        print("Warning ")
