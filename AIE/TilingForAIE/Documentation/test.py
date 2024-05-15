
import numpy 

def rows(A):
    B = sum(A)
    return B

A0 = numpy.array([0,1,2,3]).reshape(2,2)
A1 = numpy.array([1,2,3,0]).reshape(2,2)

B0 = numpy.array([2,3,0,1]).reshape(2,2)
B1 = numpy.array([0,1,2,3]).reshape(2,2)


print(A0)

print(rows(A0))

import pdb; pdb.set_trace()
E1 = sum(A0)+sum(A1) 
print(E1)


R0 = (A0@B0 + A1@B1)/E1
R1 = ((A0/E1)@B0 + (A1/E1)@B1)

print(R0)
print(R1)

