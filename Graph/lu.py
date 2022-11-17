import numpy
import math
from  Matrices.matrices import Matrix, PartitionMatrix, Vector, Scalar
from  Graph.graph import Graph, Operation, Data

def lu(A : Matrix):
    AP = PartitionMatrix(A)
    AD = Data.data_factory('a', AP)

    LP = PartitionMatrix(A)
    LP.value()[0][1] = None
    LD = Data.data_factory('a', AP)
    

    UP = PartitionMatrix(A)
    UP.value()[1][0] = None
    UD = Data.data_factory('a', AP)
    
    Operation('0', 'lu', LD[0][0],
    

    


if __name__ == "__main__":
    

    A = Matrix(
        numpy.matrix(
            [
                [ (i+1)*(j+1) for i in range(3)] for j in range(3)
            ]
        )
    )

    L,U = lu(A)

    
    
