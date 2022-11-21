import numpy
import math
from  Matrices.matrices import Matrix, PartitionMatrix, Vector, Scalar
from  Graph.graph import Graph, Operation, Data, Function
import scipy
import scipy.linalg

## L*X = B L lower triangular
def left_lower_triang(L, B):
    return  [ scipy.linalg.solve_triangular(L,B,lower=True) ]

## X*U = B U upper
## (X*U)^t = B^t L*X^t = B^t

def right_upper_triang(U, B):
    X = scipy.linalg.solve_triangular(
        numpy.transpose(U),
        numpy.transpose(B),
        lower=True)
    return [ numpy.transpose(X)]

def lu(A : Matrix):
    AP = PartitionMatrix(A)
    AD = Data.data_factory_flat(Data.data_factory('A', AP))


    
    
    LP = PartitionMatrix(Matrix(A.value()*0.0))
    LD = Data.data_factory_flat(Data.data_factory('L', LP))
    
    UP = PartitionMatrix(Matrix(A.value()*0.0))
    UD = Data.data_factory_flat(Data.data_factory('U', UP))

    PP = PartitionMatrix(Matrix(A.value()*0.0))
    PD = Data.data_factory_flat(Data.data_factory('P', PP))

    print(AD)
    import pdb; pdb.set_trace()
    decls =  [AD,LD,UD,PD]

    ## P0,L0,U0  = LU(A0)
    q1 = Operation(
        'q1', '=',
        [PD[0],LD[0],UD[0]],
        Function('lu', scipy.linalg.lu,
                 [AD[0]]
        )
    )
    
    
    q1.compute()
    print("P_0",PD[0].left.value(),
          "\n L_0", LD[0].left.value(),
          "\n U_0", UD[0].left.value(),
          "\n A_0", AD[0].left.value()
    )
    import pdb; pdb.set_trace()


    ## L0*U1 = A1 -> U1
    q2 = Operation(
        'q2', '=',
        [UD[1]],
        Function('tri', left_lower_triang,[LD[0], AD[1]])
    )
    
    q2.compute()
    print("U_1",    UD[1].left.value(),
          "\n L_0", LD[0].left.value(),
          "\n A_1", AD[1].left.value())

    import pdb; pdb.set_trace()
    
    ## L2*U0 = A2 -> U0^t L2^t = A2^t -> L2^t 
    q3 = Operation(
        'q3', '=',
        [LD[2]],
        Function(
            'utri',right_upper_triang,  [UD[0],AD[2]])
    )
    q3.compute()
    print("L_2",    LD[2].left.value(),
          "\n U_0", UD[0].left.value(),
          "\n A_2", AD[2].left.value())
    

    import pdb; pdb.set_trace()

    ## Ashur = A2 - L2*U1
    q4 = Operation(
        'q4', '=',
        LD[1],
        Operation(
            'shur', '-',
            AD[3],
            Operation('p', '*',
                      LD[2], UD[1]
            )
        )
    )
    q4.compute()
    print("L_1",    LD[1].left.value(),
          "\n A_3", AD[3].left.value(),
          "\n L_2", LD[2].left.value(),
          "\n U_1", UD[1].left.value()
    )

    import pdb; pdb.set_trace()

    ## P0,L0,U0  = LU(A0)
    q5 = Operation(
        'q5', '=',
        [PD[3],LD[3],UD[3]],
        Function('lu', scipy.linalg.lu,
                 [LD[1]]
        )
    )

    q5.compute()
    print("P_3",PD[3].left.value(),
          "\n L_3", LD[3].left.value(),
          "\n U_3", UD[3].left.value(),
          "\n L_1", AD[1].left.value()
    )
    import pdb; pdb.set_trace()
    
    ###
    ## create a graph
    ###
    G1 = Graph("C = LU(A)", [q1,q2,q3,q4,q5],decls)
    print(G1)

    ###
    ## Compute the graph for validation. Yep we can and we should run
    ## the graph
    ###
    G1.compute()

    ## we create a stmt-by-stm data dependency
    G1.dependency()

    return G1

    
    


if __name__ == "__main__":
    

    A = Matrix(
        numpy.matrix(
            [
                [ (i+j+numpy.random.rand())*(i*j+i+numpy.random.rand()) for i in range(4)] for j in range(4)
            ]
        )
    )

    P,L,U = scipy.linalg.lu(A.value())
    print(P)
    print(L)
    print(U)
    import pdb; pdb.set_trace()
                            
    
    G = lu(A)

    
    
