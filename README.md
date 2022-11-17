# Matrix Flow

There is a renaissance of fast matrix multiplication algorithms and we
think there is a need for new tools that we tried to build and use in
previous packages such as FastMMW. Once the tensor encodings are
spelled out, the algorithm is a sequence of scalar-by-matrix,
matrix-matrix addition and multiplication. We want to use these
algorithms are spring board for a framework for the computation of
matrix operations.

The advances of frameworks for convolution networks such as TensorFlow
and PyTorch show a different way of thinking and writing computations
by using DAG. Most importantly, there is a clear description of
operations based on full tensors, multiple input tensors, and one
output tensor. In the compiler world, this is equivalent to Single
Statement Assignment and data flow graph. The construction of one
tensor out of tensors are explicit (i.e., CONCATENATION) and they are
a disjoint partitions.

In this project we play with a few  basic ideas:

* compiler tools for the description of fast algorithms
* matrix partitions like concatenation to have a clean data dependency
* execution and verification of these algorithms
* hardware abstractions to play with parallelism
  * Code generation/execution for CPUs, FPGAs, and GPUs

```py
    X = 3

    A = Matrix(
        numpy.matrix(
            [
                [ i for i in range(X*2)] for j in range(X*2)
            ]
        )
    )

    B = Matrix(
        numpy.matrix(
            [
                [ i for i in range(X*2)] for j in range(X*2)
            ]
        )
    )

    alpha  = Scalar(1)
    alphai = Data('alpha', alpha)

    ## Pure Python Interface
    C = alpha*A*B
    print(C.value())

    ## Classic blocked matrix algorithm
    ## C_ij = \sum_k A_ik B_kj
    G1 = algorithm_mult_example(C, alpha,A,B,X)
    S = Schedule(G1)
    print(S.fit_hw_memory())
    ## distribute to HW and compute G1    
    S.naive_distribute_computation()

    ## deepmind format
    fact =dict(numpy.load('factorizations_r.npz', allow_pickle=True))
    a,b,c = fact['%d,%d,%d' % (X,X,X)]

    ## Fast MM algorithm
    ## P_i = (\sum A_i)*(\sum B_j) ... C_ij  = \sum P_i
    G2 = bini_mult_example(C,c, A,a,B,b)
    S2 = Schedule(G2)
    print(S2.fit_hw_memory())

    AAA = Algorithm(a,b,c)
    P =  AAA.partition_by_output(len(S.hw.pes))
    S2.naive_distribute_computation_by_output(P)
    
```
If you like to execute the code above ...
python Schedule/schedule.py


