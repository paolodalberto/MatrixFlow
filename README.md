# Matrix Flow

There is a renaissance of fast matrix multiplication algorithms and we
need tools (some we tried to build previously in the FastMMW package). In practice, 
once the tensor encodings are spelled out, any fast algorithm is
a sequence of scalar-by-matrix, matrix-matrix addition,  and
matrix-matrix multiplication operations. We want to use these algorithms as a spring
board for the computation of matrix operations by creating a framework
and see what entails.

The advances in convolution networks are mostly due to the advances of 
frameworks such as Caffe, TensorFlow, and PyTorch. These show a
different way of thinking and writing computations by using DAG and
Python. In turn, new hardware and large systems, made achievable and applicable computation previously highly expensive.
Most importantly, there is a clear description of operations
using full tensors, multiple input tensors, and one output tensor. In
the compiler world, this is equivalent to Single Statement Assignment
and data flow graph. The construction of one tensor out of tensors are
explicit (i.e., CONCATENATION) and they are disjoint partitions.

As a compiler designer, where every operation is a multi nested loop,
operands are by pointers to memory, index computations are affine
spaces, and we cannot clearly distinguish a convolution from a matrix
multiplication. Working with clearly marked operations and on full
tensors, I can assure you is a relief.  

In this project we play with a few  basic ideas:

* Compiler tools for the description of fast algorithms
* Matrix partitions like concatenation to have a clean data dependency
* Execution and verification of these algorithms
* Hardware abstractions to play with parallelism (these are to come)
  * Code generation/execution for CPUs, FPGAs, and GPUs
* Validation of fast matrix multiplication
  * All algorithm are correct and validated (independely of error analysis).
* A new (at least to us) factorization of fast algorithms
  * Thus representation of regular recursive algorithms using matrices.


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


We prepare basically a few examples so far:

## You want to play: ```Examples/play.py```

This is a good starting point to play with matrix multiplication. We
extend a numpy matrix object, created a Matrix class, and overwritten
the (*,/,+,-). So you are free to run matrix operations. Given the
DeepMind format for the description of algorithms, you can take those
ndarray and create and execute any of those beautiful algorithms.

``` python3 Examples/play.py -n 2 -m 3 -e "down"```

The process is simple: given the input matrices A,B,C, and their
ndarray a,b,c. We create a matrix partition of the operand and then we
execute the fast algorithms step by step.

We also show how to build the fast algorithm from 2x2x2 and 3x3x3 (for
the 6x6x6 which is not available) and there are two ways. You can also
validate the correctness. All of this, you will see the graphical and
quantitative error analysis of each algorithm.


```sh
Fast Matrix Multiplication MxK * KxN -> MxN 
 Introduce --M --K --N
 We show case the rest
compute
time 0.00843358039855957
(9, 23)
time 0.07758116722106934
Maximum Error 1.1368683772161603e-12
(4, 7)
time 0.06120133399963379
Maximum Error 1.2789769243681803e-12
Warning: very likely not found the algorithm '6,6,6'

Validating Brent's equations for 6x6x6_161
....................................................................................................
Equations: 46,656
6x6x6_161 algorithm is OK! No errors found!
(36, 161)
time 0.20623421669006348
Maximum Error 6.366462912410498e-12
(36, 161)
time 0.24093103408813477
Maximum Error 5.5706550483591855e-12
```


Now using ```python3 Examples/play_2.py -n 2 -m 3 -e "down"``` You
will create a graph for each algorithm. Such a graph can be executed
but if you like you can now transform and optimize the execution using
all sort of trickery. You can partition the algorithm into multiple
sub-gpraph, you can change the schedule.

Also you can take the graph, print its schedule, compile it, and
executed.  Data declarations are needed for an complete data
dependency analysis where inputs and outputs must be explicitly
defined. However, this is still an interpreted environment (even by
compiling) and you can add temporary variables and computations as you like. 

``` python3 Examples/play_4.py``` 
Is an example how to collect performance (time, GFLOPS, maximum error, max relative, error, distribution of the error) for a complete set of algorithm if the natrix has the proper factors. Anyone can reproduce the performance numbers (in their system) and the performance plot created and stored in ```Plots```

![CORES_1](https://user-images.githubusercontent.com/15663156/212575773-fb8da32f-0af5-4f7b-9ec9-81faecd8b4c3.png)


## Validation

We show an application of validation: please go into Validation
directory for more details. This is the first time we can validate a
fast algorithm. The other way is to introduce integer values and compare algorithms for the same inputs. In so far we show performance of fully correct algorithms. Also, We believe this is the first time you can create a validate
algorithms as a factorization decomposition:

Consider a matrix A of size 6x6 (MxK) multiplying a matrix B of size 6x6 (KxN), using the Bini and DeepMind format, this is 6x6x6. We can use an even shorter notation for square problems: 6.  Then Algorithm 6 can be represented, by a factorization, 2 and 3 or a 3 and 2. The algorithm 2 splits the matrix operand by 2 into 4 submatrices, and instead of using 8 products, it uses 7 products.  Each product
is computed using algorithm 3: every product is decomposed into 23 products (for a total of 161 products) and we split further the matrix operands by three into 9 sub matrices. We can choose to start with algorithm 3 and then by algorithm 2. So we have 3x2 and 2x3 algorithms:

```sh
(36, 161)
time 0.20623421669006348
Maximum Error 6.366462912410498e-12
(36, 161)
time 0.24093103408813477
Maximum Error 5.5706550483591855e-12
```

The algorithm are both correct, they have the same number of products, but they have different performance and error property. 

Briefly: we have algorithm for factors: 2,3,4 (2x2),5,6 (by 2x3 and 3x2), (no 7), 8 (2x2x2), 9, 10 (2x5 and 5x2), (no 11), 12 (2x2x3,2x3x2,3x2x2), (no 13, 14), 15 (3x5 and 5x2) ... All of these you can build them and execut them here.     

I am trying to add graphical tools for the presentation of the data dependecy: for example  graph of Strassen Algorithm 
[Digraph.gv.pdf](https://github.com/paolodalberto/MatrixFlow/files/10095377/Digraph.gv.pdf)
but it is not very clear yet. 

The introduction of the case Graph/lu.py (the LU factorization) shows how the linear algebra computation could return multiple "matrices" such as P, L, U ...  but it is not a major concern at this time.

