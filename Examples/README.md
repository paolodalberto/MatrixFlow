![Screenshot from 2023-01-13 22-43-49](https://user-images.githubusercontent.com/15663156/212459821-7e0d1894-cff5-42e2-8296-d62f9fd65ac2.png)

# Matrix Flow Examples

In this directory, we have a few examples. All of them are
work-in-progress but they all try to provide applications to the same
ideas.

Take three matrices A,B, and C and do C = AB using fast
algorithms. If you are using the deepmind repository the algorithms
are actually C^T =AB but the transposition is only for the
submatrices. If you are using the "bini" format you will have the
standard C = AB.

```play.py```

It is pure test where we take the at,bt,ct matrices representing an
fast algorithm, and we call a

```py from Graph.graph import bini```

I see Bini as the father of the fast algorithm field and the member of
my Bologna doctorate thesis committee. Here we can execute any fast
algorithm from the deepmind repository. We present cases only for
square matrices.

```play_2.py```

As above, but we create a computation DAG. The computation is written
into a graph you are going to execute. As in the validation, you can
create composed algorithms. Something that is relatively new and you
can create algorithm not available in the deepmind repository.

```py

    ## Bilinear using the deepmind format C^t = A*B
    fact     =dict(numpy.load('factorizations_r.npz', allow_pickle=True))
    a,b,c = fact['%d,%d,%d' % (X,X,X)]
    at,bt,ct = fact['%d,%d,%d' % (Y,Y,Y)]

    ## build and compute a graph
    G3 = bini_mult_example(D,c, A,a,B,b,1)

    ## compose a new algorithm and validate 
    c1,a1,b1 = bini_matrices_2(c,a,b, ct,at,bt,validate=c.shape[1]*ct.shape[1]<150)

    ## build a graph and execute it
    print(a1.shape)
    D = Scalar(0)*C
    G3 = bini_mult_example(D,c1, A,a1,B,b1,1)

```

The example play_3.py is nothing to brag about home.


```play_4.py``` is so much more interesting.

The idea is to build several algorithms (without validation because
they are correct). The strassen's algorithm is represented by the
factor 2 (we split the matrix size into 2 equal size and seven
products) and we look up the algorithm with factor 3 (three parts and
23 products).

Then we create the algorithm 2x2 (strasesn applied twice), 2x3, 3x2,
and 3x3. We build also the algorithms 2x2x3, 2x3x2, and 3x2x2. The
validation process explain in details the process. The simplest way to
describe the algorithm 2x2x3 is as follows:

The Strassen algorithm (2) is applied, the algorithm has 7
products. For each product we apply Strassen again and we have 7
sub-products for a total of 49. We use an algorithm 3 so that each
product is split into 23 for a total 161 products.

You play and appreciate the performance and error analysis for the
different algorithms 2x2x3, 2x3x2, and 3x2x2. Look for the *.png heat
error to enjoy the distribution of the error for each algorithms and
it maximum error (play with the error "up", "down", "middle" and see
the differences I spent countless hours watching them).

You can see there are two different way to create and execute the
computation. One is as above where each product is a temporary space
and one where we use a single produce (the minimum space required).


Python is a little tricky for memory management, but we can do a
reasonable job. 

```sh

## python path
export PYTHONPATH=$PWD

## play with a single core, more cores and less useful are fast algs.
export OPENBLAS_NUM_THREADS=1


paolo@fastmmw:~/MatrixFlow$ python3 Examples/play_4.py -e "up" -m 4 -n 9  -k 100 -v "t" 
 Fast Matrix Multiplication MxK * KxN -> MxN 
 Introduce --M --K --N
 We show case the rest
Matrix Size 3600
compute
time 3.123149871826172 GFLOPS 29.877528722449203
Alg 2
(1800, 1800) float64 7 TEMP SPACE GB 0.08448958396911621
Compute
compute 3.0043249130249023
Dependency
Maximum Error 1.8189894035458565e-12
Alg 3
(1200, 1200) float64 23 TEMP SPACE GB 0.1233816146850586
Compute
compute 3.137322425842285
Dependency
Maximum Error 3.410605131648481e-12
Alg 2x2
(900, 900) float64 49 TEMP SPACE GB 0.14785677194595337
Compute
compute 3.1504716873168945
Dependency
Maximum Error 4.888534022029489e-12
Alg 3x3
(400, 400) float64 529 TEMP SPACE GB 0.3153085708618164
Compute
compute 4.33605432510376
Dependency
Maximum Error 2.0804691303055733e-11
Alg 2x3
(600, 600) float64 161 TEMP SPACE GB 0.21591782569885254
Compute
compute 3.6656272411346436
Dependency
Maximum Error 1.0118128557223827e-11
Alg 3x2
(600, 600) float64 161 TEMP SPACE GB 0.21591782569885254
Compute
compute 3.6659934520721436
Dependency
Maximum Error 1.1596057447604835e-11
Alg 2x2x3
(300, 300) float64 1127 TEMP SPACE GB 0.37785619497299194
Compute
compute 5.856931447982788
Dependency
Maximum Error 2.0691004465334117e-11
Alg 2x3x2
(300, 300) float64 1127 TEMP SPACE GB 0.37785619497299194
Compute
compute 5.81263542175293
Dependency
Maximum Error 1.921307557495311e-11
Alg 3x2x2
(300, 300) float64 1127 TEMP SPACE GB 0.37785619497299194
Compute
compute 5.847616672515869
Dependency
Maximum Error 2.887645678129047e-11
Minimum space
Alg 2
(1800, 1800) float64 TEMP SPACE GB 0.012069940567016602
Compute
compute 3.0123097896575928
Dependency
Maximum Error 1.8189894035458565e-12
Alg 3
(1200, 1200) float64 TEMP SPACE GB 0.005364418029785156
Compute
compute 3.2532002925872803
Dependency
Maximum Error 3.410605131648481e-12
Alg 2x2
(900, 900) float64 TEMP SPACE GB 0.0030174851417541504
Compute
compute 3.5002994537353516
Dependency
Maximum Error 4.888534022029489e-12
Alg 3x3
(400, 400) float64 TEMP SPACE GB 0.0005960464477539062
Compute
compute 5.243701457977295
Dependency
Maximum Error 1.6370904631912708e-11
Alg 2x3
(600, 600) float64 TEMP SPACE GB 0.001341104507446289
Compute
compute 4.240681886672974
Dependency
Maximum Error 8.640199666842818e-12
Alg 3x2
(600, 600) float64 TEMP SPACE GB 0.001341104507446289
Compute
compute 4.195072889328003
Dependency
Maximum Error 8.86757334228605e-12
Alg 2x2x3
(300, 300) float64 TEMP SPACE GB 0.00033527612686157227
Compute
compute 7.156654119491577
Dependency
Maximum Error 2.114575181622058e-11
Alg 2x3x2
(300, 300) float64 TEMP SPACE GB 0.00033527612686157227
Compute
compute 7.08722710609436
Dependency
Maximum Error 1.864464138634503e-11
Alg 3x2x2
(300, 300) float64 TEMP SPACE GB 0.00033527612686157227
Compute
compute 7.18612265586853
Dependency
Maximum Error 1.9554136088117957e-11
```

It is a little disappointing to see that only Strassen's Algorithm (2)
has a clear advantage. The "compute time" spell out the execution time
"wall clock" for each algorithm.

Play with the size ```sh -m 4 -n 9 -k 100```. Play with the number of
cores ```export OPENBLAS_NUM_THREADS=1```. And play with the nature of
the matrices ```-e "up"``` (try "middle").

![Screenshot from 2023-01-13 22-46-53](https://user-images.githubusercontent.com/15663156/212459823-dea87b3c-1689-4d92-b897-4e46e922dc4f.png)