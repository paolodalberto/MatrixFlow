import numpy
import math
from  Matrices.matrices import Matrix, PartitionMatrix, Vector, Scalar, read_alpha
from Graph.graph import Graph, bini, bini_matrices_2,bini_mult_example, gen_matrix,bini_mult_example_three_temp
import networkx as nx
import matplotlib.pyplot as plt
import graphviz
import time
import seaborn as sns
from  Validation.BiniScheme import  BiniScheme
import sys
import argparse
import gc

Line = "# %s %5s %5d %1.3e %1.5e" 


if __name__ == "__main__":


    print(" Fast Matrix Multiplication MxK * KxN -> MxN ")  
    print(" Introduce --M --K --N")
    print(" We show case the rest")
        
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--M", help="Factor of A", type=int, default=4)
    parser.add_argument("-n","--N", help="Factor of A", type=int, default=3)
    parser.add_argument("-k","--K", help="Factor of A", type=int, default=3)
    parser.add_argument("-e", "--error", help="float operands for  error analysis", type =str, default =None)
    parser.add_argument("-v", "--visual", help="pretty display error", type =str, default =None)
    parser.add_argument("-r", "--relative", help="pretty display error", type =str, default =None)
    parser.add_argument("-s", "--minimumspaceonly", help="pretty display error", type =str, default =None)
    args = parser.parse_args()

    
    
    import sys 
    X = args.M 
    Y = args.N
    K = args.K
    
    F = [2,2,3]  ## we are going to work with 12x12x12 algorithms
                 ## with matrices 36x36
                 
    T = X*Y*K # 4x3x3 
    OPS = 2*T*T*T
    GIGA= 1000000000
    print("Matrix Size", T)
    for f in F:
        if T % f !=0:
            print("BYE BYE  use default dimensions")
            sys.exit()
    
    A = gen_matrix(T,T,args.error)
    B = gen_matrix(T,T,args.error)

    #import pdb; pdb.set_trace()
    ####
    ##  We do just the pure computations 
    ####
        
    ## Pure Python Interface
    print("compute")
    start = time.time()
    C = A*B
    end = time.time()
    t = end-start
    print("time",t, "GFLOPS", OPS/t/GIGA)

    print("# type  alg  size gflops    max err")
    print(Line %("pyt","reg", T,OPS/t/GIGA, 0))

    if False:
        print("compute")
        start = time.time()
        D = numpy.matmul(A.value(),B.value())
        end = time.time()
        t = end - start
        print("time",t)
        print("# type  alg  size gflops    max")
        print(Line %("pyt","reg", T,OPS/t/GIGA, numpy.max(C.value() - D.value())))



    fact =dict(numpy.load('factorizations_r.npz', allow_pickle=True))
    FA = {}
    for f in F:
        ## Bilinear using the deepmind format C^t = A*B
        AA = '%d,%d,%d' % (f,f,f)
        FA['%d' % f ] = fact[AA]

    
    c,a,b = FA['2']
    FA['2x2'] = bini_matrices_2(c,a,b, c,a,b,False)

    c,a,b = FA['3']
    FA['3x3'] = bini_matrices_2(c,a,b, c,a,b,False)

    c,a,b    = FA['2']
    ct,at,bt = FA['3']
    FA['2x3'] = bini_matrices_2(c,a,b, ct,at,bt,False)
    FA['3x2'] = bini_matrices_2(ct,at,bt, c,a,b,False)

    ## 2x2x3
    c,a,b    = FA['2x2']
    ct,at,bt = FA['3']
    FA['2x2x3'] = bini_matrices_2(c,a,b, ct,at,bt,False)

    ## 2x3x2
    c,a,b    = FA['2']
    ct,at,bt = FA['3x2']
    FA['2x3x2'] = bini_matrices_2(c,a,b, ct,at,bt,False)

    ## 3x2x2 
    c,a,b    = FA['3x2']
    ct,at,bt = FA['2']
    FA['3x2x2'] = bini_matrices_2(c,a,b, ct,at,bt,False)
    
    #import pdb; pdb.set_trace()

    Parallel = {}
    One      = {}     
    Performance = {
        'par' : Parallel,
        'one' : One
    }
    Dimension = {T : Performance}

    
    
    KEYS = list(FA.keys()) if args.minimumspaceonly is None else []


    for ver in Performance.keys():
        if args.minimumspaceonly is not None and ver =='par':
            KEYS = [] 
        else: KEYS = list(FA.keys())
        print(ver,KEYS)
        for k in KEYS:
            print("Alg", k)
            
            D = Scalar(0)*C
            c,a,b = FA[k]
            if  ver =='par':
                G3 = bini_mult_example(D,c, A,a,B,b,1)
            else:
                G3 = bini_mult_example_three_temp(D,c, A,a,B,b) #,True,True)

            if args.visual:
                E = numpy.abs(C.value()-D.value())
                if args.relative:
                    E = E/numpy.abs(C.value())
                Graph.heatmap_diff(Graph,Matrix(E),save=k+".png")
            else:
                E = numpy.abs(C.value()-D.value())
                print("MAX ERROR", numpy.max(E))
                print("MAX Relative ERROR", numpy.max(E/numpy.abs(C.value())))
                
            Dimension[T][ver][k] = {
                'time'      : G3.time,
                'gflops'    : OPS/G3.time/GIGA,
                'max_error' :  numpy.max(E),
                'max_rel_error' :  numpy.max(E/numpy.abs(C.value())),
                'size' : T
            }
            del G3; gc.collect()


    
    #print(Dimension)
    ##     type alg size gflops max rel 
    for size in Dimension:
        for typ  in Dimension[size]:
            for alg in  Dimension[size][typ]:
                cont = Dimension[size][typ][alg]
                print(Line %(typ,alg,
                             cont['size'],
                             cont['gflops'],
                             cont['max_error'])
                )
                                 
