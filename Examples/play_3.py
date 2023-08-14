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
from Hw.hw import AbstractHW, PE
from Hw.hw_code import  ROCBLAS, GPU



HW =  AbstractHW('Doohm')

if __name__ == "__main__":


    print(" Fast Matrix Multiplication MxK * KxN -> MxN ")  
    print(" Introduce --M --K --N")
    print(" We show case the rest")
        
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--M", help="Factor of A", type=int, default=2)
    parser.add_argument("-n","--N", help="Factor of A", type=int, default=2)
    parser.add_argument("-e", "--error", help="pretty display error", type =str, default =None)
    parser.add_argument("-k","--K", help="Factor of A", type=int, default=3)
    args = parser.parse_args()

    
    if args.N is None: args.N = 2
    if args.M is None: args.M = 2
    
    import sys 
    X = args.M 
    Y = args.N

    T = X*Y*args.K

    print(T)
    A = gen_matrix(T,T,args.error)
    B = gen_matrix(T,T,args.error)


    ####
    ##  We do just the pure computations 
    ####
        
    ## Pure Python Interface
    print("compute regular ")
    start = time.time()
    C = A*B
    end = time.time()
    print("time",end - start)

    ## Bilinear using the deepmind format C^t = A*B
    fact =dict(numpy.load('factorizations_r.npz', allow_pickle=True))
    a,b,c = fact['%d,%d,%d' % (X,X,X)]
    at,bt,ct = fact['%d,%d,%d' % (Y,Y,Y)]
        
    ## factor X 
    print(a.shape)
    D = Scalar(0)*C
    ## compute and dependency .... 
    G3 = bini_mult_example(D,c, A,a,B,b,1)

    if False:
        Graph.heatmap_diff(Graph,Matrix(numpy.abs(C.value()-D.value())))
    #import pdb; pdb.set_trace()
    print(G3.V[1].pretty__q())
    print(G3.V[1].pretty__C())
    print(G3.pretty__())

    

    #start = time.time()
    #HW.compute_graph_by_queue_pool(G3)
    #end = time.time()
    #print("time",end - start)
    #import pdb; pdb.set_trace()
    
    #G3.compute()
    #Graph.heatmap_diff(Graph,Matrix(numpy.abs(C.value()-D.value())))
    ## reduce temporary space
    #G3.short_temp()
    #import pdb; pdb.set_trace()

    del G3; gc.collect()

    ## factor X 
    print(a.shape)
    D = Scalar(0)*C
    ## compute and dependency .... 
    G3 = bini_mult_example_three_temp(D,ct, A,at,B,bt,1,True)

    print(G3.pretty__())
    

    code = G3.pretty__C(python_compiler = True) 
    print(code)
    ROCBLAS.compile_and_import(
        code,
        TYP = str(Graph.numpytoC(G3.declarations[0][0].type_matrix())))

    import pdb; pdb.set_trace()
    import one
    start = time.time()
    H1 = Scalar(0)*C
    H = one.fastgemm(0,A.value().A.flatten(), B.value().A.flatten(), H1.value().A.flatten())
    R = numpy.matrix(
        H
    )
    B1 = R.reshape(C.value().shape)
    H = Matrix(B1)
    end = time.time()
    print("time",end - start)


    Graph.heatmap_diff(Graph,Matrix(numpy.abs(C.value()-D.value())))
    #G3.compute()
    #Graph.heatmap_diff(Graph,Matrix(numpy.abs(C.value()-D.value())))
    

    del G3; gc.collect()
   

    
