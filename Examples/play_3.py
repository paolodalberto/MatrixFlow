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

if __name__ == "__main__":


    print(" Fast Matrix Multiplication MxK * KxN -> MxN ")  
    print(" Introduce --M --K --N")
    print(" We show case the rest")
        
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--M", help="Factor of A", type=int, default=2)
    parser.add_argument("-n","--N", help="Factor of A", type=int, default=2)
    parser.add_argument("-e", "--error", help="pretty display error", type =str, default =None)
    args = parser.parse_args()

    
    if args.N is None: args.N = 2
    if args.M is None: args.M = 2
    
    import sys 
    X = args.M 
    Y = args.N

    T = X*Y*10

    A = gen_matrix(T,T,args.error)
    B = gen_matrix(T,T,args.error)


    ####
    ##  We do just the pure computations 
    ####
        
    ## Pure Python Interface
    print("compute")
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

    
    Graph.heatmap_diff(Graph,Matrix(numpy.abs(C.value()-D.value())))
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
    G3 = bini_mult_example_three_temp(D,c, A,a,B,b,1)

    
    Graph.heatmap_diff(Graph,Matrix(numpy.abs(C.value()-D.value())))
    #G3.compute()
    #Graph.heatmap_diff(Graph,Matrix(numpy.abs(C.value()-D.value())))
    

    del G3; gc.collect()
   

    
