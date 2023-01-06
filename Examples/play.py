import numpy
import math
from  Matrices.matrices import Matrix, PartitionMatrix, Vector, Scalar, read_alpha
from Graph.graph import Graph, bini, bini_matrices_2, gen_matrix
import networkx as nx
import matplotlib.pyplot as plt
import graphviz
import time
import seaborn as sns
from  Validation.BiniScheme import  BiniScheme
import sys
import argparse

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

    T = X*Y*5


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
    start = time.time()
    D = bini(D,c,A,a,B,b,recursion=1)
    end = time.time()
    print("time",end - start)
    Graph.heatmap_diff(Graph,Matrix(numpy.abs(C.value()-D.value())))

    ## Factor Y
    D = Scalar(0)*C
    print(at.shape)
    start = time.time()
    D = bini(D,ct,A,at,B,bt,recursion=1)
    end = time.time()
    print("time",end - start)
    Graph.heatmap_diff(Graph,Matrix(numpy.abs(C.value()-D.value())))

    try:
        ac,bc,cc = fact['%d,%d,%d' % (X*Y,X*Y,X*Y)]
        D = Scalar(0)*C
        print(ac.shape)
        start = time.time()
        D = bini(D,cc,A,ac,B,bc,recursion=1)
        end = time.time()
        print("time",end - start)
        Graph.heatmap_diff(Graph,Matrix(numpy.abs(C.value()-D.value())))
    except Exception as e:
        print("Warning: very likely not found the algorithm", e)
        
        #import pdb; pdb.set_trace()
        
    ## X and Y 
    c1,a1,b1 = bini_matrices_2(c,a,b, ct,at,bt,validate=c.shape[1]*ct.shape[1]<250)
    print(a1.shape)
    D = Scalar(0)*C
    start = time.time()
    D = bini(D,c1,A,a1,B,b1,recursion=1)
    end = time.time()
    print("time",end - start)
    Graph.heatmap_diff(Graph,Matrix(numpy.abs(C.value()-D.value())))

    ## X and Y 
    c1,a1,b1 = bini_matrices_2(ct,at,bt, c,a,b,validate=c.shape[0]*ct.shape[0]<5)
    print(a1.shape)
    D = Scalar(0)*C
    start = time.time()
    D = bini(D,c1,A,a1,B,b1,recursion=1)
    end = time.time()
    print("time",end - start)
    Graph.heatmap_diff(Graph,Matrix(numpy.abs(C.value()-D.value())))
