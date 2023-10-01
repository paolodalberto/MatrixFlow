import numpy
import math
from  Matrices.matrices import Matrix, PartitionMatrix, Vector, Scalar, read_alpha
from Graph.graph import Graph, bini, bini_matrices_2,bini_mult_example, gen_matrix,algorithm_mult_example
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
    parser.add_argument("-k","--K", help="Factor of A", type=int, default=5)
    parser.add_argument("-e", "--error", help="pretty display error", type =str, default =None)
    args = parser.parse_args()

    
    if args.N is None: args.N = 2
    if args.M is None: args.M = 2
    if args.K is None: args.K = 5
    
    import sys 
    X = args.M 
    Y = args.N
    K = args.K
    
    T = X*Y*K
    print("X*Y*5=",T)
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
    t = end-start
    print("time",t, "GFLOPS", 2*T*T*T / t)

    
    ## C_ij = sum_k A_ik B_kj
    D = Scalar(0)*C
    G1 = algorithm_mult_example(D, Scalar(1),A,B,X)
    if args.error and args.error !="anything": Graph.heatmap_diff(Graph,Matrix(numpy.abs(C.value()-D.value())))
    #G1.data_dependency()
    import pdb; pdb.set_trace()
    print(G1)
    del G1; gc.collect()

    
    ## Bilinear using the deepmind format C^t = A*B
    fact =dict(numpy.load('factorizations_r.npz', allow_pickle=True))
    a,b,c = fact['%d,%d,%d' % (X,X,X)]
    at,bt,ct = fact['%d,%d,%d' % (Y,Y,Y)]
        
    ## factor X 
    print(a.shape)
    D = Scalar(0)*C
    ## compute and dependency .... 
    G3 = bini_mult_example(D,c, A,a,B,b,1)
    if args.error and args.error !="anything": Graph.heatmap_diff(Graph,Matrix(numpy.abs(C.value()-D.value())))
    print(G3)
    import pdb; pdb.set_trace()
    del G3; gc.collect()

    ## Factor Y
    D = Scalar(0)*C
    print(at.shape)
    G3 = bini_mult_example(D,ct, A,at,B,bt,1)
    if args.error and args.error !="anything": Graph.heatmap_diff(Graph,Matrix(numpy.abs(C.value()-D.value())))
    del G3; gc.collect()
    
    
    try:
        ac,bc,cc = fact['%d,%d,%d' % (X*Y,X*Y,X*Y)]
        D = Scalar(0)*C
        G3 = bini_mult_example(D,cc, A,ac,B,bc,1)
        if args.error and args.error !="anything" : Graph.heatmap_diff(Graph,Matrix(numpy.abs(C.value()-D.value())))
        del G3; gc.collect()
    except Exception as e:
        print("Warning: very likely not found the algorithm", e)
        
        #import pdb; pdb.set_trace()
        
    ## X and Y 
    c1,a1,b1 = bini_matrices_2(c,a,b, ct,at,bt,validate=c.shape[1]*ct.shape[1]<150)
    print(a1.shape)
    D = Scalar(0)*C
    G3 = bini_mult_example(D,c1, A,a1,B,b1,1)
    if args.error and args.error !="anything": Graph.heatmap_diff(Graph,Matrix(numpy.abs(C.value()-D.value())))
    del G3; gc.collect()
    
    ## X and Y 
    c1,a1,b1 = bini_matrices_2(ct,at,bt, c,a,b,validate=c.shape[0]*ct.shape[0]<5)
    print(a1.shape)
    D = Scalar(0)*C
    G3 = bini_mult_example(D,c1, A,a1,B,b1,1)
    if args.error and args.error !="anything":Graph.heatmap_diff(Graph,Matrix(numpy.abs(C.value()-D.value())))
    del G3; gc.collect()
