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
import os
from Hw.hw_code import  ROCBLAS, GPU, BLAS
Line = "# %s %5s %5d %1.3e %1.5e" 
import one
from importlib import reload 
import traceback
import sys

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
    parser.add_argument("-b", "--build", help="buildone", type =str, default =None)
    parser.add_argument("-B", "--built", help="buildone use", type =str, default =None)
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



    
    if True:
        
        print("compute G")
        start = time.time()
        D = numpy.matmul(A.value(),B.value())
        end = time.time()
        t = end - start
        print("time",t)
        print("# type  alg  size gflops    max")
        print(Line %("pyt","reg", T,OPS/t/GIGA, numpy.max(C.value() - D)))
        #import pdb; pdb.set_trace()
        

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
        #'par' : Parallel,
        'one' : One
    }
    Dimension = {T : Performance}
    
    if args.built:
        FP = {
            '2one'       : one.fastgemm2one   ,
            '3one'       : one.fastgemm3one   ,
            '2x2one'     : one.fastgemm2x2one ,
            '2x2x3one'   : one.fastgemm2x2x3one,
            '2x3one'     : one.fastgemm2x3one ,
            '2x3x2one'   : one.fastgemm2x3x2one,
            '3one'       : one.fastgemm3one   ,
            '3x2one'     : one.fastgemm3x2one ,
            '3x2x2one'   : one.fastgemm3x2x2one,
            '3x3one'     : one.fastgemm3x3one 
        } 
    
    KEYS = list(FA.keys()) if args.minimumspaceonly is None else []
    codes = {}
    
    for ver in Performance.keys():
        if args.minimumspaceonly is not None and ver =='par':
            KEYS = [] 
        else: KEYS = list(FA.keys())
        print(ver,KEYS)
        for k in KEYS:
            print("Alg", k)
            #if k=="3x3": import pdb; pdb.set_trace()
            D = Scalar(0)*C
            c,a,b = FA[k]
            #G3 = bini_mult_example_three_temp(D,c, A,a,B,b) #,True,True)
            G3 = bini_mult_example(D,c, A,a,B,b) #,True,True)
            H = Scalar(1)*D
            if args.built:
                if "GPU" in os.environ:
                    H1 = Scalar(0)*D

                    
                    H = FP[k+ver](0,
                                    A.value().A.flatten('F'), A.value().shape[0],
                                    B.value().A.flatten('F'), B.value().shape[0],
                                    H1.value().A.flatten('F'),H1.value().shape[0]
                    )
                    R = numpy.matrix(
                        H
                    )
                    B1 = R.reshape((D.value().shape[0],D.value().shape[1]), order='F')
            
            
                else:
                    H1 = Scalar(0)*D
                    H = FP[k+ver](0,
                                    A.value().A.flatten(),  A.value().shape[1],
                                    B.value().A.flatten(),  B.value().shape[1],
                                    H1.value().A.flatten(),H1.value().shape[1]
                    )
                    R = numpy.matrix(
                        H
                    )
                    
                    B1 = R.reshape(D.value().shape)
                H = Matrix(B1)

                if D.padded:
                    shape = C.matrix.shape
                    print("MAX ERROR",
                          numpy.max(numpy.fabs((
                              H.value()[0:shape[0],0:shape[1]]-C.value()))))
                else:
                    print("MAX ERROR", numpy.max(numpy.fabs((H-C).value())))
                #import pdb; pdb.set_trace()
            if args.build:
                pdb.set_trace()
                if "GPU" in os.environ:
                    code = G3.pretty__C(header=k+ver,gpu=ROCBLAS) #python_compiler = True) 
                    codes[k+ver] = code
                    ROCBLAS.compile_and_import(
                        code,
                        TYP = str(Graph.numpytoC(G3.declarations[0][0].type_matrix())),
                        signature = "fastgemm"+k+ver,
                        append = len(codes)>1, compile = False
                    )
                else:
                    code = G3.pretty__C(header=k+ver,python_compiler=True,
                                        gpu=BLAS) #python_compiler = True) 
                
                    codes[k+ver] = code
                    try:
                        #print(code)
                        BLAS.compile_and_import(
                            code,
                            TYP = str(
                                Graph.numpytoC(G3.declarations[0][0].type_matrix())
                            ),
                            signature = "fastgemm"+k+ver,
                            append = len(codes)>1,
                            compile = False
                        )
                    except Exception as e:
                        print(e)
                        print(traceback.format_exc())
                        import pdb; pdb.set_trace()
            
                #
            if args.visual:
                E = numpy.abs(C.value()-H.value())
                if args.relative:
                    E = E/numpy.abs(C.value())
                Graph.heatmap_diff(Graph,Matrix(E),save=k+".png")
            else:
                shape = C.matrix.shape
                E = numpy.abs(C.value()-D.value()[0:shape[0], 0:shape[1]])
                print("MAX ERROR", numpy.max(E))
                print("MAX Relative ERROR", numpy.max(E/numpy.abs(C.value())))
                
            Dimension[T][ver][k] = {
                'time'      : G3.time,
                'gflops'    : OPS/GIGA/(G3.time if G3.time !=0 else 1),
                'max_error' :  numpy.max(E),
                'max_rel_error' :  numpy.max(E/numpy.abs(C.value())),
                'size' : T
            }
            del G3; gc.collect()
            #if k=="3x3": import pdb; pdb.set_trace()

    
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
                                 
if args.build:

    if "GPU" in os.environ:
        ROCBLAS.setup()
    else:
        BLAS.setup()
