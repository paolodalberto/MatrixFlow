

import numpy
import math
from  Matrices.matrices import Matrix, PartitionMatrix, Vector, Scalar, read_alpha
from Graph.graph import Graph, bini, bini_matrices_2,bini_mult_example, Data, Operation,gen_matrix,algorithm_mult_example, Loop
import networkx as nx
import matplotlib.pyplot as plt
import graphviz
import time
import seaborn as sns
from  Validation.BiniScheme import  BiniScheme
import sys
import argparse
import gc
import pdb

#####
## We start a conversation how to represent the computation of matrix
## multiplicationfor for a sistolic architecture 
##
## We describe blocked matrix computations and we want to describe a
## hierarchical computaiton L3-L2-L3 and for each component L2-L1-L2
###


###
##  Partition(A)*Partition(B) -> Partition(C) we return declarations
##  and assignements for the classic matrix multiplication. We give
##  partitions as inputs 
def Addition_t(A : PartitionMatrix, B : PartitionMatrix, C: PartitionMatrix) -> list:

    ## we create Data type for each elements and this is the leaves of
    ## the basic computation
    AD = Data.data_factory_2d('ADP', A); ADT = Data.data_factory_flat(AD) 
    for i in ADT: i.inputs = True
    BD = Data.data_factory_2d('BDP', B); BDT = Data.data_factory_flat(BD)
    for i in BDT: i.inputs = True
    CD = Data.data_factory_2d('CDP', C) 
    CDT = Data.data_factory_flat(CD)
    for i in CDT: i.outputs = True

    Row = len(C.l)    # of the output partition
    Col = len(C.l[0])    # of the output partition
    BRow   = len(B.l)
    BCol   = len(B.l[0])
    ACol = len(A.l[0])
    ARow = len(A.l)

    
    
    print("Add", Row,Col,ACol)
    decls = [      ADT ,        BDT,          CDT ] 
    
    ###
    ## Computation as a sequence of assignment statements
    ## and binary operations. 
    ###
    V = []
    for i in range(min(Row,BRow,ARow)):  ## parallem in M
        for j in range(min(Col,BCol,ACol)):  ## parallel in N 
            T = Operation("p0", "+", AD[i][0] ,BD[0][j])
            R = Operation("s0", '<<', CD[i][j],T )
            #print(R)
            R.parallel = True
            V.append(R)
            
    #for v in V:
    #    print(v)
    
    return decls, V

def Addition_2(A : Matrix, B: Matrix, C: Matrix,
              MT : list,
              CT : list
    ) -> list:

    M,N = MT


    ## this describes the decomposition of the L3 matrix into L2 sub
    ## matrix without overlap:  partition at memtile.
    
    CPT = PartitionMatrix(C,[M,N])
    BPT = PartitionMatrix(B,[M,N])
    APT = PartitionMatrix(A,[M,N])
    
    
    Row = len(CPT.l)    # of the output partition
    Col = len(CPT.l[0])    # of the output partition
    BRow   = len(BPT.l)
    BCol   = len(BPT.l[0])
    ACol = len(APT.l[0])
    ARow = len(APT.l)



    M,N = CT
    
    ###
    ## Computation as a sequence of assignment statements
    ## and binary operations  
    ###
    V = []
    for i in range(min(Row,BRow,ARow)):  ## parallel M 
        for j in range(min(Col,BCol,ACol)):  ## Parallel N 
            
                
                ## Partition of a partition ( is it working? )
                CPC = PartitionMatrix(CPT.value()[i][j],[M,N])
                BPC = PartitionMatrix(BPT.value()[i][j],[M,N])
                APC = PartitionMatrix(APT.value()[i][j],[M,N])
                decl, vs = Addition_t(APC,BPC,CPC)
                gr = Graph("C = alpha*A*B", vs,decl,C)
                ## bookkeeping to remember the partition shape and format
                gr.CDP = CPC
                gr.ADP = APC
                gr.BDP = BPC
                V.append(gr)

    ###
    ## create a graph
    ###
    pdb.set_trace()
    L = Loop(
        "T",   ## You got to have a name
        V,
        [list(range(Row)),list(range(Col))],
        [APT, BPT, CPT] ## book keeping for this parition 
    )

    
    return  L


if __name__ == "__main__":




    print(" Introduce --M --K --N and tiling ")
    print(" We show case the rest")
    print(" At this time we assume that you create the decomposition so that everything click into place")
    print(" TilingAIE will provide the tiling information for the code generation.")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--M",  help="Factor of A MxK @ L3", type=int, default=8)
    parser.add_argument("-n", "--N",  help="Factor of B KxN @ L3 ", type=int, default=2048)
    parser.add_argument("-mt","--MT", help="Factor of A MTxKT @ L2", type=int, default=8)
    parser.add_argument("-nt","--NT", help="Factor of B KTxNT @ L2", type=int, default=64*4)
    parser.add_argument("-mc","--MC", help="Factor of A MCxKC @ L1", type=int, default=8)
    parser.add_argument("-nc","--NC", help="Factor of B KCxNC @ L1", type=int, default=64)
    parser.add_argument("-e", "--error", help="pretty display error", type =str, default =None)
    args = parser.parse_args()

    
    import sys 
       
    A = gen_matrix(args.M,args.N,args.error)
    B = gen_matrix(args.M,args.N,args.error)
    C = gen_matrix(args.M,args.N,args.error)
    

    ## disjoint partition of input output at core level 
    CPC = PartitionMatrix(C,[args.MC,args.NC])
    BPC = PartitionMatrix(B,[args.MC,args.NC])
    APC = PartitionMatrix(A,[args.MC,args.NC])

    ## we create the sequence of matrix operation describing the
    ## larger product. this is a classic matrix multiplication where
    ## the focus and granularity is the core subvolume
    decls, V = Addition_t(APC, BPC, CPC)
            
    ###
    ## create a graph (L2-L1-L2) where everithing fits L2
    ###
    G1 = Graph("C = A + B", V,decls,C)
    G1.CDP = CPC
    G1.ADP = APC
    G1.BDP = BPC
    
    print(G1)


    ## disjoint partition at memtile level
    D = Scalar(0)*C
    CPT = PartitionMatrix(D,[args.MT,args.NT])
    BPT = PartitionMatrix(B,[args.MT,args.NT])
    APT = PartitionMatrix(A,[args.MT,args.NT])
    
    ## we create the sequence of matrix operation describing the
    ## larger product. this is a classic matrix multiplication where
    ## the focus and granularity is the memtile level
    decls, V = Addition_t(APT, BPT, CPT)


    ###
    ## create a graph (L3-L2-L3) where everithing fits L2
    ###
    G2 = Graph("C = A + B", V,decls,C)
    G2.CDP = CPT
    G2.ADP = APT
    G2.BDP = BPT
    print(G2)

    pdb.set_trace()
    G1.compute()
    print(G1.temp_result.matrix[0])

    G2.compute()
    print(G2.temp_result.matrix[0])



    ## Now we build an operation LOOP that is a composition of graphs
    ## each graph is a L3-L2-L3 that is then represented as a L2-L1-L2
    ## We have the product of the memtile products or the matrix
    ## factorization of the matrix factorization
    
    D1 = Scalar(0)*C
    L  = Addition_2(A, B, D1, [args.MT, args.NT],[args.MC, args.NC])
    print(L)    
#    for v in L.left:
#        print(v)

    L.compute()
    for i in L.temp_result:
        print(i[0])
