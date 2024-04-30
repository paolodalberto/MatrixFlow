

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
## multiplicationfor a sistolic algorithm. 
##
## here is the vector*matrix approach
##
###



def Product(A : PartitionMatrix, B : PartitionMatrix, C: PartitionMatrix) -> list:

    AD = Data.data_factory_flat(Data.data_factory('ADP', A)) 
    for i in AD: i.inputs = True
    BD = Data.data_factory_flat(Data.data_factory('BDP', B))
    for i in BD: i.inputs = True
    
    CD = Data.data_factory_flat(Data.data_factory('CDP', C)) 
    for i in CD: i.outputs = True



    Row = len(C.l)    # of the output partition
    Col = len(C.l[0])    # of the output partition
    K   = len(B.l)
    ACol = len(A.l[0])

    print(Row,Col,ACol,K)
    decls = [      AD ,        BD,          CD ] 
    
    ###
    ## Computation as a sequence of assignment statements
    ## and binary operations. 
    ###
    V = []
    for i in range(Row):
        for j in range(Col):
            T = Operation("p0", "*", AD[i*ACol] ,BD[j])
            for k in range(1,K):
                T1 = Operation("p%d"%k, "*", AD[i*ACol+k],BD[k*Col+j])
                T = Operation('c','+',T,T1)
            R = Operation("s0", '<<',
                          CD[i*Col+j],
                          T
            )
            #print(R)
            R.parallel = True
            V.append(R)
            
    ###
    ## create a graph
    ###
    
    return decls, V


def Product_2(A : Matrix, B: Matrix, C: Matrix,
              MT : list,
              CT : list
    ) -> list:

    M,N,K = MT



    ## partition at memtile 
    CPT = PartitionMatrix(C,[M,N])
    BPT = PartitionMatrix(B,[K,N])
    APT = PartitionMatrix(A,[M,K])

    
    AD = Data.data_factory_flat(Data.data_factory('ADP', APT)) 
    for i in AD: i.inputs = True
    BD = Data.data_factory_flat(Data.data_factory('BDP', BPT))
    for i in BD: i.inputs = True
    
    CD = Data.data_factory_flat(Data.data_factory('CDP', CPT)) 
    for i in CD: i.outputs = True

    
    Row = len(CPT.l)    # of the output partition
    Col = len(CPT.l[0])    # of the output partition
    K1   = len(BPT.l)
    ACol = len(APT.l[0])
    #Cs = CPT.value()
    #Row = len(Cs)    # of the output partition
    #Col = len(Cs[0]) # as well 
    #K1 = len(BPT.value())
    #print(Row,Col,K)


    M,N,K = CT
    
    ###
    ## Computation as a sequence of assignment statements
    ## and binary operations. 
    ###
    V = []
    for i in range(Row):
        for j in range(Col):
            for k in range(K1):
                CPC = PartitionMatrix(CPT.value()[i][j],[M,N])
                BPC = PartitionMatrix(BPT.value()[k][j],[K,N])
                APC = PartitionMatrix(APT.value()[i][k],[M,K])
                decl, vs = Product(APC,BPC,CPC)
                gr = Graph("C = alpha*A*B", vs,decl,C)
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
        [list(range(Row)),list(range(Col)), list(range(K1))],
        [APT, BPT, CPT]
    )

    
    return  L

if __name__ == "__main__":



    print(" Fast Matrix Multiplication MxK * KxN -> MxN ")  
    print(" Introduce --M --K --N")
    print(" We show case the rest")
        
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--M", help="Factor of A MxK", type=int, default=2048)
    parser.add_argument("-n","--N", help="Factor of B KxN", type=int, default=2048)
    parser.add_argument("-k","--K", help="Factor of A MxK", type=int, default=2048)
    parser.add_argument("-mc","--MC", help="Factor of A @ Core", type=int, default=8)
    parser.add_argument("-nc","--NC", help="Factor of B @ Core", type=int, default=64)
    parser.add_argument("-kc","--KC", help="Factor of A @ Core", type=int, default=512)
    parser.add_argument("-mt","--MT", help="Factor of A @ Core", type=int, default=8)
    parser.add_argument("-nt","--NT", help="Factor of B @ Core", type=int, default=64*4)
    parser.add_argument("-kt","--KT", help="Factor of A @ Core", type=int, default=256*4)
    parser.add_argument("-e", "--error", help="pretty display error", type =str, default =None)
    args = parser.parse_args()

    
    import sys 
       

    A = gen_matrix(args.M,args.K,args.error)
    B = gen_matrix(args.K,args.N,args.error)
    C = gen_matrix(args.M,args.N,args.error)
    

    ## disjoint partition of input output at core level
    
    CPC = PartitionMatrix(C,[args.MC,args.NC])
    BPC = PartitionMatrix(B,[args.KC,args.NC])
    APC = PartitionMatrix(A,[args.MC,args.KC])




    decls, V = Product(APC, BPC, CPC)
            
    ###
    ## create a graph
    ###
    G1 = Graph("C = alpha*A*B", V,decls,C)
    G1.CDP = CPC
    G1.ADP = APC
    G1.BDP = BPC
    
    print(G1)


    D = Scalar(0)*C
    CPT = PartitionMatrix(D,[args.MT,args.NT])
    BPT = PartitionMatrix(B,[args.KT,args.NT])
    APT = PartitionMatrix(A,[args.MT,args.KT])
    print(CPT)
    print(BPT)
    print(APT)

    pdb.set_trace()
    
    decls, V = Product(APT, BPT, CPT)
    G2 = Graph("C = alpha*A*B", V,decls,C)
    G2.CDP = CPT
    G2.ADP = APT
    G2.BDP = BPT
    print(G2)

    pdb.set_trace()
    G1.compute()
    print(G1.temp_result.matrix)

    G2.compute()
    print(G1.temp_result.matrix)

    D1 = Scalar(0)*C
    L  = Product_2(A, B, D1, [args.MT, args.NT, args.KT],[args.MC, args.NC, args.KC])
    print(L)    
#    for v in L.left:
#        print(v)



    L.compute()
    for i in L.temp_result:
        print(i)
