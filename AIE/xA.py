

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
import AIE.conn as tiling  


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
##  partitions as inputs. Of course the partitions describe the
##  computation fully. is it funny enough ?
def Product_t(A : PartitionMatrix, B : PartitionMatrix, C: PartitionMatrix, reduction: str = None ) -> list:

    ## we create Data type for each elements and this is the leaves of
    ## the basic computation
    AD = Data.data_factory_2d('ADP', A);    ADT = Data.data_factory_flat(AD) 
    for i in ADT: i.inputs = True
    BD = Data.data_factory_2d('BDP', B);    BDT = Data.data_factory_flat(BD)
    for i in BDT: i.inputs = True
    CD  = Data.data_factory_2d('CDP', C) ;  CDT = Data.data_factory_flat(CD)
    for i in CDT: i.outputs = True

    Row = len(C.l)    # of the output partition
    Col = len(C.l[0])    # of the output partition
    K   = len(B.l)
    ACol = len(A.l[0])

    
    
    print("product_t", Row,Col,ACol,K, K==ACol)
    decls = [      ADT ,        BDT,          CDT ] 
    
    ###
    ## Computation as a sequence of assignment statements
    ## and binary operations. 
    ###
    V = []
    for i in range(Row):  ## parallem in M
        for j in range(Col):  ## parallel in N
            
            T = Operation("p0", "*", AD[i][0] ,BD[0][j])
            for k in range(1,K):  ## reduction in K
                T1 = Operation("p%d"%k, "*", AD[i][k],BD[k][j])
                T = Operation('c','+',T,T1)

            
            R = Operation("s0", '<<' if reduction is None else '+=',
                          CD[i][j],
                          T
            )
            #print(R)
            R.parallel = True
            V.append(R)
            
    #for v in V:
    #    print(v)
    
    return decls, V

## We create a partitions of the matrices and then create a product of
## the partitions. This is a two level L3-> L2 -> L1 Tiling
def Product_2(
        A : Matrix, B: Matrix, C: Matrix,
        MT : list,  ## L3-L2  tiling
        CT : list,  ## L2-L1  tiling 
              
    ) -> list:

    M,N,K = MT  ## L2/Memtile tile sizes


    ## this describes the decomposition of the L3 matrix into L2 sub
    ## matrix without overlap:  partition at memtile.
    
    CPT = PartitionMatrix(C,[M,N])
    BPT = PartitionMatrix(B,[K,N])
    APT = PartitionMatrix(A,[M,K])

    decl, vs = Product_t(APT,BPT,CPT)
    grt = Graph("Mem Tile C = alpha*A*B", vs,decl,C)
    grt.CDP = CPT
    grt.ADP = APT
    grt.BDP = BPT

    
    ## we create Data type for each elements and this is the leaves of
    ## the basic computation
    AD = Data.data_factory_2d('ADP', APT); ADF = Data.data_factory_flat(AD) 
    for i in ADF: i.inputs = True
    BD = Data.data_factory_2d('BDP', BPT); BDF = Data.data_factory_flat(BD)
    for i in BDF: i.inputs = True
    CD = Data.data_factory_2d('CDP', CPT) 
    CDF = Data.data_factory_flat(CD)
    for i in CDF: i.outputs = True

    
    Row = len(CPT.l)       # of the output partition
    Col = len(CPT.l[0])    # of the output partition
    K1   = len(BPT.l)      # K partition

    decls = [      ADF ,        BDF,          CDF ] 

    M,N,K = CT  ## L1/Core memory tile 
    
    ###
    ## Computation as a sequence of assignment statements
    ## and binary operations  
    ###
    V = []
    for i in range(Row):  ## parallel M 
        for j in range(Col):  ## Parallel N 
            for k in range(K1):   ## Is this a reduction or a parallel computation
                
                ## Partition of a partition ( is it working? )
                CPC = PartitionMatrix(CPT.value()[i][j],[M,N])
                
                APC = PartitionMatrix(APT.value()[i][k],[M,K])
                BPC = PartitionMatrix(BPT.value()[k][j],[K,N])

                ## each memtile computation the original
                ##
                ## CPT[i][j] =  APT[i][k]*BPT[k][j]
                ##
                ## it is split into core partitions and thus we
                ## describe the computation L2-L1-L2. Each operation
                ## in the list is a graph. Declaration and instruction
                ## list
                ##
                #print(k)
                #pdb.set_trace()
                decl, vs = Product_t(APC,BPC,CPC, None if k<1 else '+')
                gr = Graph("C = alpha*A*B", vs,decl,C)


                ## bookkeeping to remember the partition shape and
                ## format I should introduce these into the init
                ## interface
                gr.CDP = CPC
                gr.ADP = APC
                gr.BDP = BPC
                V.append(gr)

    ###
    ## create a graph Loop so we can summarize the computation by a
    ## single loop instead of a sequence of instructions (still is a
    ## sequence of instructions) but when we print it out or we will
    ## generate code it will look pretty
    ###
    pdb.set_trace()
    L = Loop(
        "T",   ## You got to have a name
        V,
        [list(range(Row)),list(range(Col)), list(range(K1))],
        [APT, BPT, CPT] ## book keeping for this parition 
    )

    ## And we make the loop a Graph .... 
    G =  Graph("Tiled C = alpha*A*B", [L],decls,C)
    G.tiled = True
    G.CDP = CPT
    G.ADP = APT
    G.BDP = BPT
    G.Factotum['Tiled'] = grt
    
    
    return G




def croshet(
        G : Graph,
        L3: tiling.Level,
        L2: tiling.Level,
        L1: tiling.Level,
        ROWS : int = 4 ,
        COLS : int = 4  
):

    
    if not G.tiled: return None
    


    
    ## L3 Tiling read tiling
    ## L2 tile is written into L2 from L3
    L3A = L3.copy("A")
    L3B = L3.copy("B")
    L3C = L3.copy("C")

    HA = tiling.MemoryHierarchTensors("A_L3", G.ADP)
    HB = tiling.MemoryHierarchTensors("B_L3", G.BDP)
    HC = tiling.MemoryHierarchTensors("C_L3", G.CDP)

    print(L3A)
    print("A", G.ADP)
    pdb.set_trace()
    ## tiling is physical and thus the dimensions are in reverse order
    #pdb.set_trace()


    ###
    ## We split matrices by columns and by 'memory' columns so we tile
    ## physically by the second physical dimension on a column major
    ## matrix, yeah it is not really straightforward
    ## 
    TAs = HA.read_tiling_by_parts(1,L3A,1);
    for t in TAs: print(t)

    print("B", G.BDP)
    TBs = HB.read_tiling_by_parts(1,L3B,2)
    for t in TBs: print(t)

    print("C", G.CDP)
    TCs = HC.read_tiling_by_parts(1,L3C,2)
    for t in TCs: print(t)


    CL2Partition = G.V[0].left[0].CDP
    AL2Partition = G.V[0].left[0].ADP
    BL2Partition = G.V[0].left[0].BDP
    HAT = tiling.MemoryHierarchTensors("A_L2",AL2Partition)
    HBT = tiling.MemoryHierarchTensors("B_L2",BL2Partition)
    HCT = tiling.MemoryHierarchTensors("C_L2",CL2Partition)


    L2A = L2.copy("A")
    L2B = L2.copy("B")
    L2C = L2.copy("C")

    ## L2 Tiling read tiling 
    print(L2)
    print("A", AL2Partition)
    pdb.set_trace()
    TAs = HAT.read_tiling_by_parts(0,L2A,1,colmajor=False,broadcast=True)
    for t in TAs: print(t)
    pdb.set_trace()
    print("B", BL2Partition)
    TBs = HBT.read_tiling_by_parts(1,L2B,1)
    for t in TBs: print(t)
    pdb.set_trace()
    print("C", CL2Partition)
    TCs = HCT.read_tiling_by_parts(1,L2C,1)
    for t in TCs: print(t)
    pdb.set_trace()
    # cascade reduction or local reduction?


    tc = [len(CL2Partition.l), len(CL2Partition.l[0])]
    ta = [len(AL2Partition.l), len(AL2Partition.l[0])]
    we = [ROWS,COLS]


    pdb.set_trace()
    if tc!=we and ta[1] > 1:
        G.Factotum['reduction'] = 'cascade'
    
    
    
    ## L1 Tiling for write in L1
    L1A = L1.copy("A")
    L1B = L1.copy("B")
    L1C = L1.copy("C")

    PA = PartitionMatrix(AL2Partition.value()[0][0], AL2Partition.logicalshape)
    HAC = tiling.MemoryHierarchTensors("A_L1", PA)
    PB = PartitionMatrix(BL2Partition.value()[0][0], BL2Partition.logicalshape)
    HBC = tiling.MemoryHierarchTensors("B_L1", PB)
    PC = PartitionMatrix(CL2Partition.value()[0][0], CL2Partition.logicalshape)
    HCC = tiling.MemoryHierarchTensors("C_L1", PC)
    

    pdb.set_trace()
    
    
    return G
    





if __name__ == "__main__":




    print(" Introduce --M --K --N and tiling ")
    print(" We show case the rest")
    print(" At this time we assume that you create the decomposition so that everything click into place")
    print(" TilingAIE will provide the tiling information for the code generation.")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--M",  help="Factor of A MxK @ L3", type=int, default=8)
    parser.add_argument("-n", "--N",  help="Factor of B KxN @ L3 ", type=int, default=2048)
    parser.add_argument("-k", "--K",  help="Factor of A MxK @ L3", type=int, default=2048)
    parser.add_argument("-mt","--MT", help="Factor of A MTxKT @ L2", type=int, default=8)
    parser.add_argument("-nt","--NT", help="Factor of B KTxNT @ L2", type=int, default=256)
    parser.add_argument("-kt","--KT", help="Factor of A MTxKT @ L2", type=int, default=2048)
    parser.add_argument("-mc","--MC", help="Factor of A MCxKC @ L1", type=int, default=8)
    parser.add_argument("-nc","--NC", help="Factor of B KCxNC @ L1", type=int, default=64)
    parser.add_argument("-kc","--KC", help="Factor of A MCxKC @ L1", type=int, default=512)
    parser.add_argument("-e", "--error", help="pretty cdisplay error", type =str, default =None)
    args = parser.parse_args()

    
    import sys 
       
    A = gen_matrix(args.M,args.K,args.error)
    B = gen_matrix(args.K,args.N,args.error)
    C = gen_matrix(args.M,args.N,args.error)
    

    ## disjoint partition of input output at core level 
    CPC = PartitionMatrix(C,[args.MC,args.NC])
    BPC = PartitionMatrix(B,[args.KC,args.NC])
    APC = PartitionMatrix(A,[args.MC,args.KC])

    ## we create the sequence of matrix operation describing the
    ## larger product. this is a classic matrix multiplication where
    ## the focus and granularity is the core subvolume
    decls, V = Product_t(APC, BPC, CPC)
            
    ###
    ## create a graph (L2-L1-L2) where everithing fits L2
    ###
    G1 = Graph("C = alpha*A*B", V,decls,C)
    G1.CDP = CPC
    G1.ADP = APC
    G1.BDP = BPC
    
    print(G1)


    ## disjoint partition at memtile level
    D = Scalar(0)*C
    CPT = PartitionMatrix(D,[args.MT,args.NT])
    BPT = PartitionMatrix(B,[args.KT,args.NT])
    APT = PartitionMatrix(A,[args.MT,args.KT])
    
    ## we create the sequence of matrix operation describing the
    ## larger product. this is a classic matrix multiplication where
    ## the focus and granularity is the memtile level
    decls, V = Product_t(APT, BPT, CPT)


    ###
    ## create a graph (L3-L2-L3) where everithing fits L2
    ###
    G2 = Graph("C = alpha*A*B", V,decls,C)
    G2.CDP = CPT
    G2.ADP = APT
    G2.BDP = BPT
    print(G2)

    #    pdb.set_trace()
    H = tiling.MemoryHierarchTensors("A_L3", APT)

    print("Level Memory", tiling.L3)
    print(APT)
    Ts = H.read_tiling_by_parts(0,tiling.L3,1)
    for t in Ts: print(t)
    #pdb.set_trace()
    print(BPT)
    H = tiling.MemoryHierarchTensors("A_L3", BPT)
    Ts = H.read_tiling_by_parts(0,tiling.L3,2)
    for t in Ts: print(t)
    

    
    #pdb.set_trace()
    G1.compute()
    print(G1.temp_result.matrix[0])

    G2.compute()
    print(G2.temp_result.matrix[0])

    


    ## Now we build an operation LOOP that is a composition of graphs
    ## each graph is a L3-L2-L3 that is then represented as a L2-L1-L2
    ## We have the product of the memtile products or the matrix
    ## factorization of the matrix factorization
    
    D1 = Scalar(0)*C
    L  = Product_2(
        A, B, D1,
        [args.MT, args.NT, args.KT],  ## L2 tiling 
        [args.MC, args.NC, args.KC]   ## L1 tiling 
    )
    print(L)    
    #    for v in L.left:
    #        print(v)

    croshet(L,tiling.L3,tiling.L2, tiling.L1)


    L.compute()

    
    print(L.temp_result.matrix[0])


