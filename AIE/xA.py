

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
from  AIE.conn import Level, Tiling, Traversal,MemoryHierarchTensors
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
    ACol = len(A.l[0])
    K   = min(len(B.l),ACol)

    
    
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

## We create a partitions of the matrices and then create a product of
## the partitions. This is a two level L3-> L2 -> L1 Tiling
def Product_3(
        A : Matrix, B: Matrix, C: Matrix,
        MT : list,  ## L3-L2  tiling
        CT : list,  ## L2-L1  tiling 
        ROWS : int = 4 , COLS : int = 4
    ) -> list:

    # we assume the matrix layout to me row major (all of them)

    ## this is the problem size in DDR: this is the buffer size in DDR
    M = C.matrix.shape[0]; N = C.matrix.shape[1]; K = A.matrix.shape[1]
    

    
    # This is the computation as CORE partition, the minimum
    # granularity
    Mc,Nc,Kc = CT  

    ## number of matrix columns of B and C for a single AIE column
    NC = math.ceil(N/Nc/COLS)*Nc
    ## number of matrix columns of A and rows of  B for a single AIE row
    KR = math.ceil(K/Kc/ROWS)*Kc
    ## number of matrix rows of A and C for a single AIE row
    MR = math.ceil(M/Mc/ROWS)*Mc


    # This is the computation as CORE partition, the minimum
    # granularity
    Mc,Nc,Kc = CT  
    
    NNc = NC//Nc;  MRc = MR//Mc;    KRc = KR//Kc

    if NC % Nc != 0 or KR % Kc !=0 or MR % Mc !=0 :
        # the problem has to be divisable by the core tile (other wise
        # padding from the start or padding the L2 we skip this at
        # this time )
        pdb.set_trace()


    ############################################################################################################
    ## SPLIT BY COLUMN: SPATIAL NC (paralle computation) 
    ## Split by row   : TIME    MR
    ##
    ## A will be a big matrix because each column will need it all
    ## if NC < N, we split B and C by columns and thus we have OFFSET by column (row major) SPATIAL 
    ## if MR < M, we split A and B by columns and thus we have OFFSET by row    (row major) TIME = loop
    ############################################################################################################
    ## this is a logical and physical division by AIE columns
    DDRCPC = PartitionMatrix(C,[MR,NC])
    DDRBPC = PartitionMatrix(B,[K,NC])

    DDRAPC_ = PartitionMatrix(A,[M,KR])  
    DDRAPC = PartitionMatrix(A,[MR,K])  ## this should be by row but
                                        ## it is actually distributed,
                                        ## differently because A goes
                                        ## by ROW


    # SPATIAL = OFFSET
    # TIME    = TRAVERSAL and Tiling   
                                        
    # Column wise computation: this specify the offset and as a
    # function of channels for matrix C and B
    decl, vs = Product_t(DDRAPC,DDRBPC,DDRCPC)
    grt = Graph("DDR column wise C = A*B", vs,decl,C,ADP=DDRAPC,BDP=DDRBPC,CDP=DDRCPC)

    print("Columns Graph")
    print(grt)
    pdb.set_trace()


    ############################################################################################################
    ## L2 Tiling
    ## L2 tile is actually TIME split in general
    ##
    ## The tile sizes are based on all columns but they must be true
    ## for every column! The matrix A will appear to be split at this
    ## point because it is across different columns 
    ## 
    ############################################################################################################
    ## this is the blocked computation if we use L2 tiling this should
    ## be a multiple of the core tiling.

    Mt,Nt,Kt = MT  ## Every thing will be organized as L2 tiles

    ## number of matrix columns of B and C for a single AIE column
    NtC = math.ceil(Nt/Nc/COLS)*Nc
    ## number of matrix columns of A all rows and rows of  B for a single AIE column
    KtR = math.ceil(Kt/Kc)*Kc
    ## number of matrix rows of A and C for a single AIE row
    MtR = math.ceil(Mt/Mc/ROWS)*Mc


    NNt = NtC//Nc
    MRt = MtR//Mc
    KRt = KtR//Kc

    if NtC % Nc != 0 or KtR % Kc !=0 or MtR % Mc !=0 :
        ## as above L2 has to have Core granularity

        pdb.set_trace()

    ## We partition the Column computation using the L2 tile size
    ## Spatial by colum (Columns do the Ntc
    ## Time by Row      (MtR)
    ##
    ## The traversal has to keep up the correct order thus the looping
    ## Doble buffering of any operand at L2 will affect the tiling
    ## size so this should be known by now but the DB has to be
    ## flagged so that we can build the 
    L2CPC = PartitionMatrix(DDRCPC.value()[0][0],[MtR,NtC])   
    L2BPC = PartitionMatrix(DDRBPC.value()[0][0],[KtR,NtC])
    L2APC  = PartitionMatrix(DDRAPC.value()[0][0],[MtR,KtR])

    ## This will help in finding the offset by columns (aka memtile
    ## rows) because teh computation will be split accordingly
    L2APC_ = PartitionMatrix(DDRAPC_.value()[0][0],[MtR,KtR//ROWS])

    
    
    ## We have K tiling (which is time reduce or cascade reduce) the
    ## reduction has to be by core and we have ROWs of them
    ## 

        
    #pdb.set_trace()
    print(L2APC,L2BPC,L2CPC)
    decl, vs = Product_t(L2APC,L2BPC,L2CPC)
    gl2 = Graph("L2 column wise C = A*B", vs,decl,C,ADP=L2APC,BDP=L2BPC,CDP=L2CPC)
    print("Column Graph using L2 tiles")
    print(gl2)
    pdb.set_trace()


    
    ## We take the column computation and split into cores
    L1CPC = PartitionMatrix(DDRCPC.value()[0][0],[Mc,Nc])
    L1BPC = PartitionMatrix(DDRBPC.value()[0][0],[Kc,Nc])
    L1APC = PartitionMatrix(DDRAPC.value()[0][0],[Mc,Kc])

    
    # Column wise computation using CORE tiles 
    #pdb.set_trace()
    print(L1APC,L1BPC,L1CPC)
    decl, vs = Product_t(L1APC,L1BPC,L1CPC)
    gl1 = Graph("L2 column wise C = A*B", vs,decl,C,ADP=L1APC,BDP=L1BPC,CDP=L1CPC)
    

    RR,CC = L2CPC.shape()

    gl1.Factotum['redution'] = "local"
    gl2.Factotum['redution'] = "local"

    if RR==1:
        gl1.Factotum['redution'] = "local reduction and then cascade hardware"
        gl2.Factotum['redution'] = "local reduction and then cascade hardware"


    print("column graph using l1 tiles")
    print(gl1)
    pdb.set_trace()


    #####################################################################################################
    ## L1 Tiling
    ## 
    ## We take a tile in L2 for one column.
    ##
    ## Matrix A partitions will be a broad cast by row
    ## Matrix B is a spatial split by core and thus there will be an offset
    ##
    ## This is the curious case of K tiling by cascade or
    ## local. Choosing (L2CPC.value()[0][0]) set constraints on N and
    ## M but it does not put ny constraints on K we need the L2
    ## partition of B (by column) to have a bound of K
    
    #####################################################################################################

    ## Each mem tile is a time split we can compute each separately
    ## notice that A here does not change because Mc = M but it should
    ## in general
    L1CPCfromL2 = PartitionMatrix(L2CPC.value()[0][0] ,[Mc,Nc]) ## this is one L2 tile of C
    L1BPCfromL2 = PartitionMatrix(L2BPC.value()[0][0],[Kc,Nc]) ## we need more tiles but contained in L2
    L1APCfromL2 = PartitionMatrix(DDRAPC.value()[0][0],[Mc,Kc]) 

    
    # the computation now can be parallel in C (rows) because we write
    # the columns of C in row major and thus we must finish the layout
    # or by K for example we have two C_ij but 16 A_ik pdb.set_trace()
    # in this particular case we do time split in C columns and reduce 
    print(L1APCfromL2,L1BPCfromL2,L1CPCfromL2)
    decl, vs = Product_t(L1APCfromL2,L1BPCfromL2,L1CPCfromL2)
    gl1_2 = Graph("L1 column wise C = A*B from L2 Tile", vs,decl,C,ADP=L1APCfromL2,BDP=L1BPCfromL2,CDP=L1CPCfromL2)
    

    print("column graph using l1 tiles for a L2 tile ")
    print(gl1_2)
    pdb.set_trace()



    
    ## croshet 
    

    LL3 = [ Level("DDR %d" % (i),3,parts =1)           for i in range(COLS)]
    LL2 = [ Level("L2 %d" % (i),2, 512*10243,parts =1) for i in range(COLS)]
    LL1 = [ [Level("L1 %d %d " % (i,j),1,16*1024,2)    for j in range(COLS)] for i in range(ROWS) ] 


    ## we move data from DDR L2 (there are channels and there are columns)
    ## A -> DDRAPC -> L2APC ( row major layouot but the shape is row,column)
    
    ABuffer = [ i for i in reversed(A.shape()[0]) ] 
    ATile   = [ i for i in reversed(L2APC_.logicalshape)]
    NTiles  = [ i for i in reversed(L2APC_.shape())] 
    A_DDRL2 = [
        Tiling(
            ABuffer,            # The original buffer is the original matrix 
            ATile,              # What we transfer is the L2 tiling
            [KR*j,0 ] ,   # This is a spatial partitioning 
            [
                Traversal(
                    i,
                    ATile[i],
                    NTiles[i]
                ) for i in range(2)
            ],
            -1,1
        ) for j in range(ROWS)
    ]

    print("A Tiling L3 L2")
    for d in A_DDRL2: print(d)
    pdb.set_trace()




    
    BBuffer = [ i for i in reversed(B.shape()[0]) ] 
    BTile   = [ i for i in reversed(L2BPC.logicalshape)]
    NBTiles  = [ i for i in reversed(L2BPC.shape())] 
    B_DDRL2 = [
        Tiling(
            BBuffer,
            BTile,
            [NC*j,0 ] ,
            [
                Traversal(
                    i,
                    BTile[i],
                    NBTiles[i]
                ) for i in range(1,-1,-1)
            ],
            -1,1
        ) for j in range(COLS)
    ]

    print("B Tiling L3 L2")
    for d in B_DDRL2: print(d)
    pdb.set_trace()
    C_DDRL2 = [
        Tiling(
            C.shape()[0],
            L2CPC.logicalshape,
            [0,    DDRCPC.logicalshape[1]**j] ,
            [
                Traversal(
                    i,
                    L2CPC.logicalshape[i],
                    L2CPC.shape()[i]
                ) for i in range(2)
            ],
            -1,
            1
        ) for j in range(COLS)
    ]
    
    print("C Tiling L3 L2")
    for d in C_DDRL2: print(d)
    pdb.set_trace()
    
    print("Main Proble", A,B,C)
    print("Column Problem", DDRAPC,DDRBPC,DDRCPC)
    print("Column Problem L2 Tiles", L2APC,L2BPC,L2CPC)
    print("Column Problem L1 Tiles from L2 ",L1APCfromL2,L1BPCfromL2,L1CPCfromL2)    
    print("Column Problem L1 Tiles",L1APC,L1BPC,L1CPC)

    
    
    
    

    

    
    


def croshet(
        G : Graph,
        L3: Level,
        L2: Level,
        L1: Level,
        ROWS : int = 4 ,
        COLS : int = 4  
):

    
    
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
    parser.add_argument("-nt","--NT", help="Factor of B KTxNT @ L2", type=int, default=512)
    parser.add_argument("-kt","--KT", help="Factor of A MTxKT @ L2", type=int, default=2048)
    parser.add_argument("-mc","--MC", help="Factor of A MCxKC @ L1", type=int, default=8)
    parser.add_argument("-nc","--NC", help="Factor of B KCxNC @ L1", type=int, default=64)
    parser.add_argument("-kc","--KC", help="Factor of A MCxKC @ L1", type=int, default=128)
    parser.add_argument("-e", "--error", help="pretty cdisplay error", type =str, default =None)
    args = parser.parse_args()

    
    import sys 
       
    A = gen_matrix(args.M,args.K,args.error)
    B = gen_matrix(args.K,args.N,args.error)
    C = gen_matrix(args.M,args.N,args.error)
    


    ## disjoint partition of input output at core level 
    #CPC = PartitionMatrix(C,[args.MC,args.NC])
    #BPC = PartitionMatrix(B,[args.KC,args.NC])
    #APC = PartitionMatrix(A,[args.MC,args.KC])

    ## we create the sequence of matrix operation describing the
    ## larger product. this is a classic matrix multiplication where
    ## the focus and granularity is the core subvolume
    #decls, V = Product_t(APC, BPC, CPC)
            
    ###
    ## create a graph (L2-L1-L2) where everithing fits L2
    ###
    #G1 = Graph("C = alpha*A*B", V,decls,C)
    #G1.CDP = CPC
    #G1.ADP = APC
    #G1.BDP = BPC
    
    #print(G1)
    #pdb.set_trace()

    ## disjoint partition at memtile level
    #D = Scalar(0)*C
    #CPT = PartitionMatrix(D,[args.MT,args.NT])
    #BPT = PartitionMatrix(B,[args.KT,args.NT])
    #APT = PartitionMatrix(A,[args.MT,args.KT])
    
    ## we create the sequence of matrix operation describing the
    ## larger product. this is a classic matrix multiplication where
    ## the focus and granularity is the memtile level
    #decls, V = Product_t(APT, BPT, CPT)


    ###
    ## create a graph (L3-L2-L3) where everithing fits L2
    ###
    #G2 = Graph("C = alpha*A*B", V,decls,C)
    #G2.CDP = CPT
    #G2.ADP = APT
    #G2.BDP = BPT
    #print(G2)
    #pdb.set_trace()
    

    ## Now we build an operation LOOP that is a composition of graphs
    ## each graph is a L3-L2-L3 that is then represented as a L2-L1-L2
    ## We have the product of the memtile products or the matrix
    ## factorization of the matrix factorization
    
    D1 = Scalar(0)*C
    L  = Product_3(
        A, B, D1,
        [args.MT, args.NT, args.KT],  ## L2 tiling 
        [args.MC, args.NC, args.KC]   ## L1 tiling 
    )
    print(L)    
    #    for v in L.left:
    #        print(v)

    #croshet(L,tiling.L3,tiling.L2, tiling.L1)


    #L.compute()

    
    #print(L.temp_result.matrix[0])


