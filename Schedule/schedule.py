
from  Matrices.matrices import Matrix, PartitionMatrix, Vector, Scalar, Algorithm, read_alpha
from  Graph.graph import Graph, Operation, Data, algorithm_mult_example, \
    bini_mult_example, prev_def, all_prev_instruction_indexes
from  Hw.hw import Memory, PE, AbstractHW
import numpy
import math

class Schedule:
    def __init__(
            self,
            graph : Graph,
            hw : AbstractHW = AbstractHW('Doohm')
    ):

        self.graph = graph # this is a sequence of statements
        self.hw    = hw    # the #PE determines the parallelism

        # this data dependency is at statement level.
        self.dep   = graph.dependency()
        self.inputs = None
        

        





    def fit_hw_memory(self,
                      graph : Graph = None ,
                      hw    : AbstractHW = None):

        #import pdb; pdb.set_trace()
        local = graph is None
        
        if local:
            graph = self.graph
        if graph.dep is None:
            graph.dependency()
        
        if hw is None: hw = self.hw
        


        Is = graph.inputs()
        Os = graph.outputs()

        
        ispace = 0
        for i in Is:
            print(i)
            ispace += i.space()
        ospace = 0
        for i in Os:
            print(i)
            ospace += i.space()

        return hw.memory.space() > ispace+ospace

    def fit_pe_memory(self,
                      graph : Graph = None ,
                      pe    : PE = None):

        #import pdb; pdb.set_trace()
        local = graph is None
        
        if local:
            graph = self.graph
        if graph.dep is None:
            graph.dependency()
        
        Is = graph.inputs()
        Os = graph.outputs()

        
        ispace = 0
        for i in Is:
            print("I", i)
            ispace += i.space()
        ospace = 0
        for i in Os:
            print("O", i)
            ospace += i.space()

        return pe.space() > ispace+ospace



    ### We take a graph and a HW description we split the computation
    ### of the graph in a round robin fashion.
    def naive_distribute_computation(self,
                      graph : Graph = None ,
                      hw    : AbstractHW = None):


        
        local = graph is None
        if local:
            graph = self.graph
        if graph.dep is None:
            graph.dependency()
        if hw is None: hw = self.hw

        L = []
        for pe in hw.pes:
            L.append([])

        count =0 
        for v in graph.V:
            L[count % len(L)].append(v)
            count +=1 

        print("distribute")
        for i in range(len(L)):
            
            g = Graph(hw.pes[i].name,L[i])
            print(g)

            hw.pes[i].graph  =g  
            g.dependency()
            print(self.fit_pe_memory(g,pe))

        print("Compute")
        hw.compute()


    ###
    ## if Os_group is None we distribute the outputs a round robin and
    ## then we add the assignments necessary for those outputs. The PE
    ## may recompute stuff
    ##
    def naive_distribute_computation_by_output(
            self,
            Os_groups : tuple = None
    ):

        self.graph.data_dependency()
        #import pdb; pdb.set_trace()
        Os = self.graph.outputs()
        Is = self.graph.inputs()
        
        L = []
        V = []
        for pe in self.hw.pes:
            L.append([])
            V.append([])

        uses = self.dep['uses']

        
        
        #distribute the outputs round robin ?
        if Os_groups :
            one = Os.pop()
            for d in self.graph.declarations:
                if one in d:
                    Os = d
                    break
            #import pdb; pdb.set_trace()
            W = math.ceil(len(Os_groups)/len(self.hw.pes))

            for i in range(len(self.hw.pes)):
                for j in range(W):
                    if i*W+j >= len(Os_groups): break 
                    L[math.ceil(i)].append(Os[Os_groups[i*W+j]])

            print(Os_groups)
            for l in L:
                for i in l:
                    print(i)
                print("\n")
            #import pdb; pdb.set_trace()
        else:
            count =0
            for v in Os:
                L[count % len(L)].append(v)
                count +=1

        #import pdb; pdb.set_trace()
        for i in range(len(L)): # for every output group 
            w = set()
            for v in L[i]: # for every output compute the instructions
                           # set
                w1 =  all_prev_instruction_indexes(uses, v,Is,
                                                   self.graph.adj)
                w = w.union(w1)
            I = sorted(w)
            #import pdb; pdb.set_trace()
            print([ str(i) for i in L[i]],I)
            V[i] = [ self.graph.V[i] for i in I ]

        #import pdb; pdb.set_trace()
        print("distribute")
        for i in range(len(L)):
            
            g = Graph(self.hw.pes[i].name,V[i])
            print(g)

            self.hw.pes[i].graph  =g  
            g.dependency()
            print(self.fit_pe_memory(g,self.hw.pes[i]))

        print("Compute")
        self.hw.compute()

        print("Count")
        print(self.hw.count())


if __name__ == "__main__":

    import time
    import gc
    X = 3
    Y = 4000

    A = Matrix(
        numpy.matrix(
            [
                [ 1/(1+i) for i in range(X*Y)] for j in range(X*Y)
            ]
        )
    )

    B = Matrix(
        numpy.matrix(
            [
                [ 1/(2+i) for i in range(X*Y)] for j in range(X*Y)
            ]
        )
    )

    alpha = Scalar(1)
    alphai = Data('alpha', alpha)

    ## Pure Python Interface
    print("compute")
    start = time.time()
    C = alpha*A*B
    end = time.time()
    print(end - start)
    #print(C.value())
    #import pdb; pdb.set_trace()    

    ## C_ij = sum_k A_ik B_kj
    G1 = algorithm_mult_example(C, alpha,A,B,X)

    del G1
    gc.collect()
    #S = Schedule(G1)
    #print(S.fit_hw_memory())
    #S.naive_distribute_computation()

    ## Bilinear using the deepmind format C^t = A*B
    #import pdb; pdb.set_trace()    
    fact =dict(numpy.load('factorizations_r.npz', allow_pickle=True))
    a,b,c = fact['%d,%d,%d' % (X,X,X)]
    
    
    ### We take the tensor a, b, and C, for the only porpose to split
    ### the computation by the output by many PEs.  The gamma is
    ### actually the schedule of the computation of the components of
    ### C. We compute the minimum of the maximum number of computation
    ### for partition. 
    if False:
        AAA = Algorithm(a,b,c)
        P =  AAA.partition_by_output(len(S.hw.pes))
    #import pdb; pdb.set_trace()    
                                 
    
    ###
    ## Every matrix has a tensor C matrix and c tensor
    ## We create a computation
    ## P_j = Add(A,a[j])*Add(B,b[j])
    ## then we create C_i = Add(Ps, c[i])
    ##
    D = Matrix(C.value()*0)
    
    G1 = bini_mult_example(D,c, A,a,B,b)
    del G1

    gc.collect()
    
    #S2 = Schedule(G2)
    #print(S2.fit_hw_memory())
    #S2.naive_distribute_computation_by_output(P)
    
    
    a1,b1,c1 = read_alpha('s3x3x3_23.Fawzi_b.bini.txt', numpy.float)

    if False:
        AAA1 = Algorithm(a1,b1,c1)
        P1 =  AAA1.partition_by_output(len(S.hw.pes))
    #import pdb; pdb.set_trace()    
                                 
    
    ###
    ## Every matrix has a tensor C matrix and c tensor
    ## We create a computation
    ## P_j = Add(A,a[j])*Add(B,b[j])
    ## then we create C_i = Add(Ps, c[i])
    ##
    D = Matrix(C.value()*0)
    
    G1 = bini_mult_example(D,c1, A,a1,B,b1,False)
    import pdb; pdb.set_trace()    

    
