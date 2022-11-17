from  Matrices.matrices import Matrix, PartitionMatrix, Vector, Scalar, Algorithm
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

        import pdb; pdb.set_trace()
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

    def naive_distribute_computation_by_output(
            self,
            Os_groups : tuple = None
    ):

        self.graph.data_dependency()
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
            import pdb; pdb.set_trace()
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
            import pdb; pdb.set_trace()
        else:
            count =0
            for v in Os:
                L[count % len(L)].append(v)
                count +=1

        import pdb; pdb.set_trace()
        for i in range(len(L)): # for every output group 
            w = set()
            for v in L[i]: # for every output compute the instructions
                           # set
                w1 =  all_prev_instruction_indexes(uses, v,Is,
                                                   self.graph.adj)
                w = w.union(w1)
            I = sorted(w)
            import pdb; pdb.set_trace()
            print([ str(i) for i in L[i]],I)
            V[i] = [ self.graph.V[i] for i in I ]

        import pdb; pdb.set_trace()
        print("distribute")
        for i in range(len(L)):
            
            g = Graph(self.hw.pes[i].name,V[i])
            print(g)

            self.hw.pes[i].graph  =g  
            g.dependency()
            print(self.fit_pe_memory(g,self.hw.pes[i]))

        print("Compute")
        self.hw.compute()



if __name__ == "__main__":

    X = 3

    A = Matrix(
        numpy.matrix(
            [
                [ i for i in range(X*2)] for j in range(X*2)
            ]
        )
    )

    B = Matrix(
        numpy.matrix(
            [
                [ i for i in range(X*2)] for j in range(X*2)
            ]
        )
    )

    alpha = Scalar(1)
    alphai = Data('alpha', alpha)
    ## Pure Python Interface


    C = alpha*A*B
    print(C.value())
    import pdb; pdb.set_trace()    

    G1 = algorithm_mult_example(C, alpha,A,B,X)
    
    S = Schedule(G1)
    print(S.fit_hw_memory())
    
    #import pdb; pdb.set_trace()    
    S.naive_distribute_computation()

    import pdb; pdb.set_trace()    
    fact =dict(numpy.load('factorizations_r.npz', allow_pickle=True))
    a,b,c = fact['%d,%d,%d' % (X,X,X)]

    AAA = Algorithm(a,b,c)
    import pdb; pdb.set_trace()    
    P =  AAA.partition_by_output(len(S.hw.pes))
                                 
    
    

    G2 = bini_mult_example(C,c, A,a,B,b)
    S2 = Schedule(G2)
    print(S2.fit_hw_memory())
    S2.naive_distribute_computation_by_output(P)
    
