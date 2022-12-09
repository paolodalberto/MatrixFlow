import numpy
import math
from  Matrices.matrices import Matrix, PartitionMatrix, Vector, Scalar
import networkx as nx
import matplotlib.pyplot as plt
import graphviz
###
## if we build a Abstract Syntax Tree os a sequence of matrix
## operations we have basically binary operatora such as +,* and
## =. The assignment is the only one that has no intermediary valur to
## store because the left will store the result of the right.
### 
        
class Operation:
    def __init__(
            self,
            name : str,
            op: str, 
            Left ,
            Right ,
            Dest = None
    ):
        self.touch = 0
        self.name = name
        self.operation = op
        self.left   = Left  
        self.right  = Right 
        self.result = Dest
        self.temp_result = None # this will be a Matrix/Vector

        self.temp_space = None
        self.operands = None
        
        try:
            if self.left:
                if type(self.left) is list:
                    for i in self.left:
                        i.result   = self
                else:
                    self.left.result   = self
            if self.right: self.right.result = self
        except Exception as e :
            print(e)
            import pdb; pdb.set_trace()
        
    def count(self,
              operation : str = '*',
              operands_type = [Matrix, Matrix]):

        #if self.operation == operation:
        #    print(self.operation)
        #    import pdb; pdb.set_trace()
        count = 0
        
        if self.operation == operation and self.left and self.right and \
           type(self.left.temp_result) == operands_type[0] and \
           type(self.right.temp_result) == operands_type[1]:
            count +=1
        if self.left:
            count += self.left.count(operation,operands_type)
        if self.right:
            count += self.right.count(operation,operands_type)

        #if self.operation == operation:
        #    print(self.operation,count)
            
        return count
        
    def __str__(self):
        L = "" if self.left is None else str(self.left)
        R = "" if self.right is None else str(self.right)
        tmp = self.name+"("+L  +" "+self.operation+" "+ R+")"
        
        return tmp

    
    def space(self):
        if self.temp_result and type(self.temp_result) is list:
            A = sum([ t.space() for t in self.temp_result])
        else:
            A = self.temp_result.space() if self.temp_result else 0

        if self.left and type(self.left) is list:
            L = sum([ t.space() for t in self.left])
        else:
            L = 0 if self.left is None else self.left.space()
        R = 0 if self.right is None else self.right.space()
        try:
            if L>0 and type(self.left) is Operation:
                A = max(A,L)
            if R>0 and type(self.right) is Operation:
                A = max(A,R)
        except Exception as e:
            print(e)
            import pdb;pdb.set_trace()
            
        self.temp_space = A
        return  A

    def compute(self):
        if type(self.left) is list:
            L =  [ i.compute() for i in self.left]
        else:    
            L =  self.left.compute()
        R =  self.right.compute()
        if self.operation == '+':
            self.temp_result = L + R
        elif self.operation == '-':
            self.temp_result = L - R
        elif self.operation == '*':
            self.temp_result = L * R
        elif self.operation == '/':
            self.temp_result = L / R
        elif self.operation == '=':  # =
            #import pdb;pdb.set_trace()
            self.temp_result =  R
            if type(self.left) is list:
                for i in range(len(R)):
                    self.left[i].set_value(R[i].value())
            else: self.left.temp_result.set_value(R.value())
                
        return  self.temp_result

    def next(self):
        return [self.result]
    
    def prev(self):
        return [self.left,self.right]

    ###
    ## classic left right tree visit really only the Data class will
    ## return itself ...
    ###
    
    def dependantOperands(self):
        r = []
        if self.left and type(self.left) is list:
            for left in self.left:
                tr = [ q for q in left.dependantOperands() if q not in r]
                r += tr
        if self.left:
            tr = [ q for q in self.left.dependantOperands() if q not in r]
            r += tr
        if self.right:
            tr = [ q for q in self.right.dependantOperands() if q not in r]
            r += tr

        self.operands = r
        return r

    ###
    ## After all we are trying to write codes for fast algorithms and
    ## thus Sum \alpa_i A_i is the basic computation for them
    ## 
    ### 
    
    def AdditionBini(
            As : list(), ## this is a data partition already
            I  : numpy.ndarray
    ):
        print(I)
        print(I.shape)
        O = None
        for i in range(I.shape[0]):
            if I[i] !=0:
                T = Operation(
                    "p", "*",
                    Data('i',Scalar(I[i])) ,
                    As[i]
                )
                    
                if O is None: O = T
                else:
                    O = Operation("a", "+",
                                  O ,
                                  T
                    ) 
        return O

            

class Function(Operation):
    def __init__(
            self,
            name : str,
            func ,
            Ops  : list
            
    ):
        self.name = name
        self.operation = func
        self.right = None
        self.left   = Ops
        self.result = None
        self.temp_result = None

    def count(self,
              operation : str = '*',
              operands_type = [Matrix, Matrix]):
        return 0
    def __str__(self):
        L = "" if self.left is None else str(self.left)
        R = "" if self.right is None else str(self.right)
        tmp = self.name+"("+L  +" "+str(self.operation)+" "+ R+")"
        
        return tmp
    def compute(self):
        #import pdb; pdb.set_trace()
        inps = [ o.compute().value() for o in self.left]
        outs = self.operation(*inps)
        self.temp_result = [ Matrix(o) for o in outs]
        return self.temp_result

    def dependantOperands(self):
        return self.left



###
## every leaf of the operation tree os a Data node: Matrix, Vector,
## and Scalar
###
        
class Data(Operation):
    def __init__(
            self,
            name : str,
            Dest
    ):
        self.name = name
        self.operation = 'data'
        self.left   = Dest
        self.right  = None
        self.result = None
        self.temp_result = Dest
        self.inputs = False
        self.temps  = False
        self.outputs= False
        
    def count(self,
              operation : str = '*',
              operands_type = [Matrix, Matrix]):
        return 0
    def compute(self):
        return self.temp_result
    def set_value(self, A):
        
        self.temp_result.set_value(A)
    
    def dependantOperands(self):
        return [self]

    ## We have a matrix partition we create a Data flat (row-major
    ## order) partition. Now these beauties will be sprinckled out in
    ## the computation and for data dependency we need just to compare
    ## pointers to class objects.
    def data_factory_flat(A : list):
        return [ i for row in A for i in row ]
    def data_factory(name : str , A = PartitionMatrix):
        AD = []
        Ashape = (len(A.value()), len(A.value()[0]))
        for i in range(Ashape[0]):
            R = [] 
            for j in range(Ashape[1]):
                L = Data('%s_%d,%d' %(name, i,j), A.value()[i][j])
                R.append(L)
            AD.append(R)
        return AD
    def data_factory_transpose(name : str , A = PartitionMatrix):
        AD = []
        Ashape = (len(A.value()[0]), len(A.value()))
        for i in range(Ashape[0]):
            R = [] 
            for j in range(Ashape[1]):
                L = Data('%s_%d,%d' %(name, j,i), A.value()[j][i])
                R.append(L)
            AD.append(R)
        return AD
    
###
## it is embarassing to say but given an array (like a set of
## instructions) I would like to go back and check things. I do not
## like the fact that I have to introduce -1 every where and thus I
## hide it
##
###

def reverse_range(L):
    return range(L-1, -1, -1)

###
## * uses is a list of lhs,
## * x is a Data or leaf computation
## * and inputs are inputs (declarations)
##
## we return the closest index of lhs in the use use list that assign
## a value to x
### 
def prev_def(uses, X, inputs : set):
    if X in inputs:
        return None
    
    for i in reverse_range(len(uses)):
        if X in  uses[i]:
            return i
    return None


###
## We have a leaf operation (Data) and we want to remember all the
## statemts that made possible its  computation
## * uses = past lhs (current not comprised)
## * the current lhs
## * inputs
## * adjiacent matrix of the closest def and use
##   adj[i,j] means that the statement i has a lhs that will be a rhs of stetement j 
##
## We return the set of indices for the statements that compute v lhs 
###
def all_prev_instruction_indexes(
        uses: list,
        v: Data,
        Is : set,
        adj : numpy.matrix
):
    
    # prev uses set indexes
    W = set()
    c = prev_def(uses,v,Is) 
    Q = [c]
    #print(adj)
    #print(type(adj[0,0]))
    #import pdb; pdb.set_trace()
    while len(Q)>0:
        c = Q.pop()
        W.add(c)
        for i in range(c):
            if adj[i,c] != 0 :
                Q.append(i)
    return W

    
###
## A Computation is a sequence of assignments and Partition/D definitions)
##
###
class Graph:
    def __init__( self, name : str,
                  V : list = [] , ## this is an ordered list
                  D : list = None
    ):
        self.name = name
        self.V = V
        #self.E = E

        self._inputs_  = None
        self._outputs_ = None 
        self.dep     = None

        self.declarations = D

        self.visgraph = graphviz.Digraph()


    def count(self,
              operation : str = '*',
              operands_type : list = [Matrix,Matrix]):

        count = 0
        for v in self.V:
            count += v.count(operation,operands_type)

        return count
        
    ## those operands that are on the lhs but then are recomputed
    ## afterwords
    def outputs(self, dep : dict = None):
        # so we can reuse this code for other graph and other stmt
        # sequences

        local = dep is None 
        if local:
            if self.dep is None:
                self.dep = self.dependency()
            dep = self.dep
        if self._outputs_: return  self._outputs_
        
        O = []
        L = len(dep['uses'])
        for ii in range(L) :
            for i in dep['uses'][ii]:
                A = False
                for jj in range(ii+1,L):
                    if i in dep['uses'][jj]:
                        A = True
                        break
                if not A: O.append(i)
        OO = []
        for i in O:
            if i.outputs:
                OO.append(i)
            else:
                print("TEMP?", i)
        if len(OO) != len(O):
            O = OO
        
        if local: self._outputs_ = set(O)


        return set(O)

    ## those data in the rhs that have no predecessor assignements
    def inputs(self, dep : dict = None):
        # so we can reuse this code for other graph and other stmt
        # sequences
        
        local = dep is None 

        if local:
            if self.dep is None:
                self.dep  = self.dependency()
            dep = self.dep
        if self._inputs_: return self._inputs_

        I = []
        
        L = reverse_range(len(dep['uses']))
        for ii in L :
            for i in dep['defs'][ii]:
                A = False
                for jj in reverse_range(ii):
                    if i in dep['uses'][jj]:
                        A = True
                        break
                if not A: I.append(i)


        if local: self._inputs_ = set(I)
        return set(I)

        
    def next(self, N): return N.next()
    def prev(self, N): return N.prev()
    def __str__(self):
        red = ""
        for n in self.V:
            red += str(n)+"\n"
        return red

    ## We execute each statement in the V list in order
    def compute(self, verbose = True):

        for i in self.V:
            #if verbose: print(i)
            A = i.compute()
            if verbose:
                if type(i.left) is list:
                    for ii in range(len(i.left)):
                        print(i.left[ii],"\n",A[ii].value())
                else:
                    print(i.left,"\n", A.value()) 
        
            
    ## given a statement we return the right hand side operands 
    def dependantOperands(self, op : Operation):
        r = []
        if op.operation != '=' : return None
        r += op.right.dependantOperands()
        return r

    ###
    ## for every statement in order 
    ## we compute two lists:
    ## uses : lhs where each entry is a list of one
    ## defs : rhs operands
    ###
    
    def dependency(self,
                   V : list = None # is a sequence of statements and
                                   # they could come from a different
                                   # graph
    ):
        uses = []
        defs = []

        local = V is None
        if local:
            V = self.V
        t = None
        for i in V:
            if i.operation == '=':
                if type(i.left) is list:
                    u = [] 
                    for left in i.left:
                        ut = left.dependantOperands()
                        u += ut
                else:
                        
                    u = i.left.dependantOperands()
                r = i.right.dependantOperands()
                t = i.right.space()
                defs.append(r)
                uses.append(u)
                print("lhs",u, '= rhs', r)
            else:
                u = self
                t = i.space()

                r = i.dependantOperands()
                defs.append(r[0])
                uses.append(u)
                print("lhs",u, ': rhs', r)

        dep =  {'uses' : uses,
                'defs' : defs,
                'temp' : [t]} 
        if local: self.dep = dep
        return dep

    ###
    ## give the list of statements self.V
    ## * inputs Data that are not specifiedby any statement in V
    ## * outputs lhs Data in V with no rhs read/use
    ## * adj matrix adj[i,j] == 1 the lhs of statement i is in the rhs of statement j 
    
    def data_dependency(self):
        Is = self.inputs()
        Os = self.outputs()
        dep = self.dep

        D = [] 
        
        for j in self.declarations:
            if type(j) is list:
                D += j
            
        for i in D:
            self.visgraph.node(i.name)
        
        lhs = dep['uses']
        rhs = dep['defs']

        # the outputs are defined into the V 
        # adj matrix
        AA = [ 0 for i in range(len(self.V)**2) ]
        adj = numpy.matrix(AA, dtype=int)
        adj.resize((len(self.V),  len(self.V)))
        for i in range(len(self.V)): adj[i,i] =1

        for w in lhs[0]:
            for e in rhs[0]:
                if e in D:
                    self.visgraph.edge(e.name, w.name) 

        for i in range(0,len(lhs)):
            past = lhs[0:i]
            d = lhs[i]
            for e in rhs[i]:
                
                v = prev_def(past,e,Is)
                if v is not None:
                    adj[v,i] = 1

                if e in D:
                    for w in lhs[i]:
                        self.visgraph.edge(e.name, w.name) 
        self.adj = adj
        self.visgraph.render()
        #import pdb; pdb.set_trace()

## C = alpha A B

def algorithm_mult_example(
        C : Matrix,
        alpha : Scalar,                   
        A : Matrix, B : Matrix,
        sub = None
):
    if sub:
        c = C.value().shape
        cc = [math.ceil(c[0]/sub), math.ceil(c[1]/sub)]
        sub = cc
    ## disjoint partition of input output
    CP = PartitionMatrix(C,sub)
    
    BP = PartitionMatrix(B,sub)
    AP = PartitionMatrix(A,sub)

    ## shapes
    Cs = CP.value()
    Row = len(Cs)    # of the output partition
    Col = len(Cs[0]) # as well 
    K = len(BP.value())


    ###
    ## data disjoint partition for computation. This is the
    ## declaration of the basic operands the rest will be defined by
    ## the computation ... compute. 
    ###
    alphai = Data('alpha', alpha)
    AD = Data.data_factory('a', AP)
    for j in AD:
        for i in j:
            i.inputs = True
    BD = Data.data_factory('b', BP)
    for j in BD:
        for i in j:
            i.inputs = True
    CD = Data.data_factory('c', CP)
    for j in CD:
        for i in j:
            i.outputs = True
    ## 
    decls = [alphai,
             Data.data_factory_flat(AD) ,
             Data.data_factory_flat(BD),
             Data.data_factory_flat(CD) ] 
    
    ###
    ## Computation as a sequence of assignment statements
    ## and binary operations. 
    ###
    V = []
    for i in range(Row):
        for j in range(Col):
            T = Operation("p0", "*", AD[i][0] ,BD[0][j])
            for k in range(1,K):
                T1 = Operation("p%d"%k, "*", AD[i][k],BD[k][j])
                T = Operation('c','+',T,T1)
            V.append(
                Operation("s0", '=',
                          CD[i][j],
                          Operation('c00', '*', alphai,T)
                )
            )
    
    ###
    ## create a graph
    ###
    G1 = Graph("C = alpha*A*B", V,decls)
    print(G1)

    ###
    ## Compute the graph for validation. Yep we can and we should run
    ## the graph
    ###
    G1.compute()

    ## we create a stmt-by-stm data dependency
    G1.dependency()

    return G1

"""

B is transpose ...

Multiply 2-by-2 matrix A with 2-by-2 matrix B

a11, a12, a21, a22 = A.ravel()
b11, b12, b21, b22 = B.ravel()
h1 = (a21 - a22) * b12
h2 = (a11 + a21 - a22) * (b12 + b21 + b22)
h3 = (a11 - a12 + a21 - a22) * (b21 + b22)
h4 = a12 * b21
h5 = (a11 + a21) * (b11 + b12 + b21 + b22)
h6 = a11 * b11
h7 = a22 * (b12 + b22)
c11 = h4 + h6
c12 = - h2 + h5 - h6 - h7
c21 = - h1 + h2 - h3 - h4
c22 = h1 + h7
C = np.array([c11, c12, c21, c22]).reshape(2, 2).T

A structure
>>> u
array([[ 0,  1,  1,  0,  1,  1,  0], # A0
       [ 0,  0, -1,  1,  0,  0,  0], # A1
       [ 1,  1,  1,  0,  1,  0,  0], # A2
       [-1, -1, -1,  0,  0,  0,  1]],# A3 
       dtype=int32)

B structure
>>> v,
(array([[0, 0, 0, 0, 1, 1, 0],
       [1, 1, 0, 0, 1, 0, 1],
       [0, 1, 1, 1, 1, 0, 0],
       [0, 1, 1, 0, 1, 0, 1]], dtype=int32),)

C structure
>>> w   P1  P2  P3  P4  P5  P6  P7
array([[ 0,  0,  0,  1,  0,  1,  0],
       [ 0, -1,  0,  0,  1, -1, -1],
       [-1,  1, -1, -1,  0,  0,  0],
       [ 1,  0,  0,  0,  0,  0,  1]], dtype=int32)
"""



def bini_mult_example(
        C : Matrix, CT : numpy.ndarray,
        A : Matrix, AT : numpy.ndarray,
        B : Matrix, BT : numpy.ndarray, deepmindformat = True
):

    
    subblocks = int(math.sqrt(CT.shape[0]))
    products  = CT.shape[1]
    
    ## disjoint partition of input output
    CP = PartitionMatrix(
        C,
        tuple (
            [int(math.ceil(i/subblocks)) for i in C.value().shape]
        )
    )
        


    ## disjoint partition of B and A
    BP = PartitionMatrix(
        B,
        tuple (
            [int(math.ceil(i/subblocks)) for i in C.value().shape]
        )
    )
    AP = PartitionMatrix(
        A,
        tuple (
            [int(math.ceil(i/subblocks)) for i in C.value().shape]
        )
    )

    ## shapes
    Cs = CP.value()
    Row = len(Cs)    # of the output partition
    Col = len(Cs[0]) # as well 
    K = len(BP.value())


    ###
    ## data disjoint partition for computation. This is the
    ## declaration of the basic operands the rest will be defined by
    ## the computation ... compute. 
    ###

    # linear description instead of 2d matrix  
    AD = Data.data_factory_flat(Data.data_factory('a', AP)) 
    for i in AD: i.inputs = True
    BD = Data.data_factory_flat(Data.data_factory('b', BP))
    for i in BD: i.inputs = True


    ## deepmind format need a transposition of the C to make it work

    CD = Data.data_factory_flat(
        Data.data_factory('c', CP) if not deepmindformat else  Data.data_factory_transpose('c', CP)
    )

    ## WARNING: There is a problem in transforming a computation into
    ## a DAG, the output is anbiguous unless we give a hint an
    ## assignment within this computation will be used outside of
    ## it. Think a temporary is defined and assigned within the
    ## computation but I cannot say if it is used outside and thus it
    ## is an output. Only who write the algorithms really knows.

    for i in CD: i.outputs = True
    
    for i in CD: print(i)

    ## we create a declaration of the temporary products and we
    ## provide their maximum size ... in practce each product could be
    ## of different size
    Ps = []
    for i in range(products):
        Ps.append(Data("p_%d" % i, Matrix(CP.value()[0][0].value()*0)))
    

    ## A,B,C partitions and Partial products
    decls = [AD , BD, CD, Ps ] 
    
    ###
    ## Computation as a sequence of assignment statements
    ## and binary operations. 
    ###
    V = []

    for c in range(AT.shape[1]):
        O = Operation(
            'ta', '=',
            Ps[c], # temp product 
            Operation(
                'tp_%d' % c, '*',
                Operation.AdditionBini(AD,AT[:,c]), # Sum a_iA_i
                Operation.AdditionBini(BD,BT[:,c])  # Sum a_iA_i
            )
        )
        V.append(O)
        try:
            O.compute()
        except Exception as e:
            print(e)
            print(O)
            import pdb; pdb.set_trace() 

    #import pdb; pdb.set_trace()
    for c in range(CT.shape[0]):
        O = Operation(
            'ta', '=',
            CD[c], # lhs 
            Operation.AdditionBini(Ps,CT[c,:])  # Sum p_iP_i
        )
        V.append(O)
            
                
    ###
    ## create a graph
    ###
    G1 = Graph("C = Fast A*B", V,decls)
    print(G1)

    ###
    ## Compute the graph for validation. Yep we can and we should run
    ## the graph
    ###
    G1.compute()

    ## we create a stmt-by-stm data dependency
    G1.dependency()

    return G1



if __name__ == "__main__":

    

    A = Matrix(
        numpy.matrix(
            [
                [ i for i in range(3)] for j in range(3)
            ]
        )
    )

    B = Matrix(
        numpy.matrix(
            [
                [ i for i in range(3)] for j in range(3)
            ]
        )
    )

    alpha = Scalar(0.3)
    
    ## Pure Python Interface
    C = alpha*A*B
    print(C)



    ## A compiler will parse the instructiona above and create an
    ## graph: each statement is a binary tree

    ## Terminals
    AI = Data('A', A)
    BI = Data('B', B)
    CI = Data('C', C)
    alphai = Data('alpha', alpha)
    print(AI)
    
    ## Non-Terminals and one single statement
    T1 = Operation('t1', '*', AI, BI);
    T2 = Operation('t2', '*', alphai, T1) ; print(T2)
    CT = Operation('ct', '=', CI,T2)     ; print(CT)
    print(CT)
    G = Graph("temp", [CT])
    print(G)
    
    G.compute()
    import pdb
    pdb.set_trace()
    G.dependency()
    pdb.set_trace()



    G1 = algorithm_mult_example(C, alpha,A,B,C)
    r = G1.dep
    
    ispace = 0
    for i in set(r['defs'][0]+r['defs'][1]):
        print(i)
        ispace += i.space()
    ospace = 0
    for i in set(r['uses'][0]+r['uses'][1]):
        print(i)
        ospace += i.space()
    tspace = r['temp'][0] +r['temp'][0]
    
    print("\n I: %d O: %d T:%d \n" % (ispace, ospace, tspace))
    for i in set(r['defs'][2]+r['defs'][3]):
        print(i)

    
    
