import numpy
import math
from  Matrices.matrices import Matrix, PartitionMatrix, Vector, Scalar, read_alpha
import networkx as nx
import matplotlib.pyplot as plt
import graphviz
import time
import seaborn as sns
from  Validation.BiniScheme import  BiniScheme

###
## if we build a Abstract Syntax Tree os a sequence of matrix
## operations we have basically binary operator such as +,* and =. The
## assignment is the only one that has no intermediary value to store
## because the left will store the result of the right.
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

        R = 0 if self.right is None else (
            self.right.space() if not type(self.right) is list else (
                sum([r.space() for r in self.right()])
            )
        )
            
            
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
        if self.operation == '+=':
            self.temp_result = L + R
            self.left.temp_result.set_value(self.temp_result.value())
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
            As : list(),       ## this is a list of Matrices
            I  : numpy.ndarray
    ):
        #print(I)
        #print(I.shape)
        O = None
        #for i in range(I.shape[0]): #numpy.argsort(-I):
        for i in numpy.argsort(-I):
            if I[i] ==0: continue
            elif (I[i] == 1  or I[i] == 1.0 ) :
                T = As[i]
            elif (I[i] == 1 or I[i] == -1 or I[i] == 1.0 or I[i] == -1.0) and O:
                T = As[i]

            else:
                T = Operation(
                    "p", "*",
                    Data('i',Scalar(I[i])) ,
                    As[i]
                )
                    
            if O is None: O = T
            else:
                O = Operation("a", "-" if I[i] == -1 or I[i] ==-1.0 else "+" ,
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



def bini(
        C : Matrix, CT : numpy.ndarray,
        A : Matrix, AT : numpy.ndarray,
        B : Matrix, BT : numpy.ndarray, deepmindformat = True,
        recursion : int = 1
):
    if recursion == 0:
        C = A*B
        return C

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
    AD = AP.flatten()
    BD = BP.flatten()

    ## deepmind format need a transposition of the C to make it work
    if deepmindformat:
        CD = CP.transpose().flatten() 
    else:
        CD = CP.flatten() 

    def _AdditionBini_(
            As : list(),       ## this is a list of Matrices
            I  : numpy.ndarray
    ):
        T = Scalar(0)*As[0]
        for i in range(len(I)):
            
            if I[i] ==0: continue
            T = T + Scalar(I[i])*As[i]
        return T

        
    Ps = []
    for c in range(AT.shape[1]):
        Ps.append(Matrix(CD[0].value()*0))

    ## A,B,C partitions and Partial products
    decls = [AD , BD, CD, Ps ] 
    
    ###
    ## Computation as a sequence of assignment statements
    ## and binary operations. 
    ###
    V = []

    
    recursion -=1
    for c in range(AT.shape[1]):
        
        AA = _AdditionBini_(AD,AT[:,c])
        BB = _AdditionBini_(BD,BT[:,c])
        
        Right = bini(
            Ps[c],CT,
            AA,AT,
            BB,BT,
            deepmindformat, recursion
        ) 
        Ps[c].set_value(Right.value())
                
    #import pdb; pdb.set_trace()
    for c in range(CT.shape[0]):
        T = _AdditionBini_(Ps,CT[c,:])  # Sum p_iP_i
        CD[c].set_value(T.value())

    def _single_output_(
            C : Matrix,
            Cs : list
    ):
        
        for o in Cs:
            Mat  = o.value()
            m = o.min
            M = o.max
            C.value()[m[0]:M[0],m[1]:M[1]] = Mat
            

        return C
    #import pdb; pdb.set_trace()
    #T = _single_output_(C,CD)
    return _single_output_(C,CD)


    




    

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
class Graph(Function):
    def __init__( self, name : str,
                  V : list = [] , ## this is an ordered list
                  D : list = None,
                  O : Matrix = None,
                  I : list  = None
                  
    ):
        Function.__init__(self,name, None, D)
        self.temp_result = O
        self.right = I
        self.name = name
        self.V = V
        #self.E = E

        self._inputs_  = None
        self._outputs_ = None 
        self.dep     = None

        self.declarations = D

        self.visgraph = graphviz.Digraph()


    def space(self):
        return sum([i.space() for i in self.right])   


    def compare_graph(self,B, C:Matrix):
        T = Matrix(C.value()*0)
        Z = Matrix(C.value()*0)
        self.single_output(T)
        B.single_output(Z)
        
        C.set_value( numpy.absolute(T.value()-Z.value()))
        return C

    def heatmap_diff(self, C: Matrix, mul =1.0 ):

        print("Maximum Error", numpy.max(C.value()))
        cax = plt.imshow( C.value()*mul, cmap = 'gray' ,interpolation = 'nearest' )
        plt.colorbar(cax)
        #ax = sns.heatmap( C.value() , linewidth = 0.5 , cmap = 'coolwarm' )
        plt.title( "2-D Heat Map" )
        plt.show()
        

    def single_output(self,C : Matrix):
        
        for o in self.outputs():
            Mat  = o.temp_result.value()
            m = o.left.min
            M = o.left.max
            C.value()[m[0]:M[0],m[1]:M[1]] = Mat
            

        return C
            
        
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
    def compute(self, verbose = False):
        start = time.time()
        for ds in self.declarations:
            if type(ds) is list:
                for d in ds:
                    if d.outputs:
                        d.left.set_value(d.left.value()*0)
            elif ds.outputs:
                ds.left.set_value(ds.left.value()*0)
        
        for i in self.V:
            #if verbose: print(i)
            A = i.compute()
            if verbose:
                if type(i.left) is list:
                    for ii in range(len(i.left)):
                        print(i.left[ii],"\n",A[ii].value())
                else:
                    print(i.left,"\n", A.value()) 
        
        end = time.time()
        print("compute", end - start)
        
        if self.temp_result:
            self.single_output(self.temp_result)
            return self.temp_result 
        return None

    def create_addequal(left : Data,right: Operation):
        return   Operation('ta', '+=',left,  right)


    ## this is a binary tree, where we remove a leaf the result
    ## operation will be a leaf and thus the operation above has to be
    ## modified.
    
    def rm(m : Operation): ## ?

        ##  + 
        ## l  m 
        ups = m.next()[0]

        ## rd = delete
        ## rk = keep
        if  ups.left == m:
            rd = ups.left
            rk = ups.right
        else:
            rk = ups.left
            rd = ups.right
            
        # the ups is the right child of upps
        ##      + 
        ##   ll   +
        ##       l m
        upps = ups.next()[0]


        rd.result = None
        ups.result = None 
        
        
        upps.right = rk
        rk.result = upps
        del ups
        
        

        
    def find_(m : Data, inst : Operation):
        if m == inst.left or m==inst.right:
            op = inst
        else:
            op = find_(m,inst.left)
            if op is None:
                op = find_(m,inst.right)
        return op
    
                
                
            
            
            
            
            
    
    def remove_m_from_inst(inst : Operation,m : Data):

        

        O = Operation(
            'ta', '+=',
            left, # lhs 
            right  # Sum p_iP_i
        )
        return O
    
            

    
    def short_temp(self):
        self.data_dependency()
        dep = self.dep
        lhs = dep['uses']
        rhs = dep['defs']
        
        insts = range(len(self.V))
        import pdb; pdb.set_trace()
        VV = []
        for i in insts:
            VV.append(self.V[i])
            for j in self.adj[i,:]:
                it = V[j]
                
            
            
            

        

    
    ## given a statement we return the right hand side operands 
    def dependantOperands(self):
        r = []
        r += self.right
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
            if i.operation in  ['=', '+=']:
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
                #print("lhs",u, '= rhs', r)
            else:
                u = self
                t = i.space()

                r = i.dependantOperands()
                defs.append(r[0])
                uses.append(u)
                #print("lhs",u, ': rhs', r)

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
        #self.visgraph.view()
        self.visgraph.render()
#        u.render()
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
                          T
                )
            )
    
    ###
    ## create a graph
    ###
    G1 = Graph("C = alpha*A*B", V,decls,C)
    #print(G1)

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
        B : Matrix, BT : numpy.ndarray, deepmindformat = True,
        recursion : int = 1
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
    
    #for i in CD: print(i)

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

    if recursion == 1:
        
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

    else:
        recursion -=1
        for c in range(AT.shape[1]):
            import pdb; pdb.set_trace()
            AA = Data('aa',Matrix(AD[0].left.value()))
            O0 = Operation('ta', '=',AA, Operation.AdditionBini(AD,AT[:,c]))
            
            BB = Data('bb',Matrix(BD[0].left.value()))
            O1 = Operation('tb', '=',BB, Operation.AdditionBini(BD,BT[:,c]))
            # Sum a_iA_i

            O0.compute()
            O1.compute()
            Right = bini_mult_example(
                Ps[c].left,CT,
                AA.left,AT,
                BB.left,BT,
                deepmindformat, recursion
            ) 
            
            O = Operation(
                'ta', '=',
                Ps[c], # temp product 
                Right
            )
            V.append(O0)
            V.append(O1)
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
            'ta', '+=',
            CD[c], # lhs 
            Operation.AdditionBini(Ps,CT[c,:])  # Sum p_iP_i
        )
        V.append(O)
    
                
    ###
    ## create a graph
    ###
    #import pdb; pdb.set_trace()
    G1 = Graph("C = Fast A*B", V,decls,C, [A,B] )
    #print(G1)

    ###
    ## Compute the graph for validation. Yep we can and we should run
    ## the graph
    ###
    print("Compute")
    G1.compute()

    ## we create a stmt-by-stm data dependency
    print("Dependency")
    G1.dependency()

    return G1

def bini_matrices(
        CT : numpy.ndarray,
        AT : numpy.ndarray,
        BT : numpy.ndarray,
        recursion : int = 1
):

    if recursion <= 0 :
        return CT, AT, BT

    import math
    
    recursion -= 1
        
    
    ## row *M+ col, product this is the current shape 
    s = list(AT.shape)
    
    ## row, col, product so the layout of the original matrix is
    ## "logical"
    #import pdb; pdb.set_trace()
    AT=AT.reshape(int(math.sqrt(s[0])),int(math.sqrt(s[0])),s[1])
    BT=BT.reshape(int(math.sqrt(s[0])),int(math.sqrt(s[0])),s[1])
    CT=CT.reshape(int(math.sqrt(s[0])),int(math.sqrt(s[0])),s[1])
    s = list(AT.shape)

    ## creating space for the new Gamma, Beta, and Alpha
    NAT = numpy.zeros(s[0]**4*s[2]**2,dtype = AT.dtype).reshape(( s[0]**2,s[1]**2, s[2]**2))
    NBT = numpy.zeros(s[0]**4*s[2]**2,dtype = BT.dtype).reshape(( s[0]**2,s[1]**2, s[2]**2))
    NCT = numpy.zeros(s[0]**4*s[2]**2,dtype = CT.dtype).reshape(( s[0]**2,s[1]**2, s[2]**2))

    ## these are the macro addition of the super blocks, which becomes
    ## the addition of the smaller blocks
    ## That is, A1-A2 -> A10-A20, A11-A21,
    ##                   A12-A22, A13-A23
    ## This is why I change the layout above and the code is simpler
    ## This is done for every product
    
    #import pdb; pdb.set_trace()
    for i in range(s[0]): ## row
        for ii in range(s[1]): ## col 
            for j in range(s[2]): ## prod
                NAT[i*s[0]:(i+1)*s[0],
                    ii*s[1]:(ii+1)*s[1],
                    j*s[2]:(j+1)*s[2]] = AT[i,ii,j] 

    ## Every sub-block, for example, A10-A20 will be summed using the
    ## coefficients of the algorithm: this is  obtained by
    ## multiplying the coefficients above by the algorithm AT coefficient.

    #import pdb; pdb.set_trace()
    for i in range(s[0]): ## row
        for ii in range(s[1]): ## col 
            for j in range(s[2]): ## prod
                NAT[i*s[0]:(i+1)*s[0],ii*s[1]:(ii+1)*s[1],j*s[2]:(j+1)*s[2]] *= AT

    ## Doing the same thing on NBT
                
    #import pdb; pdb.set_trace()
    for i in range(s[0]): ## row
        for ii in range(s[1]): ## col 
            for j in range(s[2]): ## prod
                NBT[i*s[0]:(i+1)*s[0],ii*s[1]:(ii+1)*s[1],j*s[2]:(j+1)*s[2]] = BT[i,ii,j] 
                
    #import pdb; pdb.set_trace()
    for i in range(s[0]): ## row
        for ii in range(s[1]): ## col 
            for j in range(s[2]): ## prod
                NBT[i*s[0]:(i+1)*s[0],ii*s[1]:(ii+1)*s[1],j*s[2]:(j+1)*s[2]] *= BT
                

    ## Doing the same thing on NCT
                
    #import pdb; pdb.set_trace()
    for i in range(s[0]): ## row
        for ii in range(s[1]): ## col 
            for j in range(s[2]): ## prod
                NCT[i*s[0]:(i+1)*s[0],ii*s[1]:(ii+1)*s[1],j*s[2]:(j+1)*s[2]] = CT[i,ii,j] 
                
    #import pdb; pdb.set_trace()
    for i in range(s[0]): ## row
        for ii in range(s[1]): ## col 
            for j in range(s[2]): ## prod
                NCT[i*s[0]:(i+1)*s[0],ii*s[1]:(ii+1)*s[1],j*s[2]:(j+1)*s[2]] *= CT


    ## now taking NA00, NA01
    ##            NA02, NA03
    ## -->        NA00, NA01, NA02. NA03 as the original  
    #import pdb; pdb.set_trace()
    NAT=NAT.reshape(s[0]**4,s[2]**2)
    NBT=NBT.reshape(s[0]**4,s[2]**2)
    NCT=NCT.reshape(s[0]**4,s[2]**2)

    ## let us do it again ?
    return bini_matrices(NCT,NAT,NBT,recursion)


def bini_matrices_2(
        ## say this is the starting algorithm 
        CT1 : numpy.ndarray, 
        AT1 : numpy.ndarray, 
        BT1 : numpy.ndarray,
        ## this is the way to upscale the algorithm
        ct1 : numpy.ndarray,
        at1 : numpy.ndarray,
        bt1 : numpy.ndarray,
        validate : bool = False
) -> list : # New C, A, B

    import math
    ## I never notices that I need the dimsions 
    def backward_dimensions(C : int, A : int , B: int):
        ## p*q = A dims         C/r *q = A -> q = r*A/C      q = (A/C)*sqrt(C*B/A)
        ## q*r = B dims                       r*rA/C = B  -> r = sqrt(C*B/A)
        ## p*r = C dims  -> p = C/r                          p = C*sqrt(A/C*B)

        r = int(math.sqrt(C*B/A))
        p = C//r
        q = r*A//C
        ## C dims, A dims, B dims
        return (p,r),(p,q), (q,r)

    ## from list to dimensions
    def backward_dimensions_L(L : list):
        M = [l.shape[0] for l in L]
        return backward_dimensions(*M)
    
    
    L = [CT1, AT1, BT1]
    l = [ct1, at1, bt1]
    
    
    ## products 
    SP = AT1.shape[1]
    sp = at1.shape[1] 

    ## get the dimensions of the original products 
    st = backward_dimensions_L(l)
    ST = backward_dimensions_L(L)

    ## reshape ALG 1
    for i  in range(len(st)):
        s = st[i] ## row x col
        l[i] = l[i].reshape(s[0],s[1],sp)

    ## reshape ALG 2
    for i  in range(len(ST)):
        s = ST[i]
        L[i] = L[i].reshape(s[0],s[1],SP)

    ## shape product of products
    R = []
    for i  in range(len(st)):
        sa = st[i] ##  row x col
        sb = ST[i] ## row x col
        R.append(numpy.zeros((sb[0]*sa[0])*(sb[1]*sa[1])*(sp*SP),dtype = AT1.dtype).reshape(( (sa[0]*sb[0]),(sa[1]*sb[1]),( sp*SP))))

    ## Now the product of product is so simple 
    for i in range(len(R)):
        a = l[i];  sa = list(a.shape)
        A = L[i];  sb = list(A.shape)
        r = R[i]
                
        for i in range(sa[0]): ## row
            for ii in range(sa[1]): ## col 
                for j in range(sp): ## prod
                    r[i*sb[0]:(i+1)*sb[0],ii*sb[1]:(ii+1)*sb[1],j*SP:(j+1)*SP] = a[i,ii,j]*A 


    if validate :
        try:
            bs = BiniScheme(True)
            bs.read_ndarray(R[1], R[2], R[0])
            bs.validate()
        except Exception as e :
            print(e)
        

    for i  in range(len(st)):
        sa = st[i] ##  row x col
        sb = ST[i] ## row x col
        R[i] = R[i].reshape((sa[0]*sb[0])*(sa[1]*sb[1]),(sp*SP))

    return R

    
    

if __name__ == "__main__":

    import sys 
    X = 2
    #Y = 16*27
    Y = 16*27

    
    if True:
        A = Matrix(
            numpy.matrix(
                [
                    [ numpy.random.uniform(-1,1) + 1/(1+i) for i in range(X*Y)] for j in range(X*Y)
                ]
            )
        )

        B = Matrix(
            numpy.matrix(
                [
                    [ numpy.random.uniform(-1,1) + 2/(2+i) for i in range(X*Y)] for j in range(X*Y)
                ]
            )
        )
    else:
        A = Matrix(
            numpy.matrix(
                [
                    [ (1+i) for i in range(X*Y)] for j in range(X*Y)
                ]
            )
        )

        B = Matrix(
            numpy.matrix(
                [
                    [ (2+i) for i in range(X*Y)] for j in range(X*Y)
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




    ## Bilinear using the deepmind format C^t = A*B
    #import pdb; pdb.set_trace()    
    fact =dict(numpy.load('factorizations_r.npz', allow_pickle=True))
    a,b,c = fact['%d,%d,%d' % (X,X,X)]
    at,bt,ct = fact['%d,%d,%d' % (3,3,3)]


    R = 2

    print(a.shape)
    D = Scalar(0)*C
    D = bini(D,c,A,a,B,b,recursion=1)
    Graph.heatmap_diff(Graph,Matrix(numpy.abs(C.value()-D.value())))

    a1 = a*1
    b1 = b*1
    c1 = c*1
    for recursion in range(0,R):
        print(recursion)
        c1,a1,b1 = bini_matrices_2(c1,a1,b1,ct,at,bt)
        print(a1.shape)
        D = Scalar(0)*C
        D = bini(D,c1,A,a1,B,b1,recursion=1)
        Graph.heatmap_diff(Graph,Matrix(numpy.abs(C.value()-D.value())))


    
    a1 = at*1
    b1 = bt*1
    c1 = ct*1
    print(a1.shape)
    D = Scalar(0)*C
    D = bini(D,c1,A,a1,B,b1,recursion=1)
    Graph.heatmap_diff(Graph,Matrix(numpy.abs(C.value()-D.value())))

    for recursion in range(0,R):
        print(recursion)
        c1,a1,b1 = bini_matrices_2(c1,a1,b1, c,a,b)
        print(a1.shape)
        D = Scalar(0)*C
        D = bini(D,c1,A,a1,B,b1,recursion=1)
        Graph.heatmap_diff(Graph,Matrix(numpy.abs(C.value()-D.value())))
        


    sys.exit(0)
    for recursion in range(0,R):
        print(recursion)
        c1,a1,b1 = bini_matrices(c,a,b, recursion)
        D = Scalar(0)*C
        D = bini(D,c1,A,a1,B,b1,recursion=1)
        Graph.heatmap_diff(Graph,Matrix(numpy.abs(C.value()-D.value())))
        #import pdb; pdb.set_trace()        
    

        
    for recursion in range(1,R):
        print(recursion)
        D = Scalar(0)*C
        D = bini(D,c,A,a,B,b,recursion=recursion)
        Graph.heatmap_diff(Graph,Matrix(numpy.abs(C.value()-D.value())))
        #import pdb; pdb.set_trace()        


    if False:


        
        a1,b1,c1 = read_alpha('s3x3x3_23.Fawzi_b.bini.txt', numpy.float)
        for recursion in range(1,4):
            print(recursion)
            D = Scalar(0)*C
            D = bini(D,c1,A,a1,B,b1,False,recursion=recursion)
            Graph.heatmap_diff(Graph,Matrix(numpy.abs(C.value()-D.value())))
            #import pdb; pdb.set_trace()        
