###
## You have a set of experiments in the past, say N samples 
## You have a set today, say M sample.
##
## Can we say any thing if M belongs to N?  do M have the same
## properties of N or not ?
##
##
## https://docs.scipy.org/doc/scipy/reference/stats.html
##
## If you like this subject, check the github
## Software
## https://github.com/paolodalberto/FastCDL/
## Paper
## https://github.com/paolodalberto/FastCDL/blob/master/PDF/seriesPaolo.pdf
###


import numpy
import math
import scipy 




###
##  Classics: The M and the N Sample have the same distribution ?
##  Kolmogorov's is bread and butter. This is mostly a comparison
##  about the "average" but using CDF ... I love CDFs and the
##  multivariate can be reduced easily .. "well easily is subjective"
##
###

def KS_use(
        x :numpy.array = numpy.random.rand(10),
        y :numpy.array = numpy.random.rand(10)
):
    # the H0 assumption is equal distribution
    #
    # the statistics is the difference of the CDF based on the data
    # the p-value is the error we commit if we assume that H0 is
    # false. 
    
    statistic, p_value = scipy.stats.ks_2samp(x, y)

    print("K-S statistic:", statistic)
    print("p-value:", p_value) 

    return statistic, p_value


###
## what if the Reference and the Sample is not a sample in R but a
## sample in R^4 ?  KS compares Cumulative distribution functions and
## the method is sensitive to change in averages.
##
## We worked this problem out in FastCDL
## https://github.com/paolodalberto/FastCDL
## in several ways
##
##
## Here we write a summary approach
###


class Node:
    def __init__(self, value : numpy.array, color: int ):
        self.value = value
        self.color = color
        self.dom_count = 0
        self.dominates = []
        self.dominated = []
        
    def __str__(self):
        return "C %d-%d: %s" % (self.color,self.dom_count , str(self.value))

def dominates(x : Node, y : Node)  -> bool:
    return numpy.greater_equal(x.value,y.value).all()

def dominated(x : Node, y : Node)  -> bool:
    return dominates(y,x)

def distance(x : Node, y : Node):
    
    return numpy.mean((x.value-y.value)*2)


class Graph:
    def __init__(self,
                 V: list = [] ,
                 dominates = dominates,
                 dominated = dominated
                 ):

        self.V = V
        self.noconnection = numpy.finfo(numpy.float32).max
        self.m = None
        self.M = None
        self.adjc = None
        self.dominates = dominates
        self.dominated = dominated
        
        if len(V) > 0 :
            self.construct()

    def __str__(self):
        import pdb; pdb.set_trace()
        return "\n".join([ str(v) for v in self.V])
            
    def next(self, v):
        return v.dominates
    def prev(self, v):
        return v.dominated
    


    
        
    def adj(self) :
        ## A = NxM N vector of dimension M 

        A = self.V

        N,M = len(A), A[0].value.shape[0]
        print(N,M)

        self.m = Node(A[0].value*1,-2)
        self.M = Node(A[0].value*1,-1)

        for i in range(N):
            a = A[i]
            self.m.value = numpy.minimum(self.m.value,a.value)
            self.M.value = numpy.maximum(self.M.value,a.value)
        
        self.m.value = self.m.value-1
        self.M.value = self.M.value+1
        A.append(self.m)
        A.insert(0,self.M)

        N = len(A)
        for i in range(N):
            a = A[i]
            for j in range(i+1,N):
                b = A[j]
                if self.dominates(a, b):
                    if not b in a.dominates: 
                        a.dom_count+= 1 if (a.color == b.color or a.color<0) else 0
                        a.dominates.append(b)
                    if not a in b.dominated:
                        b.dominated.append(a)

        #Top to bottom, direct domination only
        Vs = sorted(A, key= lambda x: -x.dom_count)
        for i in range(N):
            a = Vs[i]
            R = [d for d in a.dominates]
            for j in range(len(a.dominates)):
                b = Vs[j]
                for q in a.dominates:
                    if b in q.dominates:
                        if b in R:           R.remove(b)
                        if a in b.dominated: b.dominated.remove(a)
                        
            a.dominates = sorted(R, key=lambda x: -x.dom_count)
            #a.dom_count=1
           
        
    def construct(self):
        
        self.adj()


        
        
    def bfs(self, start_node):
        """
        Performs a breadth-first search on a graph.
        
        Args:
            graph: A dictionary representing the graph, where keys are nodes and 
                   values are lists of their neighbors.
            start_node: The node to start the search from.
        """
        
        visited = []  # Keep track of visited nodes
        queue = [start_node]  # Use a queue for FIFO behavior
        
        while queue:
            current_node = queue.pop(0)
        
            if current_node not in visited:
                #print(current_node, end=" ")  # Process the current node
                visited.append(current_node)
        
                # Add unvisited neighbors to the queue
                for neighbor in self.next(current_node):
                    if neighbor not in visited:
                        queue.append(neighbor)
        return visited
        
    def dfs(self,
            node   ,
            visited : list =  []):
        if node not in visited:
            #print("node", node)
            visited.append(node)
            for n in self.next(node):
                self.dfs(n, visited)
        return visited

    def order(self):
        print(self)
        return self.bfs(self.M)


if __name__ == '__main__':



    import matplotlib.pyplot as plt


    if True:
        ## Assume we have an history of R sample 
        R  = scipy.stats.gamma(1.2).rvs(100)
        
        ## N is the sample we have for this experiment
        N  = scipy.stats.norm().rvs(30)
        
        ## If they have the same distribution they have a similar distance
        ## by KS and the p-value is really the one telling if we have any
        ## confidence in the error small pvalue error in considering H0
        ## false is small and they are different ... Notice the different
        ## number of sample for each .. the historical data can be stored
        ## as CDF directly. 
        KS_use(R,N)
        
        fig, ax = plt.subplots(1, 1)
        ax.hist(R, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        ax.hist(N, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        plt.show()
        R  = scipy.stats.norm().rvs(100)
        N  = scipy.stats.norm().rvs(10)
        KS_use(R,N)
        
        
        fig, ax = plt.subplots(1, 1)
        ax.hist(R, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        ax.hist(N, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        plt.show()


    if True:
        QQ = 500
        M = 3
        X = []
        Y = []
        for i in range(QQ):
            X.append(Node(scipy.stats.norm().rvs(M), 0))
            Y.append(Node(numpy.random.rand(M)-0/2, 1))

        G = Graph(X+Y)

        print("-------")
        #for v in sorted(G.V, key=lambda x: x.dom_count):
        #    print(v)
    
    
        Vs = G.dfs(G.M)

        print("-------")
        #for v in Vs[0:20]+Vs[20:]:
        #    print(v)
        
        Vs = G.bfs(G.M)
    
        print("-------")
        #for v in Vs[0:20]+Vs[20:]:
        #    print(v)
        

        fig, ax = plt.subplots(1, 1)
        X = numpy.cumsum([ x.dom_count for x in reversed(Vs) if x.color in [1]])
        Y = numpy.cumsum([ x.dom_count for x in reversed(Vs) if x.color in [0]])
        X = X/X[-1]
        Y = Y/Y[-1]


        KS_use(X,Y)
        ax.plot(X, color='red')
        ax.plot(Y)
        plt.show()
