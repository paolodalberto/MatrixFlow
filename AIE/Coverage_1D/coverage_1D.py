"""
torch.Size([1,  85, 3072]) #:12
torch.Size([1,  86, 3072]) #:12
torch.Size([1,  93, 3072]) #:12
torch.Size([1, 101, 3072]) #:12
torch.Size([1, 103, 3072]) #:12
torch.Size([1, 105, 3072]) #:12
torch.Size([1, 113, 3072]) #:12
torch.Size([1, 114, 3072]) #:24
torch.Size([1, 116, 3072]) #:12
torch.Size([1, 119, 3072]) #:24
torch.Size([1, 120, 3072]) #:12
torch.Size([1, 121, 3072]) #:12
torch.Size([1, 126, 3072]) #:12
torch.Size([1, 127, 3072]) #:24
torch.Size([1, 128, 3072]) #:12
torch.Size([1, 130, 3072]) #:12
torch.Size([1, 131, 3072]) #:12
torch.Size([1, 132, 3072]) #:36
torch.Size([1, 134, 3072]) #:12
torch.Size([1, 137, 3072]) #:24
torch.Size([1, 138, 3072]) #:24
torch.Size([1, 140, 3072]) #:12
torch.Size([1, 142, 3072]) #:12
torch.Size([1, 146, 3072]) #:12
torch.Size([1, 147, 3072]) #:12
torch.Size([1, 149, 3072]) #:12
torch.Size([1, 150, 3072]) #:12
torch.Size([1, 151, 3072]) #:12
torch.Size([1, 152, 3072]) #:12
torch.Size([1, 153, 3072]) #:12
torch.Size([1, 154, 3072]) #:12
torch.Size([1, 158, 3072]) #:12
torch.Size([1, 160, 3072]) #:48
torch.Size([1, 161, 3072]) #:12
torch.Size([1, 163, 3072]) #:12
torch.Size([1, 167, 3072]) #:12
torch.Size([1, 168, 3072]) #:12
torch.Size([1, 169, 3072]) #:12
torch.Size([1, 170, 3072]) #:24
torch.Size([1, 172, 3072]) #:12
torch.Size([1, 173, 3072]) #:12
torch.Size([1, 175, 3072]) #:12
torch.Size([1, 176, 3072]) #:24
torch.Size([1, 179, 3072]) #:12
torch.Size([1, 183, 3072]) #:12
torch.Size([1, 184, 3072]) #:12
torch.Size([1, 187, 3072]) #:24
torch.Size([1, 188, 3072]) #:12
torch.Size([1, 190, 3072]) #:24
torch.Size([1, 200, 3072]) #:12
torch.Size([1, 201, 3072]) #:36
torch.Size([1, 202, 3072]) #:36
torch.Size([1, 206, 3072]) #:12
torch.Size([1, 209, 3072]) #:12
torch.Size([1, 213, 3072]) #:12
torch.Size([1, 215, 3072]) #:24
torch.Size([1, 219, 3072]) #:12
torch.Size([1, 222, 3072]) #:12
torch.Size([1, 224, 3072]) #:12
torch.Size([1, 226, 3072]) #:12
torch.Size([1, 230, 3072]) #:12
torch.Size([1, 233, 3072]) #:12
torch.Size([1, 237, 3072]) #:12
torch.Size([1, 239, 3072]) #:24
torch.Size([1, 255, 3072]) #:12
torch.Size([1, 256, 3072]) #:12
torch.Size([1, 261, 3072]) #:12
torch.Size([1, 264, 3072]) #:12
torch.Size([1, 266, 3072]) #:12
torch.Size([1, 270, 3072]) #:12
torch.Size([1, 273, 3072]) #:12
torch.Size([1, 285, 3072]) #:12
torch.Size([1, 289, 3072]) #:12
torch.Size([1, 305, 3072]) #:12
torch.Size([1, 309, 3072]) #:12
torch.Size([1, 315, 3072]) #:12
torch.Size([1, 319, 3072]) #:12
torch.Size([1, 337, 3072]) #:12
torch.Size([1, 339, 3072]) #:12
torch.Size([1, 373, 3072]) #:12
torch.Size([1, 384, 3072]) #:24
"""



import math
import  pdb
import copy

def size(x : list) -> int:
    p =1
    for i in x:
        p*=i
    return p
    
## p
def padding(
        a : list, ## function call parameter size [shapes, count, pad] 
        p : list  ## the physical function call we pad 'a' to
):
    pad = p[0]-a[0]
    return pad*a[1]

def cost(
        A : list , ## [shape, count, pads ]
        s : int ,  ## starting index
        M : int    ## physical computation index A[M] is the actuall computation
):        
    
    ## A[s] will dominate the cost so there is no need to go farther
    
    A[M][-1] = 0 ## you cannot do better than zero padding
    
    ## each problem is computed by A[M]
    for i in range(s,M):
        #print(len(A), M, A[i])
        A[i][-1] = min (A[i][-1], padding(A[i],A[M]))

    ## obviously no need padding if it is done in HW
    
    for i in range(M+1,len(A)):
        #print(A[M],len(A[M]),len(A[M][0]))
        a,c,d = A[M]
        m = A[i][0] % a ==0
        a = a*math.ceil(A[i][0]/a)
            
            
        if m :  A[i][-1] = 0 ## multiple of A[M] 
        else:
            #print(A[M])
            G =  padding(A[i],[a,c,d])
            A[i][-1] = min (A[i][-1],G )
            

def partition(
        P : list,  ## list of indexes of A that we have HW function
                   ## computation, we just compute the cost of these
        A : list ## list of size, count
):

    ## initialize all costs
    for a in A:
        #print(a, a)
        if len(a)==2: a.append(-1)
        a[2] = 1000000000 ## large baby

    cost(A, 0,P[0])
    for p in range(1,len(P)):
        #pdb.set_trace()
        cost(A, P[p-1],P[p])

    c =0
    for a in A:
        c+=a[-1]
        
        
    return [c,P]


def partition_2(
        P : list,  ## list of indexes of A that we have HW function
                   ## computation, we just compute the cost of these
        A : list ## list of size, count
):
    N = len(A)
    A = copy.deepcopy(A)

    X = [ A[p][0] for p in P]
    chip = False
    for x in X:
        A.append([x,0,0])
        for i in range(2,10):
            p = x*i
            if p> A[N-1][0]:
                if not chip:
                    cip = True
                    A.append([p,0,0])
                    
                break
            if p in X: continue
            #X.append(p)
            A.append([p,0,0])

    
    A = sorted(A, key= lambda x: (x[0],-x[1]))
    #print(A)
    #import pdb; pdb.set_trace();
    N = len(A)
    c = 0
    for i in range(N):
        if A[i][2] == 0: continue 
        b = N
        for j in range(i+1,N):
            if A[j][2] ==0:
                b = j
                break
        if b<N:
            A[i][2] = A[i][1]*(A[b][0] -A[i][0])
        c+=A[i][2]
    #import pdb; pdb.set_trace();
    return [c,P]




if __name__ == "__main__":
    
    # Size and Count, cost
    A = [
        [ 85 , 12, 1000000000],
        [ 86 , 12, 1000000000 ],
        [ 93 , 12, 1000000000 ],
        [ 101, 12, 1000000000 ],
        [ 103, 12, 1000000000 ],
        [ 105, 12, 1000000000 ],
        [ 113, 12, 1000000000 ],
        [ 114, 24, 1000000000 ],
        [ 116, 12, 1000000000 ],
        [ 119, 24, 1000000000 ],
        [ 120, 12, 1000000000 ],
        [ 121, 12, 1000000000 ],
        [ 126, 12, 1000000000 ],
        [ 127, 24, 1000000000 ],
        [ 128, 12, 1000000000 ],
        [ 130, 12, 1000000000 ],
        [ 131, 12, 1000000000 ],
        [ 132, 36, 1000000000 ],
        [ 134, 12, 1000000000 ],
        [ 137, 24, 1000000000 ],
        [ 138, 24, 1000000000 ],
        [ 140, 12, 1000000000 ],
        [ 142, 12, 1000000000 ],
        [ 146, 12, 1000000000 ],
        [ 147, 12, 1000000000 ],
        [ 149, 12, 1000000000 ],
        [ 150, 12, 1000000000 ],
        [ 151, 12, 1000000000 ],
        [ 152, 12, 1000000000 ],
        [ 153, 12, 1000000000 ],
        [ 154, 12, 1000000000 ],
        [ 158, 12, 1000000000 ],
        [ 160, 48, 1000000000 ],
        [ 161, 12, 1000000000 ],
        [ 163, 12, 1000000000 ],
        [ 167, 12, 1000000000 ],
        [ 168, 12, 1000000000 ],
        [ 169, 12, 1000000000 ],
        [ 170, 24, 1000000000 ],
        [ 172, 12, 1000000000 ],
        [ 173, 12, 1000000000 ],
        [ 175, 12, 1000000000 ],
        [ 176, 24, 1000000000 ],
        [ 179, 12, 1000000000 ],
        [ 183, 12, 1000000000 ],
        [ 184, 12, 1000000000 ],
        [ 187, 24, 1000000000 ],
        [ 188, 12, 1000000000 ],
        [ 190, 24, 1000000000 ],
        [ 200, 12, 1000000000 ],
        [ 201, 36, 1000000000 ],
        [ 202, 36, 1000000000 ],
        [ 206, 12, 1000000000 ],
        [ 209, 12, 1000000000 ],
        [ 213, 12, 1000000000 ],
        [ 215, 24, 1000000000 ],
        [ 219, 12, 1000000000 ],
        [ 222, 12, 1000000000 ],
        [ 224, 12, 1000000000 ],
        [ 226, 12, 1000000000 ],
        [ 230, 12, 1000000000 ],
        [ 233, 12, 1000000000 ],
        [ 237, 12, 1000000000 ],
        [ 239, 24, 1000000000 ],
        [ 255, 12, 1000000000 ],
        [ 256, 12, 1000000000 ],
        [ 261, 12, 1000000000 ],
        [ 264, 12, 1000000000 ],
        [ 266, 12, 1000000000 ],
        [ 270, 12, 1000000000 ],
        [ 273, 12, 1000000000 ],
        [ 285, 12, 1000000000 ],
        [ 289, 12, 1000000000 ],
        [ 305, 12, 1000000000 ],
        [ 309, 12, 1000000000 ],
        [ 315, 12, 1000000000 ],
        [ 319, 12, 1000000000 ],
        [ 337, 12, 1000000000 ],
        [ 339, 12, 1000000000 ],
        [ 373, 12, 1000000000 ],
        [ 384, 24, 1000000000 ]
    ]            
                 
                 
    ## this order is a topological order

    A = sorted(A, key=lambda x: x[0])

    
    def map_(
            P : list,  ## list of indexes of A that we have HW function
                   ## computation, we just compute the cost of these
            A : list = A
    ) -> list :
        return partition_2(P,A)

    import itertools
    from functools import reduce
    from multiprocessing import Pool
    import math

    print(len(A))
    Q = 10000000000
    T = None


    def reduce_(A,B):
        c0, a0 = A
        c1, a1 = B
        return A if c0<c1 else B

    def binomial_coeff(n, k):
        return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

    #P = [3,len(A)//3,len(A)//2,len(A)-1]
    #print("step", step(A,P))
    R = []
    for k in range(7,9) :
        print("K", k,  binomial_coeff(len(A), k))
        #we create all (n over k) ordered P and we choose the fastest


        if False:
            
            for x in itertools.combinations([i for i in range(len(A))], k):
                print(map_(x))
                print(partition(x,A))
        else:
            with Pool(32) as pool:
                result = pool.imap_unordered( map_, itertools.combinations([i for i in range(len(A))], k))
                f_result = reduce(reduce_, result)

                print("Pool",f_result)
            #print(A)
            B = partition(f_result[1],A)
            print(B,f_result)
            R.append(f_result)


    for r in R:
        print(r)
