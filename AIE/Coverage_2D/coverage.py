


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
    pad = 0
    for i in range(len(p[0])):
        pad += p[0][i]-a[0][i]
    return pad*a[1]

def partition_2(
        P : list,  ## list of indexes of A that we have HW function
                   ## computation, we just compute the cost of these
        A : list ## list of size, count
):


    cost = 0
    for ai in range(len(A)):
        a = A[ai]
        b = 10000000000
        for pi in P:
            if pi ==ai :
                b = 0
                break

            p = A[pi]
            w = 1
            for s in range(len(p[0])):
                w *= math.ceil(a[0][s]/p[0][s])*p[0][s] - a[0][s]

            if b>w:
                b = w
            if b == 0: break
            
         cost += a[1]*b
            
        
    return [cost,P]




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
