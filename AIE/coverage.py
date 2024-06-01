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



def partition(
        A : list, ## list of size, count
        P : list,  ## list of indexes of A
):
    #print("P", P)
    def padding(a, p):
        pad = 0
        #print(a,p)
        for i in range(len(a[0])):
            pad += p[0][i]-a[0][i]
        return pad*a[1]

    def cost(A, s,M):
        lpad = 0
        rpad = 0
        
        for i in range(s,M):
            #print(len(A), M, A[i])
            if A[i][-1]!=0 :
                A[i][-1] = padding(A[i],A[M])

        A[M][-1] = 0
        for i in range(M+1,len(A)):
            m = True
            #print(A[M],len(A[M]),len(A[M][0]))
            for j in range(len(A[M][0])):
                if A[i][0][j]%A[M][0][j]!=0:
                    m = False
            if m :
                #print("##", i,M)
                #pdb.set_trace()
                A[i][-1] = 0

    ## initialize all costs
    for a in A:
        #print(a)
        if len(a)==2: a.append(-1)
        a[2] = 1000000000

    cost(A, 0,P[0])
    for p in range(1,len(P)):
        #print(P[p-1],P[p])
        #pdb.set_trace()
        cost(A, P[p-1],P[p])

    cost =0
    for a in A:
        cost+=a[-1]
        
            
    return cost
    
    
        
if __name__ == "__main__":
    

    A = [
        [ [ 85 ], 12 ],
        [ [ 86 ], 12 ],
        [ [ 93 ], 12 ],
        [ [ 101], 12 ],
        [ [ 103], 12 ],
        [ [ 105], 12 ],
        [ [ 113], 12 ],
        [ [ 114], 24 ],
        [ [ 116], 12 ],
        [ [ 119], 24 ],
        [ [ 120], 12 ],
        [ [ 121], 12 ],
        [ [ 126], 12 ],
        [ [ 127], 24 ],
        [ [ 128], 12 ],
        [ [ 130], 12 ],
        [ [ 131], 12 ],
        [ [ 132], 36 ],
        [ [ 134], 12 ],
        [ [ 137], 24 ],
        [ [ 138], 24 ],
        [ [ 140], 12 ],
        [ [ 142], 12 ],
        [ [ 146], 12 ],
        [ [ 147], 12 ],
        [ [ 149], 12 ],
        [ [ 150], 12 ],
        [ [ 151], 12 ],
        [ [ 152], 12 ],
        [ [ 153], 12 ],
        [ [ 154], 12 ],
        [ [ 158], 12 ],
        [ [ 160], 48 ],
        [ [ 161], 12 ],
        [ [ 163], 12 ],
        [ [ 167], 12 ],
        [ [ 168], 12 ],
        [ [ 169], 12 ],
        [ [ 170], 24 ],
        [ [ 172], 12 ],
        [ [ 173], 12 ],
        [ [ 175], 12 ],
        [ [ 176], 24 ],
        [ [ 179], 12 ],
        [ [ 183], 12 ],
        [ [ 184], 12 ],
        [ [ 187], 24 ],
        [ [ 188], 12 ],
        [ [ 190], 24 ],
        [ [ 200], 12 ],
        [ [ 201], 36 ],
        [ [ 202], 36 ],
        [ [ 206], 12 ],
        [ [ 209], 12 ],
        [ [ 213], 12 ],
        [ [ 215], 24 ],
        [ [ 219], 12 ],
        [ [ 222], 12 ],
        [ [ 224], 12 ],
        [ [ 226], 12 ],
        [ [ 230], 12 ],
        [ [ 233], 12 ],
        [ [ 237], 12 ],
        [ [ 239], 24 ],
        [ [ 255], 12 ],
        [ [ 256], 12 ],
        [ [ 261], 12 ],
        [ [ 264], 12 ],
        [ [ 266], 12 ],
        [ [ 270], 12 ],
        [ [ 273], 12 ],
        [ [ 285], 12 ],
        [ [ 289], 12 ],
        [ [ 305], 12 ],
        [ [ 309], 12 ],
        [ [ 315], 12 ],
        [ [ 319], 12 ],
        [ [ 337], 12 ],
        [ [ 339], 12 ],
        [ [ 373], 12 ],
        [ [ 384], 24 ]
    ]

    import itertools

    print(len(A))
    Q = 10000000000
    T = None
    for P in itertools.combinations([i for i in range(len(A))], 4):
        
        B = partition(A,P)
        if B<Q:
            Q = B
            T = P
    
    print(T,Q)
    B = partition(A,T)
    for a in A:
        print(a)
