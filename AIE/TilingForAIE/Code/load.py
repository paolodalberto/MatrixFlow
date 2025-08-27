#################################
##
## Consider this as an experimental sand box where you can do several
## things: explore the distribution of Q,K,V on real data or create
## your own. We can investigate the effects of scaling or not scaling
## Q and K. We can investigate and isolate the effects of quantization
## and how we quantize (blocks and shapes).
##  
## Here we have basically two basic methods:
##
## 1) quant and dist are used to compare and visualize the phi-3.5
##    data set. We want to see the average distribution and the effect
##    of quantization and its error distributions.
##
## 2) comp: is a Swiss knife where we compare different algorithm and
##    different data format float32 and float16 (+ int8 for the sage
##    mode). We can turn off the normalization of the matrices Q, K
##    and see if there is any improvement. 
##
#################################



import numpy
import matplotlib.pyplot as plt
import code_gemm 
import argparse

## Initialization MHA is the class with all the multi head activation
## algorithms ... the MHA computes a single SM(QK^t)V there is no
## batch or heads 

mha = code_gemm.MHA()

## the breaking of the computation into smaller computations we
## clarify later and in more details.
RT = [128,128,1,1]

## Layers 
L = 32





def analyze(ii, heads = 32, M =128, I = 3072, norm= False, echo = True):

    if echo: print("LAYER", ii)
    E =  numpy.zeros((5,2))
    El =  [ [] for i in range(5) ]
    
    kt = "phi/Phi-3.5-mini/gqo_4_%d_k_dump.txt"
    qt = "phi/Phi-3.5-mini/gqo_4_%d_q_dump.txt"
    vt = "phi/Phi-3.5-mini/gqo_4_%d_v_dump.txt"
    K = numpy.loadtxt(kt % ii ).reshape(M,I)
    Q = numpy.loadtxt(qt % ii).reshape(M, I)
    V = numpy.loadtxt(vt % ii).reshape(M, I)

    #rt = "phi/Phi-3.5-mini/gqo_4_%d_output_dump.txt"
    #R1 = numpy.loadtxt(rt % ii).reshape(M,I)

    
    K = K.transpose()
    #####
    ##
    ## REFERENCE 
    ##
    ## 32 heads ... this is the computational reference each head is
    ## computed in the original precision and using numpy and scipy
    ## only

    def bl( Q : numpy.array, ## operand 
            K : numpy.array, ## operand 
            V : numpy.array):
        return mha.sage_direction_analysis(Q,K,V,
                                           [RT[0], RT[1],1],
                                           KN=False)

    
    H = I//heads
    R = [[],[], [], []] 
    for i in range(heads):
        #print("head",i)
        r = bl(
            Q[:,i*H:(i+1)*H],
            K[i*H:(i+1)*H,:] ,
            V[:,i*H:(i+1)*H]
        )
        #xm = numpy.min(MC)
        #m  = numpy.mean(MC)
        #mx = numpy.max(MC)
        
        #print(r)
        R[0] += r[0]
        R[1] += r[1]
        R[2] += r[2]
        R[3] += r[3]
        
    m,a,M,X = numpy.mean(R[0]),\
        numpy.mean(R[1]),\
        numpy.mean(R[2]),\
        numpy.max(R[3])
    
    
    return (ii,m,a,M,X) 
    
###
## We assume we have a data set for 32 Layers.
##
## ARGS: layer i, we split the matrices into blocks <m,n>
##
## There are H=32 heads overall Q,K,V are of size M=128 x I= 3072 and
## we split I into 32 parts. That is the basic operation Q,K,V is
## actually of size 128x96 32 times ....split I by H in the columns
## space
##
## Phi model has a "rotational computation" but we will show how we
## compare the algorithms
###
def quant(i, m,n,H = 32, M =128, I = 3072, Norm = False):

    E =  numpy.zeros((5,2))
    El =  [ [] for i in range(5) ]
    
    kt = "phi/Phi-3.5-mini/gqo_4_%d_k_dump.txt"
    qt = "phi/Phi-3.5-mini/gqo_4_%d_q_dump.txt"
    vt = "phi/Phi-3.5-mini/gqo_4_%d_v_dump.txt"
    K = numpy.loadtxt(kt % i ).reshape(M,I)
    Q = numpy.loadtxt(qt % i).reshape(M, I)
    V = numpy.loadtxt(vt % i).reshape(M, I)

    rt = "phi/Phi-3.5-mini/gqo_4_%d_output_dump.txt"
    R1 = numpy.loadtxt(rt % i).reshape(M,I)
    fig, axs = plt.subplots(1, 3)


    
    def pl(k, K,H,M,I, title, m, n ):
        #print(K.shape,M,I,m,n) 
        E = numpy.zeros((H,2))

        M = I//H
        T = numpy.zeros((K.shape[0],M))
        for i in range(H):
            #print("i", K.shape,i,  i*M,(i+1)*M)
            P = K[:, i*M:(i+1)*M]
            #print(P.shape)
        
            Kq,Ks,Kz =  code_gemm.MHA.quantize_int8_block(P, [m,n])
            KR =  code_gemm.MHA.de_quantize_float16_block([Kq,Ks,Kz])
            #if i==0: print(Ks.shape)
            fabs = numpy.fabs(P-KR)
            MX = numpy.max(fabs)
            AV = numpy.mean(fabs)
            #print(i, MX,AV)
            T += fabs
            E[i,:] = [MX,AV]
            
        axs[k].imshow(T/H, cmap='hot', interpolation='nearest')
        #fig.colorbar(im1, ax=axs[k]) #
        axs[k].set_title(title)
        #import pdb; pdb.set_trace()
        #print( E[:,0])
        #print(T)
        return [T,E]
    
    if Norm:
        ## These are designed to reduce the range of the Q and K
        ## operands (K is not transpose)
        
        Q = Q/numpy.sqrt(Q.shape[1]) 
        K = K- numpy.mean(K,axis=0)[None,:]
    EQ = pl(0,Q,H,M,I,"Q mean",m,n)
    EK = pl(1,K,H,M,I,"K mean",m,n)
    EV = pl(2,V,H,M,I,"V mean",m,n)
    print(m,n)
    print("Q", numpy.max(EQ[1][:,0]))
    print("K", numpy.max(EK[1][:,0]))
    print("V", numpy.max(EV[1][:,0]))
    
    plt.show()


###
## We take a layer and produce an average heat for the 32 heads Q,K,V
## The average is one way to measure the zebras behavior.  We create
## png plots for each layer
###
    
def dist(i, H = 32, M =128, I = 3072):

    E =  numpy.zeros((5,2))
    El =  [ [] for i in range(5) ]
    
    kt = "phi/Phi-3.5-mini/gqo_4_%d_k_dump.txt"
    qt = "phi/Phi-3.5-mini/gqo_4_%d_q_dump.txt"
    vt = "phi/Phi-3.5-mini/gqo_4_%d_v_dump.txt"
    K = numpy.loadtxt(kt % i ).reshape(M,I)
    Q = numpy.loadtxt(qt % i).reshape(M, I)
    V = numpy.loadtxt(vt % i).reshape(M, I)

    rt = "phi/Phi-3.5-mini/gqo_4_%d_output_dump.txt"
    R1 = numpy.loadtxt(rt % i).reshape(M,I)
    fig, axs = plt.subplots(1, 3)

    def pl(k, K,H,M,I, title ):
    
        i =0
        M = I//H
        KM = K[:, i*M:(i+1)*M]

        for i in range(1,H):
            KM += K[:, i*M:(i+1)*M]
        KM = KM/H

        im1 = axs[k].imshow(KM, cmap='hot', interpolation='nearest')
        fig.colorbar(im1, ax=axs[k]) #
        axs[k].set_title(title)

    pl(0,Q,H,M,I,"Q mean")
    pl(1,K,H,M,I,"K mean")
    pl(2,V,H,M,I,"V mean")
    plt.savefig(("i%d"%i)+".png")
    plt.close()
    #plt.show()


##############
##  We take a layer (i) We read the Q,K,V and shape them as
##  M=128xI=3072 Then we split the computation into 32 independent
##  Q_i, K_i, V_i of size 128x96 each
###
    

def comp(i, H = 32, M =128, I = 3072, norm= False, echo = True):

    if echo: print("LAYER", i)
    E =  numpy.zeros((5,2))
    El =  [ [] for i in range(5) ]
    
    kt = "phi/Phi-3.5-mini/gqo_4_%d_k_dump.txt"
    qt = "phi/Phi-3.5-mini/gqo_4_%d_q_dump.txt"
    vt = "phi/Phi-3.5-mini/gqo_4_%d_v_dump.txt"
    K = numpy.loadtxt(kt % i ).reshape(M,I)
    Q = numpy.loadtxt(qt % i).reshape(M, I)
    V = numpy.loadtxt(vt % i).reshape(M, I)

    rt = "phi/Phi-3.5-mini/gqo_4_%d_output_dump.txt"
    R1 = numpy.loadtxt(rt % i).reshape(M,I)

    
    K = K.transpose()
    #####
    ##
    ## REFERENCE 
    ##
    ## 32 heads ... this is the computational reference each head is
    ## computed in the original precision and using numpy and scipy
    ## only
    R = mha.heads(Q,K,V,  H, mha.shead)   


    G = R1-R
    A = numpy.fabs(G)
    a =numpy.mean(A)
    b =numpy.max(A)
    ## this is to show there is a rotation and extra computation I
    ## will not be able to reproduce ... so I will focus only on the
    ## computation at hand
    #print(i, "RScipy2 Heads L1 %1.3e %1.3e" % (a,b))
    El[0] = list(G.flatten())
    E[0,:] = [ a,b]

                   
    ## same as above but float16
    Q16 = numpy.ndarray.astype(Q,numpy.float16)
    K16 = numpy.ndarray.astype(K,numpy.float16)
    V16 = numpy.ndarray.astype(V,numpy.float16)

    ###
    ##  REFERENCE In float16
    ### 
    One = mha.heads(
        Q16,K16,V16, H,
        mha.shead
    )
#    print(One.dtype)
    
    G = R-One
    A = numpy.fabs(G)
    a =numpy.mean(A)
    b =numpy.max(A)
    print(i,"Scipy L1 %1.3e %1.3e" % (a,b))
    El[1] += list(G.flatten())
    E[1,:] = [ a,b]

    ## This is yet another reference where each computation is
    ## separated and it is the based of any block computation such as
    ## the following operations
    two = mha.heads(
        Q16,K16,V16, H,
        mha.ddr_computation_)
    #print(two.dtype)
    G = R-two
    A = numpy.fabs(G)
    a =numpy.mean(A)
    b =numpy.max(A)
    print(i,"separ L1 %1.3e %1.3e" % (a,b))
    El[2] += list(G.flatten())
    E[2,:] = [ a,b]

    ###
    ## Flash attention or block attention where we split the
    ## computation in blocks RT = [m,n, x,x]
    ## Q is split into mxM parallel blocks
    ## K is split Mxn (and so V)
    ##
    ## You may say this is the reference we want to match for the sage
    ## attention. notice the normalization of Q and K can be applied
    ## also here if beneficial
    
    RT[2] = 1 if norm else 0
    #if echo: print("block")
    def bl( Q : numpy.array, ## operand 
            K : numpy.array, ## operand 
            V : numpy.array):
        return mha.ddr_computation_block(Q,K,V,RT,False)

    three = mha.heads(
        Q16,K16,V16, H,
        bl)

    #print(three.dtype)
    G = R-three
    A = numpy.fabs(G)
    a =numpy.mean(A)
    b =numpy.max(A)
    if echo:print(i,"block L1 %1.3e %1.3e" % (a,b))
    El[3] += list(G.flatten())
    E[3,:] = [ a,b]

    ####
    ## SAGE computation as above but Q*K^t is actually
    ## quant(Q)*quant(K^t) *scale(Q)*scale(K) 
    ##

    ###
    ## please go check the computation mha.sage_computation_block
    ##
    ###
    
    #if echo: print("sage")
    def sagebl( Q : numpy.array, ## operand 
            K : numpy.array, ## operand 
            V : numpy.array):
        return mha.sage_computation_block(Q,K,V,[1,1,1,1],KN=norm)

    four= mha.heads(
        Q16,K16,V16, H,
        sagebl
    )
    #print(four.dtype)
        
    G = R-four
    A = numpy.fabs(G)
    a =numpy.mean(A)
    b =numpy.max(A)
    if echo: print(i, "sage  L1 %1.3e %1.3e" % (a,b))
    El[4] += list(G.flatten())
    E[4,:] = [ a,b]

    ## El is a list of all absolute error per algorithms and E has the
    ## basic statistics/summary
    return [El, E]





if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='A simple program that greets the user.')
 
    parser.add_argument('-a', '--analyze', choices=['true', 'false'], default='false',
                        help='analyze the direction of SM.')
    parser.add_argument('-s', '--sequential', choices=['true', 'false'], default='true',
                        help='sequential or pool.')
    parser.add_argument('-c', '--compare', choices=['true', 'false'], default='false',
                        help='compare algorithms .')
    parser.add_argument('-d', '--distribution', choices=['true', 'false'], default='false',
                        help='distribution of the operands .')
    parser.add_argument('-q', '--quantization', choices=['true', 'false'], default='false',
                        help='how quantization affect the matrix representation .')
    
    args = parser.parse_args()
    
    

    if args.analyze == 'true':
        ## you want to look at the distribution of the layers and heads ?
        if args.sequential=='true':

            results = [ analyze(0) ] # , analyze(17), analyze(31) ]

        if args.sequential!='true':
            from multiprocessing import Pool
            R  = [i for i in range(32)]
            with Pool(processes=16) as pool: # Create a pool with 4 worker processes
                results = pool.map(analyze, R)

        for r in results:
            print("layer %d min %f mean  %f and max %f maxfabs %f" % (r[0], r[1], r[2], r[3], r[4]))
            
    if args.distribution == 'true':
        ## you want to look at the distribution of the layers and heads ?
        R  = [i for i in range(32)]
        for i in R:
            dist(i)


    if args.quantization=='true' :
        # how a layer is affected by quantization, shape of the
        # quantization and normalization
        results = [quant( 0,16,16), quant( 0,16,16,Norm=False),
                   quant(0, 128,1), quant(0, 128,1,Norm=False),
                   quant(17,16,16), quant(17,16,16,Norm=False),
                   quant(17,128,1), quant(17,128,1,Norm=False)
                   ]

    if args.compare == 'true': 
            
        if args.sequential=='true':

            results = [ comp(i) for i in range(32)] # , comp(17), comp(31) ]

        if args.sequential!='true':
            from multiprocessing import Pool
            R  = [i for i in range(32)]
            with Pool(processes=16) as pool: # Create a pool with 4 worker processes
                results = pool.map(comp, R)
                #print(results)
            
        ## collecting the results 
        E =  numpy.zeros((5,len(results),2))
        El =  [ [] for i in range(5) ]
        for i in range(len(results)):
            E[:,i,:] = results[i][1]
        for j in range(5):
            El[j] += results[i][0][j]

        print("Averages L1 errors")
        titles =[ 
            "scipy 64     L1 %1.3e MAX %1.3e" % (numpy.mean(E[0,:,0]),numpy.max(E[0,:,1])),
            "scipy        L1 %1.3e MAX %1.3e" % (numpy.mean(E[1,:,0]),numpy.max(E[1,:,1])),
            "sepa         L1 %1.3e MAX %1.3e" % (numpy.mean(E[2,:,0]),numpy.max(E[2,:,1])),
            "block        L1 %1.3e MAX %1.3e" % (numpy.mean(E[3,:,0]),numpy.max(E[3,:,1])),
            "sage         L1 %1.3e MAX %1.3e" % (numpy.mean(E[4,:,0]),numpy.max(E[4,:,1]))
        ]
        for t in titles:
            print(t)

        #plotting the errors as  we should 
        names =[ 
            "scipy64",
            "scipy",
            "sepa",
            "block",
            "sage"
        ]
            
                
        import matplotlib.pyplot as plt
        i = 0
        for e in El:
            print(titles[i])
            plt.hist(e,1000)
            plt.title(titles[i])
            #plt.yscale('log')
            plt.savefig(names[i]+".png") 
            #plt.show()
            i+=1

        

