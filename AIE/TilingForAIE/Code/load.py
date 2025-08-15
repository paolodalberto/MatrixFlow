import numpy
import matplotlib.pyplot as plt
import code_gemm 

mha = code_gemm.MHA()
RT = [1,1,1,1]
L = 32

def quant(i, m,n,H = 32, M =128, I = 3072):

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
        
            Ks,Kq =  code_gemm.MHA.quantize_int8_block(P, [m,n])
            KR =  code_gemm.MHA.de_quantize_float16_block([Ks,Kq])
            if i==0: print(Ks.shape)
            fabs = numpy.fabs(P-KR)
            MX = numpy.max(fabs)
            #if MX>16.0:
            #    import pdb; pdb.set_trace()
            
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
    

    EQ = pl(0,Q/numpy.sqrt(M),H,M,I,"Q mean",m,n)
    print("Q", numpy.max(EQ[1][:,0]))
    EK = pl(1,K- numpy.mean(K,axis=1)[:,None],H,M,I,"K mean",m,n)
    print("K", numpy.max(EK[1][:,0]))
    EV = pl(2,V,H,M,I,"V mean",m,n)
    print("V", numpy.max(EV[1][:,0]))
    
    plt.show()

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
    

def comp(i, H = 32, M =128, I = 3072):

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
    #R2 = mha.shead(Q,K,V)
 
        
    #G = R1-R2
    #A = numpy.fabs(G)
    #a =numpy.mean(A)
    #b =numpy.max(A)
    #print(i, "RScipy L1 %1.3e %1.3e" % (a,b))

    #    plt.hist(G.flatten(),100)
    #    plt.show()

    R = mha.heads(Q,K,V,  H, mha.shead)   
    #G = R-R2
    #A = numpy.fabs(G)
    #a =numpy.mean(A)
    #b =numpy.max(A)
    #print(i, "Cross Scipy Heads L1 %1.3e %1.3e" % (a,b))

#    plt.hist(G.flatten(),100)
#    plt.show()

    G = R1-R
    A = numpy.fabs(G)
    a =numpy.mean(A)
    b =numpy.max(A)
    print(i, "RScipy2 Heads L1 %1.3e %1.3e" % (a,b))
    El[0] = list(G.flatten())
    E[0,:] = [ a,b]

#    plt.hist(G.flatten(),100)
#    plt.show()
    

#    import pdb; pdb.set_trace()

                   
    ## same as above but float16
    Q16 = numpy.ndarray.astype(Q,numpy.float16)
    K16 = numpy.ndarray.astype(K,numpy.float16)
    V16 = numpy.ndarray.astype(V,numpy.float16)
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

    def bl( Q : numpy.array, ## operand 
            K : numpy.array, ## operand 
            V : numpy.array):
        return mha.ddr_computation_block(Q,K,V,RT,False)

    three = mha.heads(
        Q16,K16,V16, H,
        bl)
    #    three,_,_,_  = mha.ddr_computation_block(Q16,K16,V16,RT)
    #print(three.dtype)
    G = R-three
    A = numpy.fabs(G)
    a =numpy.mean(A)
    b =numpy.max(A)
    print(i,"block L1 %1.3e %1.3e" % (a,b))
    El[3] += list(G.flatten())
    E[3,:] = [ a,b]


    def sagebl( Q : numpy.array, ## operand 
            K : numpy.array, ## operand 
            V : numpy.array):
        return mha.sage_computation_block(Q,K,V,[1,1,1,1],KN=False)

    four= mha.heads(
        Q16,K16,V16, H,
        sagebl
    )
    #print(four.dtype)
        
    G = R-four
    A = numpy.fabs(G)
    a =numpy.mean(A)
    b =numpy.max(A)
    print(i, "sage  L1 %1.3e %1.3e" % (a,b))
    El[4] += list(G.flatten())
    E[4,:] = [ a,b]

    return [El, E]





if __name__ == "__main__":

    from multiprocessing import Pool

    if False:
        R  = [i for i in range(32)]
        for i in R:
            dist(i)
        
    if True:
        results = [comp(0) ]
        
    if False :
        results = [quant(0,1,1),quant(0,8,8),quant(0,16,16),
                   quant(17,1,1),quant(17,8,8),quant(17,16,16)
                   ]
        
    if False:
        R  = [i for i in range(32)]
        with Pool(processes=32) as pool: # Create a pool with 4 worker processes
            results = pool.map(comp, R)
            #print(results)
        


        E =  numpy.zeros((5,len(R),2))
        El =  [ [] for i in range(5) ]
        for i in R:
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
            plt.yscale('log')
            plt.savefig(names[i]+".png") 
            plt.show()
            i+=1

