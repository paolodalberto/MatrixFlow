"""
 Copyright 2019 Xilinx Inc.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""



import copy
import math
from typing import Any, Dict
import numpy

import pdb
import scipy

Frequency = (2)*10**9    # 1.8GHz
Bandwidth = 4 * 8*(2**30) # 6 GBs

import os 
from code_gemm import Gemm, MHA, CONV
import pickle


###
##    y = CONV(x)+b  
##    x = [-1, hx,wx,cx,type]
##    y = [-1, hy,wy,cy,type]
##    b = [-2, 1, 1, cy,type]
##    w = [cy,h, w, cx,type], p = [0,hp, wp, 0, 0], s = [0,hs,ws,0,0]
##
##    Problem definition [y,x,w,b,p,s]
###
    
class CONVLib(CONV):

    def __init__(
            self,
            memtilesizeinbits : int = (512*2**10)*8,
            ROWS               : int = 4 ,
            COLS               : int = 2 ,
            CoreCSpacebits     : int = 2*(8*2**10)*8, ## 2 banks    
            CoreASpacebits     : int = 2*(8*2**10)*8, ## 2 banks        
            CoreBSpacebits     : int = 2*(8*2**10)*8, ## 2 banks        
            h_align : int = 8,
            w_align: int  = 8,
            c_align: int  = 8,
            frequency : int = Frequency, # 2GHz
            to_core_channel_bandwidth_gbits : int = Bandwidth,  # 4GBs 
            to_mem_channel_bandwidth_gbits  : int = Bandwidth,  # 4GBs
            memory : int = 0
    ):

        
        CONV.__init__(self,
            memtilesizeinbits,ROWS,COLS,
            CoreCSpacebits,CoreASpacebits,CoreBSpacebits,
            h_align,w_align,c_align,frequency,
            to_core_channel_bandwidth_gbits,to_mem_channel_bandwidth_gbits)
        

        self.perfdictionary = {}
        self.best = {}

        if memory==2 and os.path.exists("perfconvdict.pkl"):
            self.load()
        if memory==1 and os.path.exists("bestconvdict.pkl"):
            with open("bestconvdict.pkl", 'rb') as f:
                self.best = pickle.load(f)
                f.close()

    # we save and load the dictionary stored as pickle
    def save(self,name:str= "dict.pkl"):

        with open("bestconv"+name, 'wb') as f:
            pickle.dump(self.best, f)
        with open("perfconv"+name, 'wb') as f:
            pickle.dump(self.perfdictionary, f)

    def load(self,name:str= "dict.pkl"):
        
        if len( self.best)==0 and  os.path.exists("bestconv"+name):
            with open("bestconv"+name, 'rb') as f:
                self.best = pickle.load(f)
                f.close()
        if len(self.perfdictionary)==0 and os.path.exists("perfconv"+name):
            with open("perfconv"+name, 'rb') as f:
                self.perfdictionary = pickle.load(f)
                f.close()

    ###
    ##  1) the basic computation is by h in incremental steps streaming
    ##  from memtile
    ##
    ##  2) the input and output are split by column by the w dimension
    ###
    def gen_fm_par_fm_(self, X, expsub : int = None):
        y,x,w,b,p,s = X

        # space in Bytes
        Mem = self.Mem//8
        
        SY = self.cspace//y[-1]//2  ## ping output
        SX = self.aspace//x[-1]//2  ## ping input 
        SW = self.bspace//w[-1]     ## whole

        ## aligned to MemTile requirements
        y = self.align_t(y)
        x = self.align_t(x)
        ww = self.align_t(w)
        b = self.align_t(b)

        ## all valid stream using ping pong at  core level
        MM = 1
        MM = {} ; self.perfdictionary

        def add_time(cm : list, P:list) -> float  :
            key = str(cm)+',' + str(cm[1])
            if key in self.perfdictionary:
                
                timem = self.perfdictionary[key]
            else:
                timem =  self.time_estimates(cm, cm[1])
                MM[key] = timem
            key = str(cm)+',' + str(P)
            if key in self.perfdictionary:
                x,p,timet = self.perfdictionary[key]
            else:
                timet =  self.time_estimates(cm,P)
                MM[key]=  timet
            return timet


        P = [y,x,ww,b,p,s]


        if str(P) in self.best:
            print("Done From Memory", P,self.best[str(P)])
            return self.best[str(P)], {}, {}
        
        ## basic unit for the output core
        uc = [-1, 1, 1, self.c_align, y[-1]]

        ## basic unit for the output Memtile
        um = self.projection_c_m_o(uc,y)


        W = []

        ## height computation. Usually a core will stream by height
        ## and this means that a core may work on a single H block but
        ## the memtile can be the whole dimensions. 

        for h in range(1,y[1]):
            
            ## we split the computatio by column by width one core
            ## will see only half the number of widths

            for w in range(1,math.ceil(y[2]/self.COLS)):

                ## min block per core Each Core will write a sub group
                ## of channels: this is because the computation must
                ## be independent.
                
                COUT = max(
                    math.ceil(y[3]/self.ROWS/self.c_align)*self.c_align,
                    self.c_align
                )
                
                ## we have H, W/COLS, COUT/ROWS this is the valid set
                ## for a core in the system.
                for c in range(self.c_align,COUT+1, self.c_align):
                    
                    ## for cx in range(self.c_align,x[3]+1, 2*self.c_align):
                    ##
                    ## we are enforcing that we work on the whole input channel  
                    ##
                    if True:
                        cx = x[3] if expsub is None else expsub
                        uc = [-1, h , w, c, y[-1]]

                        ## this will project back to two column and
                        ## four rows: this is because memtile is
                        ## overall
                        um = self.projection_c_m_o(uc,y)

                        if um[1]>y[1] or um[2]>y[2]: continue

                        # uc projection into input x core
                        xc = self.projection(uc,X)
                        xc[3] = cx

                        # um projection into input x memtile 
                        xm = self.projection(um,X)
                        xm[3] = cx

                        if xm[1]>x[1] or xm[2]>x[2]: continue

                        ## current core weights and bias
                        wc = [uc[3],ww[1],ww[2],xc[3], ww[4]]
                        bc = [ b[0],b[1],b[2],uc[3], b[4]]

                        ## current mem weight and bias 
                        wm = [um[3],ww[1],ww[2],xm[3], ww[4]]
                        bm = [ b[0],b[1],b[2],um[3], b[4]]

                        
                        
                        if self.space_e(uc) < SY and self.space_e(xc) < SX and  self.space_e(wc) + self.space_e(bc) < SW and \
                           self.space_m(um) +self.space_m(xm) +self.space_m(wm) +self.space_m(bm)  < Mem : # MEM
                            
                            core_problem = [uc, xc,wc,bc,p,s]
                            mem_problem  = [um, xm,wm,bm,p,s]

                            timet = add_time([core_problem, mem_problem],P)
                            W.append([[core_problem, mem_problem],timet])
                            
                            
                            ## now let's us find the largest
                            ## subproblem that fit memtile
                            
                            for hh in range(h,y[1],h) :
                                for wwh in range(w,math.ceil(y[2]/self.COLS),w):
                                    uc = [-1, hh , wwh, c, y[-1]]

                                    ## projection on memmtile (col and row)
                                    um = self.projection_c_m_o(uc,y)
                                    if um[1]>y[1] or um[2]>y[2]: continue

                                    ## projection of the input in memtile
                                    xc = self.projection(uc,X)
                                    xc[3] = cx
                                    xm = self.projection(um,X)
                                    xm[3] = cx
                                    
                                    if xm[1]>x[1] or xm[2]>x[2]: continue
                                    
                                    ## the core computatio does not
                                    ## change and thus the W and B
                                    if  self.space_m(um) +self.space_m(xm) +self.space_m(wm) +self.space_m(bm)  < Mem : # MEM
                                        mem_problem  = [um, xm,wm,bm,p,s]

                                        timet = add_time([core_problem, mem_problem],P)
                                        W.append([[core_problem, mem_problem],timet])
                                        

        
#        for k,v in MM.items():
#            if len(v)==2:
#                pdb.set_trace()
        if (len(W))==0 :
            print("Done", P, len(MM),"None")
            return W, {}, {}
        if False:
            S = sorted(
                W,
                key = lambda x:
                ( -self.compute(x[0]), -self.fat_mem(x[0]))
            )
        
        
            Ref = self.minimum_computation_time(P)
            print("expsub", expsub,"TOP 3 by the fattest core computation")
            for i in range(min(3,len(S))):
                time = self.time_estimates(S[i][0],P)
                #print(self.compute(S[i]), "y, x,  wc, b, p,s", S[i])
                print("time", time,"Ref",Ref, "slowd", time/Ref,"ycore", S[i][0][0],"ymem", S[i][1][0])
            #pdb.set_trace()   

        Ref = self.minimum_computation_time(P)
        S1 = sorted(
            W,
            key = lambda x:
            ( x[1],-self.compute(x[0]), -self.fat_mem(x[0]))
        )
        if False:
            print("TOP 3 by the fastest overal computation")
            for i in range(min(3,len(S))):
                time = self.time_estimates(S1[i],P)
                #print(self.compute(S[i]), "y, x,  wc, b, p,s", S[i])
                print("time", time,"Ref",Ref, "slowd", time/Ref,"ycore", S1[i][0][0],"ymem", S1[i][1][0])
                
        
        B = {}
        B[str(P)] = S1[0]
        print("Done", P,  len(MM),S1[0][-1])
        return S1[0], MM,B 


            

###
##     The problem size for a GEMM is [m,n,k,atypem, btype, ctype)
##     C = m x n  ctype in bits
##     A = m x k  atype in bits
##     B = k x n  atype in bits
##
##     We are considering building a library and thus we want to reuse
##     the estimate for all GEMM.  We have a dictionary for the
##     * cluster computation core mem.
##       mem is a multiple of the core so the comptutation will be a symmetric one
##     * cluster computation ddr -> mem -> core
##           we assume that the mem - core is reused when possible
##     * given a problem P we have the best estimate.
##           We can search further for solution with other constraints if necessary.
###

class GemmLib(Gemm):

    def __init__(
            self,
            memtilesizeinbits : int = (512*2**10)*8,
            ROWS               : int = 4 ,
            COLS               : int = 2 ,
            CoreCSpacebits     : int = 2*(8*2**10)*8, ## 2 banks    
            CoreASpacebits     : int = 2*(8*2**10)*8, ## 2 banks        
            CoreBSpacebits     : int = 2*(8*2**10)*8, ## 2 banks        
            m_align : int = 16,
            n_align: int  = 8,
            k_align: int  = 16,
            frequency : int = Frequency, 
            to_core_channel_bandwidth_gbits : int = Bandwidth,  # 4GBs 
            to_mem_channel_bandwidth_gbits  : int = Bandwidth,   # 4GBs
            memory : bool = True
    ):

        Gemm.__init__(self,
            memtilesizeinbits,ROWS,COLS,
            CoreCSpacebits,CoreASpacebits,CoreBSpacebits,
            m_align,n_align,k_align,frequency,
            to_core_channel_bandwidth_gbits,to_mem_channel_bandwidth_gbits)

        
        self.perfdictionary = {}
        self.memcoredictionary = {}
        self.best = {}

        if memory and os.path.exists("mcdict.pkl"):
            self.load()


    # we save and load the dictionary stored as pickle
    def save(self,name:str= "dict.pkl"):
        import pickle 

        with open("best"+name, 'wb') as f:
            pickle.dump(self.best, f)
        with open("mc"+name, 'wb') as f:
            pickle.dump(self.memcoredictionary, f)
        with open("perf"+name, 'wb') as f:
            pickle.dump(self.perfdictionary, f)

    def load(self,name:str= "dict.pkl"):
        import pickle 

        with open("mc"+name, 'rb') as f:
            self.memcoredictionary = pickle.load(f)
        with open("best"+name, 'rb') as f:
            self.best = pickle.load(f)
        with open("perf"+name, 'rb') as f:
            self.perfdictionary = pickle.load(f)


    # this is in case we want to sort only elements that are a
    # multiple of core (x = core size)
    def Mod(self, x:list, P:list):
        cm,cn,ck, _,_,_ = x
        m,n,k, _,_,_    = P

        t = (m%cm)+  (n % cn) +  (k% ck)
        return   100 if t>0 else 1

    # this is in case we want to sort only elements that are a
    # multiple of core but the x is memtile.
    def ModM(self, x:list, P:list,Q):
        m,n,k, _,_,_    = P

        cm,cn,ck, _,_,_ = self.coresubvolume(x,P,Q)
        t = (m%cm)+  (n % cn) +  (k% ck)

        return  100 if t>0 else 1
        
        




    ## estimate ddr_ddr_ time estimate given a memtile subproblem
    def time_estimates(
            self,
            Tsub    : list , ## memtile subproblem
            P       : list , ## aligned problem size
            v       : bool = False,
            core    : list = None, ## core sub problem
            cluster : float = None ## if the luster is already computed, this is the time. 
    ):

        ## this is Memtile space in Bytes (the unit of every opspace
        Mem = self.Mem//8 # bytes

        MA,NA,KA,atype, btype, ctype = P
        #pdb.set_trace()
        norm    = self.opspace(Tsub) < Mem 
        dbb     = self.opspace_dbb(Tsub) < Mem
        dba     = self.opspace_dba(Tsub) < Mem
        dba_dbb = self.opspace_dba_dbb(Tsub) < Mem 
        Ksplit = Tsub[2]<KA


        TotalTime = 0 
        TT = self.get_time_cluster(Tsub) if core is None else (
            self.get_time_cluster_with_latency(Tsub,core) if cluster is None else cluster
        )

        if Ksplit:

            if v: print("Ksplit")
            ## in principle we DB A and B and we cycle in the K dimension
            TotalTime = 0
            for j in range(0, NA, Tsub[1]):
                for i in range(0, MA, Tsub[0]):

                    # LOAD A,B, (compute C, load A,B)[n-1] compute C
                    TotalTime += (self.opspace(Tsub)/(self.ddrchannels*self.tomembandwidth)) +\
                        max (
                            TT,
                            (self.opspace(Tsub)/(self.ddrchannels*self.tomembandwidth))
                        )* (KA//Tsub[2] -1 ) \
                        + TT
                    
                    TotalTime += (Tsub[0]*Tsub[1])//8/(self.COLS*self.tocorebandwidth) ## writing C in Memtile
                    TotalTime += (Tsub[0]*Tsub[1])//8/(self.ddrchannels*self.tomembandwidth)  ## writing C in DDR
                
        elif dbb :
            if v: print("dbb")
        
            # We stream A by DB
            TotalTime = 0
            for i in range(0, MA, Tsub[0]):
                ## LOAD A_i once and 
                TotalTime += (Tsub[0]*Tsub[2])//8/(self.ddrchannels*self.tomembandwidth)
                
                ## We DB B with compute ... 
                ## load B ( compute C , load B) 
                TotalTime += (Tsub[1]*Tsub[2])//8/(self.ddrchannels*self.tomembandwidth) + \
                    max(
                        TT,
                        (Tsub[1]*Tsub[2])//8/(self.ddrchannels*self.tomembandwidth)
                    )* (NA/Tsub[1]-1) + \
                    TT
                TotalTime += (Tsub[0]*Tsub[1])//8/(self.COLS*self.tocorebandwidth) ## writing C in Memtile
                TotalTime += (Tsub[0]*Tsub[1])//8/(self.ddrchannels*self.tomembandwidth)  ## writing C in DDR
                #print(TotalTime)
            #pdb.set_trace()
        elif  dba :
            if v: print("dba")
            # We stream A by DB
        
            TotalTime = 0
            for j in range(0, NA, Tsub[1]):
                ## load B_j once and 
                TotalTime += (Tsub[1]*Tsub[2])//8/(self.ddrchannels*self.tomembandwidth)
        
                ## load A ( compute C , load A) 
                TotalTime += (Tsub[0]*Tsub[2])//8/(self.ddrchannels*self.tomembandwidth) + \
                    max(
                        TT,
                        (Tsub[0]*Tsub[2])//8/(self.ddrchannels*self.tomembandwidth)
                    )* (MA/Tsub[0]-1) + \
                    TT
                
                TotalTime += (Tsub[0]*Tsub[1])//8/(self.COLS*self.tocorebandwidth) ## writing C in Memtile
                TotalTime += (Tsub[0]*Tsub[1])//8/(self.ddrchannels*self.tomembandwidth)  ## writing C in DDR
        elif dba_dbb:
            if v: print("dba_dbb")
            
            ## We double buffer both A and B
            ## first A
            tick = (Tsub[0]*Tsub[2])//8/(self.ddrchannels*self.tomembandwidth)

            TotalTime = tick
            
            for i in range(0, MA, Tsub[0]):
                # iteration A
                
                ## load B ( compute C , load A) 
                CTime = (Tsub[1]*Tsub[2])//8/(self.ddrchannels*self.tomembandwidth) + \
                    max(
                        TT
                        (Tsub[1]*Tsub[2])//8/(self.ddrchannels*self.tomembandwidth)
                    )* (NA/Tsub[1]-1) + \
                    TT
        
                TotalTime += max(CTime,tick)
                TotalTime += (Tsub[0]*Tsub[1])//8/(self.COLS*self.tocorebandwidth) ## writing C in Memtile
                TotalTime += (Tsub[0]*Tsub[1])//8/(self.ddrchannels*self.tomembandwidth)  ## writing C in DDR
        
        if TotalTime ==0:
            if v: print("None", norm, dba,dbb, dba_dbb,Ksplit)

            ## non of the above fit in Memtile Time => infinity
            TotalTime = 1000000.0
            #pdb.set_trace()
            #print("Ksplit", Ksplit, KA//Tsub[2], "total time second ",  TotalTime)

        return TotalTime

    

    def coresubvolume(
            self,
            memsubvolume : list,
            P : list,
            Q : int
    ):

        MA,NA,KA,_,_,_ = P
        if Q == 0 :
            ## C Tall
            subvolume    =  [ i for i in memsubvolume]
            subvolume[0] =  subvolume[0]//self.ROWS
            subvolume[1] =  subvolume[1]//self.COLS
            subvolume[2] =  KA
        else:
            ## C Fat
            subvolume    =  [ i for i in memsubvolume]
            subvolume[0] =  subvolume[0]//self.COLS
            subvolume[1] =  subvolume[1]//self.ROWS
            subvolume[2] =  KA
        return subvolume

    ###
    ## This provide the overall time estimate and I know it is very
    ## optimistic.
    ##
    ## But if you are pssimistic, you will not get out of the bed.
    ###
    def time(
            self,
            coresubvolume : list,
            memsub        : list,
            P             : list,
            cluster : float=  None 
    ):
        
        
        MA,NA,KA, _,_,_ = P
        subvolume = memsub
        
        ## this is Memtile space in Bytes (the unit of every opspace
        Mem = self.Mem//8 # bytes
        norm    = self.opspace(subvolume) < Mem 
        if norm == False:
            #pdb.set_trace()
            # we split K
            TTsub = [i for i in subvolume]
            for i in range(2, 16, 2):
                TTsub[2] = math.ceil(subvolume[2]//i)
                norm     = self.opspace_dba_dbb(TTsub) < Mem
                if norm:
                    #pdb.set_trace()
                    subvolume =  TTsub
                    Ksplit = True
                    break
            
        #time0 = self.time_estimates(subvolume,P)
        time0 = self.time_estimates(subvolume,P,core=coresubvolume,cluster=cluster )
        
        return time0
            

        
        
    
        
                            
    ## this is the code generator: problem size and operands 
    ## M, N, K, atype, btype, ctype  = X
    ## C =MxN (ctype). A = MxK (atype).  B = KxN (btype)
    def gen_fm_par_fm_(
            self,
            X : list, 
            expsub : list = None,
            scaling : int = 32
    ):

        ## C =MxN (ctype). A = MxK (atype).  B = KxN (btype)
        M, N, K, atype, btype, ctype  = X

        #print("GEMM Problem M,N,K,atype,btype,ctype ", [M,N,K, atype,btype,ctype])

        
        ## ping pong space for the core (there are two banks we use
        ## only one streaming A nd B, Notice the space is normalized
        ## as number of elements more than bits.  We assume here two
        ## banks per tensor.

        #Space for C is different .. we may acumulate directly in 32
        #bits which is different

        scaling = 32  ## accumulation is done in 32
        SC = self.cspace//scaling #len(ofm.banks) we don't



        SA = self.aspace//atype//2
        SB = self.bspace//btype//2

        ## this is Memtile space in Bytes (the unit of every opspace
        Mem = self.Mem//8 # bytes
        
        # the alignement for input 
        MA = math.ceil(M/ self.m_align)* self.m_align
        NA = math.ceil(N/self.n_align)* self.n_align
        ali = self.k_align
        KA = math.ceil(K/  ali)* ali

        P = [MA,NA,KA,atype,btype,ctype]
        #print("ALIGNED", P)
        if str(P) in  self.perfdictionary:
            Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q, TotalTime        =    self.perfdictionary[str(P)]
            return Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q, [], TotalTime
        ## number of aligned blocks 
        mt = MA// self.m_align
        nt = NA// self.n_align
        kt = KA// ali

        ## all valid stream using ping pong at  core level
        MM = 1
        MC = self.memcoredictionary
        MM = self.perfdictionary
        best = self.best


        if str(P) in best:

            coresubvolume,  Tsub, P, TotalTime = best[str(P)]
            FatC = 1 if Tsub[0] == coresubvolume[0]*self.COLS else 0 
        
            Q = self.generate_Q(FatC)
            #print(Q)
            #print(P,"MC aaditions",MCaddition , "MM additions ", MMaddition )
            
            #pdb.set_trace()
            norm    = self.opspace(Tsub) < Mem 
            dbb     = self.opspace_dbb(Tsub) < Mem
            dba     = self.opspace_dba(Tsub) < Mem
            dba_dbb = self.opspace_dba_dbb(Tsub) < Mem 
            
            Ksplit = Tsub[2]<KA

            return Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q, {},{},{},  TotalTime
            
        scaling = 32/ctype  ## accumulation is done in 32

        MCaddition = 0; TMC= {}
        MMaddition = 0; TMM= {}
        #pdb.set_trace()
        for m in range(mt) :
            cm = (m+1)*self.m_align
            for n in range(nt): 
                cn = (n+1)*self.n_align
                for k in range(kt):
                    kn = (k+1)*ali
                    if  cm*cn < SC and cm*kn < SA and kn*cn < SB :
                        # These are valid solutions, we count the
                        # number of elements per core memory (as
                        # number of elements)
                        q =  [cm, cn,kn, atype,btype,ctype]
                        
                        
                        for p in [0,1]:
                            subvolume = self.memsubvolume(q,P,p)

                            if subvolume > P: continue

                            if self.opspace_dba(subvolume)<Mem or self.opspace_dbb(subvolume)<Mem:
                                #pdb.set_trace()
                                key = str([q, subvolume])
                                if not key in MC: 
                                    ctime = self.get_time_cluster_with_latency(subvolume, q)
                                    #MC[key] = [q, subvolume, ctime]
                                    TMC[key] = [q, subvolume, ctime]
                                    
                                    MCaddition +=1
                                else:
                                    _,_,ctime = MC[key] 

                                key = str([q, subvolume,P])
                                if not key in MM:
                                    
                                    time = self.time(q, subvolume,P,cluster = ctime)
                                    #MM[key] = [q, subvolume,P, time]
                                    TMM[key] = [q, subvolume,P, time]
                                    MMaddition +=1

                                else:
                                     _,_,_, time = MM[key] 
                                key = str(P)
                                if not key in best:
                                    #pdb.set_trace()
                                    best[key] =  [q, subvolume,P, time]
                                else:
                                    
                                    _,_,_, rtime = best[key]
                                    if rtime > time:
                                        #pdb.set_trace()
                                        best[key] =  [q, subvolume,P, time]
                                        

        #pdb.set_trace()
        if expsub is None and str(P)  in best:
            W = [best[str(P)]]
        else:
            #pdb.set_trace()
            W = sorted( MM.values(), key= lambda x: (x[-1]*self.Mod(x[0],P)*(1000 if x[3]!=P else 1)))

        if W is None or  len(W)==0:
            return None, None,None,None,None, None,None, None, TMM , TMC , {}, 1000
            
        Tsub = W[0][1]
        coresubvolume = W[0][0]
        FatC = 1 if Tsub[0] == coresubvolume[0]*self.COLS else 0
        
        Q = self.generate_Q(FatC)
        #print(Q)
        #print(P,"MC aaditions",MCaddition , "MM additions ", MMaddition )

        #pdb.set_trace()
        norm    = self.opspace(Tsub) < Mem 
        dbb     = self.opspace_dbb(Tsub) < Mem
        dba     = self.opspace_dba(Tsub) < Mem
        dba_dbb = self.opspace_dba_dbb(Tsub) < Mem 

        Ksplit = Tsub[2]<KA
        
        
        
        TotalTime = W[0][-1]
#        if P == [512, 512, 1024, 16, 8, 16]:
#            pdb.set_trace()
        if not str(P) in best:
            best[str(P)] =  [coresubvolume, Tsub, P,  TotalTime]
        else:
            _,_,_, rtime = best[str(P)]
            if rtime> TotalTime:
                best[str(P)] = [coresubvolume, Tsub, P,  TotalTime]
        
            
        #print("Ksplit", Ksplit, KA//Tsub[2], "total time second ",  TotalTime)
        print("Done search ", P,MCaddition,MMaddition)
        return Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q, TMM,TMC, best, TotalTime
    

            
        
###
##     The problem size for a MHA is [d,L,type)
##     Q = L0 x d  type
##     K = d x L1  type in bits
##     V = L1 x d  type in bits
##     G = L1 x L1  Gated
##     Here we assume all operand of the same precision/type
###
class MHALib(MHA):

    def __init__(
            self,
            memtilesizeinbits : int = (512*2**10)*8,
            ROWS               : int = 4 ,
            COLS               : int = 2 ,
            CoreCSpacebits     : int = 2*(8*2**10)*8, ## 2 banks    
            CoreASpacebits     : int = 2*(8*2**10)*8, ## 2 banks        
            CoreBSpacebits     : int = 2*(8*2**10)*8, ## 2 banks        
            m_align : int = 8,
            n_align: int  = 8,
            k_align: int  = 8,
            frequency : int = Frequency, # 2GHz
            to_core_channel_bandwidth_gbits : int = Bandwidth,  # 4GBs 
            to_mem_channel_bandwidth_gbits  : int = Bandwidth,  # 4GBs
            memory : bool = True
    ):

        
        MHA.__init__(self,
            memtilesizeinbits,ROWS,COLS,
            CoreCSpacebits,CoreASpacebits,CoreBSpacebits,
            m_align,n_align,k_align,frequency,
            to_core_channel_bandwidth_gbits,to_mem_channel_bandwidth_gbits)
        
        self.memcoredictionary = {}
        self.best = {}

        if memory and os.path.exists("mcdict.pkl"):
            self.load()


    # we save and load the dictionary stored as pickle
    def save(self,name:str= "dict.pkl"):
        import pickle 

        with open("bestmha"+name, 'wb') as f:
            pickle.dump(self.best, f)
        with open("mcmha"+name, 'wb') as f:
            pickle.dump(self.memcoredictionary, f)

    def load(self,name:str= "dict.pkl"):
        import pickle 

        with open("mcmha"+name, 'rb') as f:
            self.memcoredictionary = pickle.load(f)
        with open("bestmha"+name, 'rb') as f:
            self.best = pickle.load(f)



    ## estimate ddr_ddr_ time estimate given a memtile subproblem
    ## m,n,Qtime, KVtime,P,space,t= X
    ## d, L0, L1, r,dtype = P
    def time_estimates(
            self,
            X : list , ## memtile subproblem
            gate : bool = False
            
    ):
        #pdb.set_trace()
        m,n,Qtime, KVtime,P,space,t= X
        d, L0, L1, r,dtype = P

        q0 = [m, d, dtype]
        k0 = [d, n, dtype]
        v0 = [n, r, dtype]
        
        q0k0    = [m,n, dtype]
        gated   = [m,n, dtype]
        expq0k0 = [m,n, dtype]
        expq0k0v0 = [m, r, dtype]
        
        N_t = [m,r,   dtype]
        D_t = [m,1,   dtype]
        M_t = [m,1,   dtype]
        M_t_1 = [m,1, dtype]
        
        Ncores = self.ROWS*self.COLS        

        total_time = (self.space_m(k0)+ self.space_m(v0))/self.tocorebandwidth
        
        if KVtime<=Ncores:
            
            ## cluster Mem to core
            TT = 0
            load = (self.space_m(q0)+ (self.space_m(gated) if gate else 0 ))/self.tocorebandwidth

            core_time = []
            for k in range(KVtime):
                
                cycles = self.compute([m,n,d,dtype,dtype,dtype]) # numpy.matmul(Q0,KI)
                if gate: cycles += self.element_compute([m,n,dtype])              # + gated
                cycles += self.element_compute([m,n,dtype])              # numpy.max(T,1)
                cycles += self.element_compute([m,n,dtype])              # numpy.maximum(M,M1)
                cycles += 2*self.element_compute([m,n,dtype])              # numpy.exp(T-M1[:,None])
                cycles += 2*self.element_compute([m,n,dtype])              # numpy.exp(M1-M)
                cycles += 2*self.element_compute([m,1,dtype])            # D/S  + sum(T.transpose())
                cycles += self.element_compute([m,n,dtype])+ \
                    self.compute([m,d,n,dtype,dtype,dtype])      #  N/S[:,None] + numpy.matmul(T,VI)
                    
                core_time.append(cycles)
                
            M = max(core_time) 
            M = M* math.ceil(KVtime/Ncores)
            ## this is to combine the partial results 
            time = (M + \
                    (KVtime-1)* self.element_compute([m,n,dtype]))/self.core_frequency

            time += 2*(self.space_m([m,n,dtype])+self.space_m([m,1,dtype]))*(KVtime-1)/self.tocorebandwidth
            write = (self.space_m(expq0k0v0))/self.tocorebandwidth
                
            TT = load + time + write

            load = (self.space_m(q0)+ (self.space_m(gated) if gate else 0))/self.tomembandwidth

            write = (self.space_m(expq0k0v0))/self.tomembandwidth
            #pdb.set_trace()
            if KVtime <= self.ROWS:
                Qtime /= self.COLS
            WQ = math.ceil(L0/m)
            total_time = load +\
                max(
                    load,
                    TT,
                    write
                )*Qtime + \
                TT+write
        else:
            total_time = 1000000000
        return total_time


    def Q_i_gated(self, Q, K, V, exptype : int = 16, Qping : int = 1, gate: bool = True ):
            
        # what is the size of Q_0, K_i, V_i so that exp(Q_0*K_i) * V_i
        # fit in a core?
        
        W = []
        it = math.ceil(K[1]/self.k_align)
        iq = math.ceil(Q[0]/self.m_align)

        #     d     L0(Q) L1(K/V) r     type
        P = [ K[0], Q[0], K[1],   V[1], V[-1] ]
        dr = [ K[0],V[1]]
        for i in range(1,iq):
            for j in range(1,it):
                qm = i*self.m_align
                kn = j*self.k_align

                q0 = [qm, Q[1], Q[2]]
                k0 = [K[0], kn, K[2]]
                v0 = [kn, V[1], V[2]]
                
                q0k0     = [qm,kn, exptype]
                gated    = [qm,kn, exptype] # we have an extra addition
                expq0k0  = [qm,kn, exptype]
                expq0k0v0 = [qm,V[1], exptype]

                N_t = [qm,V[1], exptype]
                D_t = [qm,1, exptype]
                M_t = [qm,1, exptype]
                M_t_1 = [qm,1, exptype]
                
                space =self.space_m(q0)*Qping + \
                    self.space_m(k0) + self.space_m(v0) + \
                    self.space_m(expq0k0) + \
                    (self.space_m(gated) if gate else 0) +  \
                    self.space_m(N_t)+ self.space_m(N_t)+ \
                    self.space_m(M_t) + self.space_m(M_t_1)

                if space < (self.cspace + self.aspace + self.bspace)//8:
                    time = self.time_estimates([qm,kn, math.ceil(iq/i), math.ceil(it/j),P,space,0], gate = gate)
                    W.append([qm,kn, math.ceil(iq/i), math.ceil(it/j), P, space,time])
        return W


    ## tiling code geneation we have the problem to solve [d,l,type in
    ## bits] X
    def gen_fm_par_fm_(
            self,
            X : list,
            gated : bool= True 
            
    ) -> list :

        d, L0, L1, r, dtype = X
        
        d  = math.ceil(d/self.n_align)*self.n_align
        L0 = math.ceil(L0/self.m_align)*self.m_align
        L1 = math.ceil(L1/self.k_align)*self.k_align
        r  = math.ceil(r/self.n_align)*self.n_align
        
        Q = [L0, d , dtype ]  ## matrix shapes
        K = [d, L1 , dtype ]  
        V = [L1, r , dtype ]  
        
        
        P = [d,L0,L1,r, dtype]
        if str(P) in self.best:
              return self.best[str(P)], {}, {}
           
        S = self.Q_i_gated(Q,K,V,gated)
        
        W1 = sorted(
            S,               #  time, cores 
            key = lambda x: ( x[-1], x[2])
        )
        M = {}
        for w in W1:
            key = str(w[0:4])+str(w[5])
            if not key in self.memcoredictionary and w[-1]<100:
                M[key] = w
            
        #for w in W1[0:1]:
        #    pdb.set_trace()
        #    print(w, self.time_estimates(w,gated))
        
        print("Done", X, W1[0])
        B= {}
        B[str(P)] = W1[0] 
        
        return W1[0], B, M


def test_gemm() :
    from multiprocessing import Pool

    gem = GemmLib()
    print(len(gem.memcoredictionary),len(gem.perfdictionary),len(gem.best))
    #pdb.set_trace()
    
    Space = []

    Start = 128
    Step  = 64
    

    for M in range(Start,Start*8, Step):
        for N in range(Start,Start*8, Step):
            for K in range(Start,Start*8, Step):


                Space.append([M,N,K,8,8,8])

    
    print(len(Space))

    ## this is the map phase and it is parallel.  we take advantage
    ## that each thread will share the dictionary and each will
    ## introduce a new element but in a local copy. The result will
    ## have the new dictionaries
    with Pool(min(16,len(Space))) as pool:
      result = pool.map( gem.gen_fm_par_fm_, Space)

    print("Reduce")
    ## this is the reduce phase, where we join the dictionary so that
    ## the next iteration we will have more knowledge about the
    ## solution space.
    for r in result:
        Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q, MM,MC, best,TotalTime = r
        gem.memcoredictionary.update(MC)
        gem.perfdictionary.update(MM)
        gem.best.update(best)
        del MM
        del MC
        del best


    ## this is just take a look at the solution space we tried to
    ## update.
    print(len(gem.memcoredictionary),len(gem.perfdictionary),len(gem.best))
    ## SAVE The PICKLEs 
    gem.save()
    
    for k in Space:
        if not str(k) in gem.best: continue
        
        [core,mem, P,time] =  gem.best[str(k)]
        #print(P)
        Ref = gem.minimum_computation_time(P)
        #print(Ref)
        print(P,"TFLOPS %1.4f" % (2*P[0]*P[1]*P[2]/time/1000000000000), "RTFOLPS %1.4f " % (2*P[0]*P[1]*P[2]/Ref/1000000000000))

###
## We plot the pickles 
##
###
def plot():

    import matplotlib.pyplot as plt
    
    gem = GemmLib()
    print(len(gem.memcoredictionary),len(gem.perfdictionary),len(gem.best))


    t = {}
    x =[]
    y = []
    for k in gem.perfdictionary.keys():
        q,m,P,time = gem.perfdictionary[k]
        ops = (P[0]*P[1]*P[2]*2)
        if ops in t:
            t[ops] = min(t[ops],ops/time/1000000000000)
        else:
            t[ops] = ops/time/1000000000000
    for k,v in t.items():
        x.append(k)
        y.append(v)
    plt.scatter(x, y, c ="red")
        

    x = []
    y = []
    for k in gem.best.keys():
        q,m,P,time = gem.best[k]
        ops = (P[0]*P[1]*P[2]*2)
        x.append(ops)
        y.append(ops/time/1000000000000)
        
    plt.scatter(x, y, c ="blue")
    plt.xlabel("OPS")
    plt.ylabel("TFLOPS")
    plt.title("4x2 GEMM bits %d,%d,%d" % (q[3],q[4],q[5]))
    #plt.show()
    plt.savefig("temp.png")
    plt.show()



def validate(P : list):
    gem = GemmLib(memory=False)
    pdb.set_trace()
    P = P
    result = gem.gen_fm_par_fm_(P)
    mem, core,dba,dbb, dba_dbb, Ksplit,FatC, Q, MM,MC, best,time = result
    gem.memcoredictionary.update(MC)
    gem.perfdictionary.update(MM)
    gem.best.update(best)
    pdb.set_trace()
    Ref = gem.minimum_computation_time(P)
    key = str([core, mem])
    if key in  gem.memcoredictionary:
        _,_,ctime = gem.memcoredictionary[key]
        time2 = gem.time(core, mem,P,cluster=ctime)
        print("diff", Ref,time2, time)
    key = str([core, mem,P])
    if key in gem.perfdictionary:
        _,_,_,ctime2 = gem.perfdictionary[key]
        
    time1 = gem.time(core, mem,P)
    print("diff", Ref,time1, time)


def test_mha():
    mha = MHALib(memory=False)


    
    
    P = [77, 768, 768, 77, 16]
    P0 = [77,  768, 77, 16]
    Ref = mha.minimum_computation_time(P0,True)
    RT,B,M = mha.gen_fm_par_fm_(P,True)
    print(RT,len(B), len(M))
    qm,kn, QTime, KVtime, dr, space,time = RT
    time = mha.time_estimates(RT,True)
    print(P,"time", time, "ref", Ref, "slowdown", time/Ref)

def test_mha_2() :
    from multiprocessing import Pool

    gem =  MHALib()
    print(len(gem.memcoredictionary),len(gem.best))
    #pdb.set_trace()
    
    Space = []

    Start = 128
    Step  = 64
    

    for d in range(40,96, 8):
        for L0 in range(256,1024, 32):
            for r in range(40,96,8):
                Space.append([d,L0,L0,r,8])
            
    
    print(len(Space))

    ## this is the map phase and it is parallel.  we take advantage
    ## that each thread will share the dictionary and each will
    ## introduce a new element but in a local copy. The result will
    ## have the new dictionaries
    with Pool(min(16,len(Space))) as pool:
      result = pool.map( gem.gen_fm_par_fm_, Space)

    print("Reduce")
    ## this is the reduce phase, where we join the dictionary so that
    ## the next iteration we will have more knowledge about the
    ## solution space.
    for r in result:
        
        RT,best,MC = r
        gem.memcoredictionary.update(MC)
        gem.best.update(best)
        del MC
        del best


    ## this is just take a look at the solution space we tried to
    ## update.
    print(len(gem.memcoredictionary),len(gem.best))
    ## SAVE The PICKLEs 
    gem.save()
    
    for k in Space:
        if not str(k) in gem.best: continue
        
        w =  gem.best[str(k)]
        time = w[-1]
        #print(P)
        P = [k[0],k[1],k[3],k[4]]
        Ref = gem.minimum_computation_time(P,True)
        #print(Ref)
        print(P,"TFLOPS %1.4f" % (2*P[0]*P[1]*P[2]/time/1000000000000), "RTFOLPS %1.4f " % (2*P[0]*P[1]*P[2]/Ref/1000000000000))

###
## We plot the pickles 
##
###
def plot_mha():

    import matplotlib.pyplot as plt
    
    gem = MHALib()
    print(len(gem.memcoredictionary),len(gem.perfdictionary),len(gem.best))


    t = {}
    x =[]
    y = []
    for k in gem.memcoredictionary.keys():
        q,k,Qt,Kt,P,space,time = gem.memcoredictionary[k]
        ops = gem.ops(P)
        if ops in t:  t[ops] = min(t[ops],ops/time/1000000000000)
        else:         t[ops] = ops/time/1000000000000
    for k,v in t.items():
        x.append(k)
        y.append(v)
    plt.scatter(x, y, c ="red")
        

    x = []
    y = []
    for k in gem.best.keys():
        q,k,Qt,Kt,P,space,time = gem.best[k]
        ops = gem.ops(P)
        
        x.append(ops)
        y.append(ops/time/1000000000000)
        
    plt.scatter(x, y, c ="blue")
    plt.xlabel("OPS")
    plt.ylabel("TFLOPS")
    plt.title("4x2 MHA bits")
    #plt.show()
    plt.savefig("mha.png")
    plt.show()
def test_conv() :
    from multiprocessing import Pool
    from functools import reduce
    
    gem =  CONVLib(memory =1)
    print( len(gem.perfdictionary),len(gem.best))
    #pdb.set_trace()
    
    Space = []

    Start = 128
    Step  = 64
    s = [0,1,1,0]
    p = [0,0,0,0]


    
    for h in range(256, 512, 128):
        for w in range(256, 512, 128):
#    for h in range(128, 256, 128):
#        for w in range(128, 256, 128):
            for cin in range(8,32, 16):
                for cout in range(8,32, 16):
                    for kh in range(0,8,4):
                        if kh ==0: kh =1
                        for kw in range(0,8,4):
                            if kw ==0: kw =1
                            x = [-1,h, w, cin, 8]
                            y = [-1,h, w, cout,8]
                            ww = [cout,  kh, kw, cin ,8]
                            b = [-2,  1,   1,  cout ,8]
                            Space.append([y,x,ww,b,s,p]) 
                    
    
    print(len(Space))
    
    """
    r = gem.gen_fm_par_fm_(Space[0])
    pdb.set_trace()
    print(len(r))
    RT,MM,best = r
    gem.perfdictionary.update(MM)
    gem.best.update(best)
    del MM
    del best
    r = gem.gen_fm_par_fm_(Space[0])
    pdb.set_trace()
    """

    def reduce_(A,B):
        A[1].update(B[1])
        A[2].update(B[2])
        return A
    
    ## this is the map phase and it is parallel.  we take advantage
    ## that each thread will share the dictionary and each will
    ## introduce a new element but in a local copy. The result will
    ## have the new dictionaries
    with Pool(min(8,len(Space))) as pool:
      result = pool.imap_unordered( gem.gen_fm_par_fm_, Space)
      f_result = reduce(reduce_, result)

      
    print("Reduce")
    ## this is the reduce phase, where we join the dictionary so that
    ## the next iteration we will have more knowledge about the
    ## solution space.

    gem.load()
    RT,MM,best = f_result
    gem.perfdictionary.update(MM)
    gem.best.update(best)
    del MM
    del best


    ## this is just take a look at the solution space we tried to
    ## update.
    print(len(gem.best), len(gem.perfdictionary))
          
    ## SAVE The PICKLEs 
    gem.save()
    
    for k in Space:
        if not str(k) in gem.best: continue
        ops = gem.ops(k)
        c,time =  gem.best[str(k)]
        
        Ref = gem.minimum_computation_time(k)
        #print(Ref)
        print(k,"TFLOPS %1.4f" % (ops/time/1000000000000), "RTFOLPS %1.4f " % (ops/Ref/1000000000000))

###
## We plot the pickles 
##
###
def plot_conv():

    import matplotlib.pyplot as plt
    
    gem = CONVLib(memory=2)
    print(len(gem.perfdictionary),len(gem.best))
    
    
    #for k in gem.perfdictionary.keys():
    #    w = gem.perfdictionary[k]
    #    if len(w) ==2 :
    #        pdb.set_trace()

    x = []
    y = []
    for k in gem.best.keys():
        ref = gem.minimum_computation_time(eval(k))
        ops = gem.ops(eval(k))
        x.append(ops)
        y.append(ops/ref/1000000000000)
        #print("max",k,ops,ops/ref/1000000000000)
    plt.scatter(x, y, c ="black")

    t = {}
    x =[]
    y = []
    for k in gem.perfdictionary.keys():
        time = gem.perfdictionary[k]
        #print(k)
        c,P = eval(k)
        ops = gem.ops(P)
        if ops in t:  t[ops] = min(t[ops],ops/time/1000000000000)
        else:         t[ops] = ops/time/1000000000000
    for k,v in t.items():
        x.append(k)
        y.append(v)
    plt.scatter(x, y, c ="red")
        
    x = []
    y = []
    for k in gem.best.keys():
        problem, time = gem.best[k]
        ops = gem.ops(eval(k))
        x.append(ops)
        y.append(ops/time/1000000000000)
        #print(k,ops,ops/time/1000000000000)
    plt.scatter(x, y, c ="blue")

        

    plt.xlabel("OPS")
    plt.ylabel("TFLOPS")
    plt.title("4x2 CONV bits")
    #plt.show()
    plt.savefig("conv.png")
    plt.show()



if __name__ == "__main__":

    #test_gemm()
    #plot()
    #validate([16, 96, 96, 8, 8, 8])

    #test_mha_2()
    #plot_mha()

    test_conv()
    plot_conv()
