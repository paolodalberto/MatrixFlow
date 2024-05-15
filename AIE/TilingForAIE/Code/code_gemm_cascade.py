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

Frequency = (1)*10**9    # 1GHz
Bandwidth = 4 * 8*(2**30) # 4 GBs
BandwidthCore = 4 * 8*(2**30) # 4 GBs

import os 
from code_gemm import Gemm
import pickle


###
##     The problem size for a GEMM is [m,n,k,atypem, btype, ctype)
##     C = m x n  ctype in bits
##     A = m x k  atype in bits
##     B = k x n  atype in bits 
###

class GemmCascade(Gemm):

    def __init__(
            self,
            memtilesizeinbits : int = (512*2**10)*8,
            ROWS               : int = 4 ,
            COLS               : int = 2 ,
            CoreCSpacebits     : int = 2*(8*2**10)*8, ## 2 banks    
            CoreASpacebits     : int = 2*(8*2**10)*8, ## 2 banks        
            CoreBSpacebits     : int = 2*(8*2**10)*8, ## 2 banks        
            m_align : int = 16,
            n_align: int  = 16,
            k_align: int  = 16,
            frequency : int = Frequency, 
            to_core_channel_bandwidth_gbits : int = BandwidthCore,  # 4GBs 
            to_mem_channel_bandwidth_gbits  : int = Bandwidth,   # 4GBs
            memory : bool = True,
            Mtrickery : list = None
    ):

        Gemm.__init__(self,
            memtilesizeinbits,ROWS,COLS,
            CoreCSpacebits,CoreASpacebits,CoreBSpacebits,
            m_align,n_align,k_align,frequency,
            to_core_channel_bandwidth_gbits,to_mem_channel_bandwidth_gbits)
        
        self.Mtrickery = None
        if Mtrickery and len(Mtrickery)>0:
            self.Mtrickeri = [i for i in Mtrickery]

    def minimum_computation_time_clu(
            self,
            P : list
    ):

        Comp = self.compute(P)
        SpaceA= self.space_m([P[0],P[2],P[3]])
        SpaceC= self.space_m([P[0],P[1],P[5]])
        
        Ncores = self.COLS*self.ROWS
        
        comptime = Comp/Ncores/self.core_frequency
        comttime = (SpaceA+SpaceC)/(2*min(self.COLS, self.ROWS)*self.tocorebandwidth) 
        
        
        return max(comptime,comttime)

    def compute(self,x,t : bool = False):
        m,n,k,atype,btype,ctype = x
        
        B = 256
        
        
        if   ctype == 8 :         B = 256
        elif ctype == 16:         B = 128
        elif ctype == 32 :        B = 32

        if t and self.Mtrickery:
            mm,nn,kk = self.Mtrickery
            print(mm,m)
            m = max(m,mm)

            #n = max(n,nn)
            #k = max(k,kk)
            
        return math.ceil(2*m*n*k/B)


    def memsubvolume(
            self,
            coresubvolume : list,
            P : list,
    ):

        MA,NA,KA,_,_,_ = P
        ## C Tall
        subvolume    =  [ i for i in coresubvolume]
        subvolume[0] =  MA
        subvolume[1] =  subvolume[1]*self.COLS
        subvolume[2] =  subvolume[2]*self.ROWS

        return subvolume

    def get_time_cluster_with_latency_ax(self, memsub : list, core : list, ax : int = 0,
                                         c : int = 1,a : int = 1, b : int = 1 ):

        # each core compute independently a sub volume, we coutn the
        # number of cores
        Ncores = self.COLS*self.ROWS

        ## B will stay in core and A we pay the first round 
        start = self.opspace(core)/(self.tocorebandwidth)
        #             Total cycles / By cores = cycle per core
        #             Cycles per core / frequency  = Time

        times = math.ceil(memsub[ax]/core[ax])
        
        computetime = (times-1)*self.compute(core,True)/self.core_frequency
        #   Byte per channel =    Bytes to read / number of channels two columns and two rows ~ 4 channels
        #           byte per channel / bandwidth = time
        # we read A row and write C
        commtime    = (times-1)*(c*self.space_m([core[0], core[1], core[-1]])+
                                 a*self.space_m([core[0], core[2], core[3]])+
                                 b*self.space_m([core[1], core[2], core[4]])
                                 )/(self.tocorebandwidth)

        ## last compute + write = reduction
        end = self.compute(core)/self.core_frequency + c*self.space_m([core[0], core[1], core[-1]])/(self.tocorebandwidth)
        
        Mtime = max(
            computetime,
            commtime 
        )


        time = start+Mtime+end
        ref = self.minimum_computation_time_clu(memsub)

        if ref>time:
            print(computetime,commtime,ref)
            pdb.set_trace()
            ref = self.minimum_computation_time_clu(memsub)
            
        return time



    ## estimate ddr_ddr_ time estimate given a memtile subproblem
    def time_estimates(
            self,
            Tsub : list , ## memtile subproblem
            P    : list , ## aligned problem size
            v : bool = False,
            core : list = None
    ):

        ## this is Memtile space in Bytes (the unit of every opspace
        Mem = self.Mem//8 # bytes

        MA,NA,KA,atype, btype, ctype = P
        #pdb.set_trace()
        norm    = False  # self.opspace(Tsub) < Mem 
        dbb     = self.opspace_dbb(Tsub) < Mem
        dba     = self.opspace_dba(Tsub) <= Mem
        if  self.opspace_dba(Tsub)<self.opspace_dbb(Tsub):
            if dbb:
                dba=False
        dba_dbb = False  # self.opspace_dba_dbb(Tsub) < Mem 
        Msplit = Tsub[0]<MA


        TotalTime = 0
        debug = []
        # we stream A and C, B in Cores
        TT = self.get_time_cluster_with_latency_ax(Tsub,core,ax =0, c=0, a=0 if dbb else 1, b= 0 if dba else 0) #a = 1 if dba else 0,b = 1 if dbb else 0)
                
        if dba :
                        
            ## 
            TotalTime = 0
            K = 0 
            
            K += (self.space_m([P[1],P[2],P[4]])/(self.ddrchannels*self.tomembandwidth))
            K += (self.space_m([core[1],core[2],core[4]])/(self.tocorebandwidth))
            TotalTime += K
            
            for j in range(0, NA, Tsub[1]):

                ## To be a valid and meaningful  mk should  KA because of the reduction 
                for k in range(0, KA, Tsub[2]):
                    ## we move B into CORE
                    debug.append([j,k,"l",K])
                    

                    K = 0
                    # LOAD A, (compute C, load A)[n-1] compute C
                    K  += \
                        self.space_m([Tsub[0],Tsub[2],Tsub[3]])/(self.ddrchannels*self.tomembandwidth) +\
                        max (
                            TT,
                            (self.space_m([Tsub[0],Tsub[2],Tsub[3]])/(self.ddrchannels*self.tomembandwidth))
                        )* (MA//Tsub[0] -1 ) \
                        + TT 
                    
                    debug.append([j,k,"c",K])
                    TotalTime += K 


                Red = (self.ROWS-1)*(self.space_m([core[0],core[1],32])/(self.tocorebandwidth))
                Red += (self.space_m([core[0],core[1],core[5]])/(self.tocorebandwidth))
                
                K = (Tsub[0]*Tsub[1])//8/(self.ddrchannels*self.tomembandwidth)  ## writing C in DDR
                TotalTime +=K + Red
                debug.append([j,k,"c",K])
                
        elif dbb :
            ## 
            #if P[0] == 64 : pdb.set_trace()
            ## we move A into CORE
            TotalTime = 0
            K = 0 
            
            K += (self.space_m([P[0],P[2],P[3]])/(self.ddrchannels*self.tomembandwidth))
            K += (self.space_m([core[0],core[2],core[3]])/(self.tocorebandwidth))
            ## To be a valid and meaningful  mk should  KA because of the reduction 
            TotalTime += K

            for j in range(0, NA, Tsub[1]):
                for k in range(0, KA, Tsub[2]):
                                
                    debug.append([j,k,"l",K])
                    

                    K = 0
                    # LOAD B, (compute C, load B)[n-1] compute C
                    K  += \
                        self.space_m([Tsub[1],Tsub[2],Tsub[4]])/(self.ddrchannels*self.tomembandwidth) +\
                        max (
                            TT,
                            (self.space_m([Tsub[1],Tsub[2],Tsub[4]])/(self.ddrchannels*self.tomembandwidth))
                        )* (MA//Tsub[0] -1 ) \
                        + TT 
                    
                    debug.append([j,k,"c",K])
                    TotalTime += K 


                Red = (self.ROWS-1)*(self.space_m([core[0],core[1],32])/(self.tocorebandwidth))
                Red += (self.space_m([core[0],core[1],core[5]])/(self.tocorebandwidth))
                
                K = (Tsub[0]*Tsub[1])//8/(self.ddrchannels*self.tomembandwidth)  ## writing C in DDR
                TotalTime +=K + Red
                debug.append([j,k,"c",K])
            

        ref = self.minimum_computation_time(P)
            
        if TotalTime ==0 or ref>TotalTime:
            
            if v: print("None", norm, dba,dbb, dba_dbb,Ksplit)

            ## non of the above fit in Memtile Time => infinity
            #TotalTime = 1000000.0
            pdb.set_trace()
            ref = self.minimum_computation_time(P)
            #print("Ksplit", Ksplit, KA//Tsub[2], "total time second ",  TotalTime)
        if v:
            for d in debug:
                print(d)
        return TotalTime, debug


    
    
    def time(
            self,
            coresubvolume : list,
            P             : list,
            memtile       : list = None,
            v : bool = False
    ):

        
        #pdb.set_trace()
        MA,NA,KA, _,_,_ = P

        #if v: pdb.set_trace()
        if memtile is None:
            subvolume = self.memsubvolume(coresubvolume,P)
        else:
            subvolume = [i for i in memtile]
        
        ## this is Memtile space in Bytes (the unit of every opspace
        Mem = self.Mem//8 # bytes
        ## we need just to stream A nad C in Mem tile to stream 
        norm    = self.opspace_ca(subvolume) < Mem 
        if norm == False:
            #pdb.set_trace()
            # we split K
            TTsub = [i for i in subvolume]
            for i in range(2, 16, 2):
                TTsub[0] = math.ceil(subvolume[0]//i)
                norm     = self.opspace_dba_dbb(TTsub) <= Mem
                if norm:
                    #pdb.set_trace()
                    subvolume =  TTsub
                    split = True
                    break
        if norm:    
            #time0 = self.time_estimates(subvolume,P)
            time0,deb = self.time_estimates(subvolume,P,v=v,core=coresubvolume)
        else:
            
            time0,deb  = 1000000, []
        return time0,deb
            

        
    
        
                            
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

        print("GEMM Problem M,N,K,atype,btype,ctype ", [M,N,K, atype,btype,ctype])

        
        ## ping pong space for the core (there are two banks we use
        ## only one streaming A nd B, Notice the space is normalized
        ## as number of elements more than bits.  We assume here two
        ## banks per tensor.

        #Space for C is different .. we may acumulate directly in 32
        #bits which is different

        scaling = 8  ## accumulation is done in 32
        SC = self.cspace//scaling #len(ofm.banks) we don't



        SA = self.aspace//atype//2
        SB = self.bspace//btype

        ## this is Memtile space in Bytes (the unit of every opspace
        Mem = self.Mem//8 # bytes
        
        # the alignement for input 
        MA = math.ceil(M/ self.m_align)* self.m_align
        NA = math.ceil(N/self.n_align)* self.n_align
        ali = self.k_align
        KA = math.ceil(K/  ali)* ali

        P = [MA,NA,KA,atype,btype,ctype]
        print("ALIGNED", P)
        ## number of aligned blocks 
        mt = MA// self.m_align
        nt = NA// self.n_align
        kt = KA// ali

        ## all valid stream using ping pong at  core level
        MM = 1
        MMl = []
        scaling = 32/ctype  ## accumulation is done in 32
        
        for m in range(1,mt + 1 ) :
            cm = m*self.m_align
            for n in range(1,nt+1): 
                cn = n*self.n_align
                for k in range(1,kt + 1):
                    kn = k*ali
                    #print(cm*cn,SC,cm*kn,SA, kn*cn, SB)
                    if  cm*cn <= SC and cm*kn <= SA and kn*cn <= SB :
                        # These are valid solutions, we count the
                        # number of elements per core memory (as
                        # number of elements)
                        q =  [cm, cn,kn, atype,btype,ctype]

                        #if cm ==32 and cn == 32 and kn == 128:
                        #    pdb.set_trace()
                        #subvolume = self.memsubvolume(q,P)
                        #if (subvolume[0] > P[0] or subvolume[1] > P[1] or subvolume[2] > P[2]): continue
                        #if self.opspace_ca(subvolume)>Mem: continue 
                        #if subvolume[2] != P[2] : continue

                        MMl.append(q)

                        """
                        if expsub is not None: 
                            ## now we explore the subvolume that make
                            ## possible to double buffer the larger matrix
                            
                            if expsub[0] in [0,1]:
                                C1subvolume = self.memsubvolume(q,P,expsub[0])
                                if self.opspace_dba(C1subvolume)<Mem:
                                    MMl.append(q)
                            elif expsub[0] ==2:
                                ## Q1 Tall C
                                C1subvolume    =   [ i for i in q]
                                As = MA% q[0]==0  and NA % q[1]==0 and KA % q[2] ==0
                                
                                #if self.opspace_dbb(C1subvolume) <Mem and As:
                                if  As:
                                    MMl.append(q)
                            else:
                                MMl.append(q)
                        
                        
                        else:
                            MMl.append(q)
                        
                        """
                        #print(MM, MMl)
        #print(MM,MMl)
        #pdb.set_trace()
        ## maximum computation and then by maximum ratio
        ## Ratio is compute / A+B operand space in bytes 
        #        W = sorted( MMl, key= lambda x: ( -self.compute(x)
        #                                          ,-self.ratio(x)
        #                                          #,self.perimeter(x)
        #                                         )
        #                   )
        
        
        
        #        print("TOP 3 Time, compute, ratio ")
        #        for w in W[0:3]: print(w,self.time(w,P),self.compute(w),self.ratio(w),w)
        
        W1 = sorted( MMl, key= lambda x: ( self.time(x,P),-self.compute(x),-self.ratio(x) ))
        #        pdb.set_trace()
        #        print("TOP 3 Time, compute, ratio ")
        #        for w in W1[0:3]: print(w,self.time(w,P),self.compute(w),self.ratio(w),w)
        
        #        pdb.set_trace()
        
        if len(W1) ==0:
            pdb.set_trace()
        coresubvolume = W1[0]
        Tsub = self.memsubvolume(coresubvolume,P)
        time,_ = self.time(coresubvolume,P)

        #pdb.set_trace()

        dbb     = self.opspace_dbb(Tsub) < Mem
        dba     = self.opspace_dba(Tsub) < Mem
        dba_dbb = self.opspace_dba_dbb(Tsub) < Mem 

        Ksplit = Tsub[0]<MA




        return Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,0, "", W1, time



    

def test_gemm(cols=2,actbit=16,weibit=8) :
    gem = GemmCascade(COLS=cols)
    Times = []



    M, N, K = [ 64,2048, 8192 ]
    
    #M,N,K = [512,768,768]
    Ref = gem.minimum_computation_time([M,N,K,actbit,weibit,actbit])
    print(Ref, M*K*N*2/Ref/10**12)
    
    
    pdb.set_trace()
    RR = gem.gen_fm_par_fm_([M,N,K,actbit,weibit,actbit])
    Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q,  W,time  = RR
    print("## P", [M,N,K],Ref)
    print("## Time", time, "padding M", M%coresubvolume[0], "N", N % coresubvolume[1], "K", K % coresubvolume[2])
    print("## xslowdown", time/Ref, Tsub, coresubvolume)
    print("## TFLOPS", Ref, M*K*N*2/time/10**12)
    Mod = M%coresubvolume[0]!=0 or  N % coresubvolume[1] != 0 or  K % coresubvolume[2]!=0
    Times.append([[M,N,K],Tsub, coresubvolume,Mod,Ksplit, FatC,time])


    D  = [
        [ 32,32,128, 8,8,8] ,
        [ 32,64,128, 8,8,8] ,
        [ 64,64,128, 8,8,8]
    ]

    pdb.set_trace()
    t, r = gem.time(W[0],[M,N,K,actbit,weibit,actbit],v=True)
    for d in D:
        if d in W:
            time, _ = gem.time(d,[M,N,K,actbit,weibit,actbit])
            print(d, M*K*N*2/time/10**12)
            print(time)
            
    pdb.set_trace()
            

    if Mod:
        RR = gem.gen_fm_par_fm_([M,N,K,actbit,weibit,actbit],[2])
        Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q,  W,time  = RR
        print("## P", [M,N,K],Ref)
        print("## Time", time, "padding M", M%coresubvolume[0], "N", N % coresubvolume[1], "K", K % coresubvolume[2])
        print("## xslowdown", time/Ref)
        print("## TFLOPS", Ref, M*K*N*2/time/10**12)
        Mod = M%coresubvolume[0]!=0 or  N % coresubvolume[1] != 0 or  K % coresubvolume[2]!=0
        Times.append([[M,N,K],Tsub, coresubvolume,Mod,Ksplit, FatC,time])
    pdb.set_trace()
    return W,gem




"""

static constexpr uint32_t L1_M = 8;
static constexpr uint32_t L1_K = 128;
static constexpr uint32_t L1_N = 64;

static constexpr uint32_t L2_M = 8;
static constexpr uint32_t L2_K = 512;
static constexpr uint32_t L2_wgt_K = 2048;
static constexpr uint32_t L2_N = 64;

static constexpr uint32_t L3_M = 8;
static constexpr uint32_t L3_K = 2048;
static constexpr uint32_t L3_N = 2048;


       M       K       N  group_size  Executetime(ns)  run_aie_time(ns)  A_copy_time(ns)  A_sync_time(ns)  C_copy_time(ns)  C_sync_time(ns)      ratio           OPS    TFLOPS  I_TFLOPS  II_TFLOPS    III_TFLOPS  IV_TFLOPS
0    1.0  2048.0  2048.0        32.0        2227950.5         2178850.0           1900.0           9500.0           1750.0          22400.0   2.203842  8.388608e+06  0.003850  0.174203   0.175702  8.388608e-10   0.274610
1    1.0  2048.0  8192.0        32.0        1013400.0          986900.0           1000.0           8600.0           1200.0          14250.0   2.614960  3.355443e+07  0.034000  0.174203   0.175702  3.355443e-09   0.274710
2    1.0  8192.0  2048.0        32.0         979500.0          951950.0           1050.0           8300.0            600.0           8550.0   2.812660  3.355443e+07  0.035248  0.174206   0.175722  3.355443e-09   0.274710
3    8.0  2048.0  2048.0        32.0         595250.0          562100.0           3800.0          10900.0           2600.0          13800.0   5.569089  6.710886e+07  0.119390  1.381467   1.405613  6.710886e-09   2.181977
4    8.0  2048.0  8192.0        32.0        1084700.0         1019400.0           4050.0          13000.0           9200.0          38750.0   6.020098  2.684355e+08  0.263327  1.381467   1.405613  2.684355e-08   2.188338
5    8.0  8192.0  2048.0        32.0        1029650.0          972750.0          12800.0          18850.0           2400.0          13450.0   5.526150  2.684355e+08  0.275955  1.381626   1.405777  2.684355e-08   2.188338
6   16.0  2048.0  2048.0        32.0         595750.0          558600.0           3700.0           8450.0           5400.0          11200.0   6.235837  1.342177e+08  0.240275  2.050809   2.088857  1.342177e-08   4.096000
7   16.0  2048.0  8192.0        32.0        1005200.0          937500.0           4400.0          13750.0          15400.0          35250.0   6.734978  5.368709e+08  0.572662  2.050809   2.088857  5.368709e-08   4.096000
8   16.0  8192.0  2048.0        32.0        1083400.0         1015050.0          15200.0          27000.0           5750.0          19550.0   6.308843  5.368709e+08  0.528911  2.051159   2.089220  5.368709e-08   4.096000
9   32.0  2048.0  2048.0        32.0         631850.0          584850.0           5700.0          13750.0           9800.0          17600.0   7.438474  2.684355e+08  0.458982  2.706471   2.759539  4.079131e+00   4.096000
10  32.0  2048.0  8192.0        32.0        1095100.0         1000950.0           5000.0          14250.0          31350.0          40100.0   8.597388  1.073742e+09  1.072723  2.706471   2.759539  4.084305e+00   4.096000
11  32.0  8192.0  2048.0        32.0        1131350.0         1054950.0          21000.0          24200.0          14700.0          16000.0   6.752994  1.073742e+09  1.017813  2.707081   2.760173  4.088541e+00   4.096000
12  64.0  2048.0  2048.0        32.0         612700.0          558750.0           6400.0           9200.0          17200.0          11200.0   8.805288  5.368709e+08  0.960843  3.221432   3.289672  4.084903e+00   4.096000
13  64.0  2048.0  8192.0        32.0        1219900.5         1092450.0           7100.0          14350.5          63100.0          40599.5  10.447614  2.147484e+09  1.965750  3.221432   3.289672  4.089057e+00   4.096000
14  64.0  8192.0  2048.0        32.0        1119300.0         1039500.0          24650.0          26750.0          16800.0          18100.0   7.129456  2.147484e+09  2.065881  3.222297   3.290574  4.090398e+00   4.096000
"""

def test_gemm_f(cols=4,actbit=8,weibit=8) :
    gem = GemmCascade(COLS=cols, m_align=8)
    Times = []

    P0 = [ 16,  8192,  2048 ,8,8,8]
    P1 = [ 32,  2048,  2048 ,8,8,8]

    core = [8,64,128,8,8,8]
    
    #M,N,K = [512,768,768]
    Ref = gem.minimum_computation_time(P0)
    print(Ref, P0[0]*P0[1]*P0[2]*2/Ref/10**12)
    Ref = gem.minimum_computation_time(P1)
    print(Ref, P1[0]*P1[1]*P1[2]*2/Ref/10**12)
    
    
    pdb.set_trace()
    time0, d0 = gem.time(core,P0);
    print(time0, P0[0]*P0[1]*P0[2]*2/time0/10**12)
    time1, d1 = gem.time(core,P1)
    print(time1, P0[0]*P0[1]*P0[2]*2/time1/10**12)
    pdb.set_trace()




    
    return W,gem

    
if __name__ == "__main__":

#    test_gemm(4,16,8)
    test_gemm_f(4,8,8)
