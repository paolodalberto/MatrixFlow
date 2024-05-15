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

Frequency = (1)*10**9     # 1 GHz
Bandwidth = 1 * 8*(2**30) # 4 GBs


###
##     The problem size for a GEMM is [m,n,k,atypem, btype, ctype)
##     C = m x n  ctype in bits
##     A = m x k  atype in bits
##     B = k x n  atype in bits 
###
class Gemm():

    ## memtilesizeinbits : MemTile Size in Bits 
    ## Overlay Rows 
    ## Overlay Columns
    ## Space for C in bits
    ## Space for A in bits
    ## Space for B in bits
    ## alignment on the m dimension of the problem
    ## alignment on the n dimension of the problem
    ## alignment on the k dimension of the problem
    ## frequency of the core in Hz
    ## bandwidth per channel from memtile to core
    ## bandwidth per channel from DDR to memtile 
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
            to_mem_channel_bandwidth_gbits  : int = Bandwidth   # 4GBs
            

    ):
        self.Mem = memtilesizeinbits*COLS
        self.ROWS = ROWS
        self.COLS = COLS
        self.cspace = CoreCSpacebits
        self.aspace = CoreASpacebits
        self.bspace = CoreBSpacebits
        self.m_align =  m_align
        self.n_align =  n_align
        self.k_align =  k_align
        self.core_frequency = frequency
        self.tocorebandwidth = to_core_channel_bandwidth_gbits 
        self.tomembandwidth  = to_mem_channel_bandwidth_gbits 
        self.ddrchannels = self.COLS*2
        self.perfdictionary = {}


    ## element wise computation estimate we work with only matrices
    ## and thus this is a matrix element wise operation
    ## m,n,ctype = x    
    def element_compute(self,x):
        m,n,ctype = x

        ops = m*n

        B = 256
        if   ctype == 8 :         B = 256
        elif ctype == 16:         B = 128
        elif ctype == 32 :        B = 64
        
        return math.ceil(ops/B)

    ## problem [m,n,k, abyte, bbyte, ctype] = P max between ideal time
    ## and ideal communication time
    def minimum_computation_time(
            self,
            P : list
    ):

        Comp = self.compute(P)
        Space= self.space(P)

        Ncores = self.COLS*self.ROWS

        comptime = Comp/Ncores/self.core_frequency
        comttime = Space/(2*(min(self.COLS, self.ROWS))*self.tomembandwidth) #+ \
            #Space/(3*(min(self.COLS, self.ROWS))*self.tocorebandwidth) 
        
        
        return max(comptime,comttime)
        
    def __str__(self):
        A = "ROWS : %d  COLS %d" % (self.ROWS, self.COLS)
        B = "MemTile  %d MB"  % (self.Mem//8//1024//1024)
        C = "Core Space C %d  KB A %d  KB B %d  KB" %  ( self.cspace//8//1024 ,self.aspace//8//1024, self.bspace//8//1024)
        return A +"\n" + B + "\n" + C + "\n"  
        
    ## Tall C algorithm
    def generate_Q1(self):
        C1 = self.generate_C1()
        A1 = self.generate_A1()
        B1 = self.generate_B1()
        L = C1 + "sum_k (" + A1 + "*" + B1 + ")" 

        return L
    ## Tall C algorithm C 
    def generate_C1(self):
        lines = ""
        for i in range(self.COLS):
            line = "|"
            for j in range(0,self.ROWS):
                line += "Ci+%d,j+%d " %(i,j)
            line += "|\n"
            lines += line
        return lines
    ## Tall C algorithm A 
    def generate_A1(self):
        lines = ""
        for i in range(self.COLS):
            line = "|"
            line += "Ai+%d,k " %(i)
            line += "|\n"
            lines += line
        return lines
    ## Tall C algorithm B 
    def generate_B1(self): 
        line = "|"
        for j in range(self.ROWS):
            line += "Bk,j+%d " %(j)
        line += "|\n"
        return line


    ## Fat C algorithm 
    def generate_Q0(self):
        C1 = self.generate_C0()
        A1 = self.generate_A0()
        B1 = self.generate_B0()
        L = C1 + "sum_k (" + A1 + "*" + B1 + ")" 

        return L
    ## Fat C algorithm C
    def generate_C0(self):
        lines = ""
        for j in range(0,self.ROWS):
            line = "|"
            for i in range(self.COLS):
                line += "Ci+%d,j+%d " %(i,j)
            line += "|\n"
            lines += line
        return lines
    
    ## Fat C algorithm A
    def generate_A0(self):
        lines = ""
        for i in range(self.ROWS):
            line = "|"
            line += "Ai+%d,k " %(i)
            line += "|\n"
            lines += line
        return lines

    ## Fat C algorithm B
    def generate_B0(self): 
        line = "|"
        for j in range(self.COLS):
            line += "Bk,j+%d " %(j)
        line += "|\n"
        return line

    ## Q ==1  Tall C algorithm
    ## Q ==0  Fat  C algorithm
    def generate_Q(self, Q : int ):
        if Q == 0   : return self.generate_Q0()
        elif Q == 1 : return self.generate_Q1()
        return " NONE "
        

        
    ## space in Bytes for C, A, and B 
    def space(self,x):
        m,n,k,atype,btype,ctype = x
        return (ctype*m*n//8 +atype*m*k//8 + btype*k*n//8)
    ## space in bytes for a matrix A [m,n,type]
    def space_m(self,A):
        m,n,atype = A
        return atype*m*n//8
    ## space in elements for a matrix A [m,n,type]
    def space_e(self,A):
        m,n,atype = A
        return m*n
    
    ## the A and B operand space in Bytes but B is double buffered
    ## this could be used for memtile space estimate
    def opspace_dbb(self,x):
        m,n,k,atype,btype,ctype = x
        return (atype*m*k//8 + 2*btype*k*n//8)

    ## the A and B operand space in Bytes and both A and B is double
    ## buffered this could be used for memtile space estimate. 
    def opspace_dba_dbb(self,x):
        m,n,k,atype,btype,ctype = x
        return (2*atype*m*k//8 + 2*btype*k*n//8)

    ## the A and B operand space in Bytes but A is double buffered
    ## this could be used for memtile space estimate
    def opspace_dba(self,x):
        m,n,k,atype,btype,ctype = x
        return (2*atype*m*k//8 + btype*k*n//8)

    ## the A and B operand space in Bytes
    def opspace(self,x):
        m,n,k,atype,btype,ctype = x
        return (atype*m*k//8 + btype*k*n//8)

    ## the A and C operand space in Bytes
    def opspace_ca(self,x):
        m,n,k,atype,btype,ctype = x
        return (atype*m*k//8 + ctype*m*n//8)

    ## Perimeters in number of elements
    def perimeter(self,x):
        m,n,k,atype,btype,ctype = x
        return (m+n+k)


    ## number of cycles for 2*n*m*k operations where the final type
    ## describes the effective number of operations per cycle
    ## [m,n,k,atype,btype,ctype] = x
    def compute(self,x):
        m,n,k,atype,btype,ctype = x
        B = 256
        if   ctype == 8 :         B = 256
        elif ctype == 16:         B = 128
        elif ctype == 32 :        B = 64
        
        return math.ceil(2*m*n*k/B)

    ## the ratio is the comparison between computation (cycles) and
    ## communication in Bytes for the input operands. With the proper
    ## dimensions and multiplicative factors we can compute TIME we
    ## want to find a tiling where we maximize this ratio or minimize
    ## the estimate time
    def ratio(self,x):
        return self.compute(x)/self.opspace(x)

    ## take a subvolume/computation and an overlay. Assume this will
    ## be in MemTile. What will be the time estimate to solve this
    ## computation?
    def get_time_cluster(self, sub):

        # each core compute independently a sub volume, we coutn the
        # number of cores
        Ncores = self.COLS*self.ROWS

        #             Total cycles / By cores = cycle per core
        #             Cycles per core / frequency  = Time 
        computetime = self.compute(sub)/Ncores/self.core_frequency
        
        #   Byte per channel =    Bytes to read / number of channels two columns and two rows ~ 4 channels
        #           byte per channel / bandwidth = time             
        commtime  = self.opspace(sub)/(2*(min(self.COLS, self.ROWS))*self.tocorebandwidth)
        #print(computetime,commtime)
        Mtime = max(
            computetime,
            commtime 
        )
        return Mtime

    ## take a subvolume/computation and an overlay. Assume this will
    ## be in MemTile. What will be the time estimate to solve this
    ## computation? Now we pay for the prolog and epilog
    def get_time_cluster_with_latency(self, memsub : list, core :
                                      list):

        # each core compute independently a sub volume, we coutn the
        # number of cores
        Ncores = self.COLS*self.ROWS
        
        start = self.opspace(core)/(2*self.tocorebandwidth)
        #             Total cycles / By cores = cycle per core
        #             Cycles per core / frequency  = Time
        
        computetime = (math.ceil(memsub[2]/core[2])-1)*self.compute(core)/self.core_frequency
        #   Byte per channel =    Bytes to read / number of channels two columns and two rows ~ 4 channels
        #           byte per channel / bandwidth = time             
        commtime    = (math.ceil(memsub[2]/core[2])-1)*self.opspace(core)/(2*self.tocorebandwidth)

        end = self.compute(core)/self.core_frequency +  self.space_m([core[0],core[1],core[5]])/self.tocorebandwidth
        
        Mtime = max(
            computetime,
            commtime 
        )
        return start+Mtime+end

    ## as other get_time_cluster computation estimates byt we
    ## introduce an Axe that we stream. this is not used in this class
    ## but in the next ones
    def get_time_cluster_with_latency_ax(self, memsub : list, core : list, ax : int =2 ):

        # each core compute independently a sub volume, we coutn the
        # number of cores
        Ncores = self.COLS*self.ROWS
        
        start = self.opspace(core)/(2*self.tocorebandwidth)
        #             Total cycles / By cores = cycle per core
        #             Cycles per core / frequency  = Time

        times = math.ceil(memsub[ax]/core[ax])
        
        computetime = (times-1)*self.compute(core)/self.core_frequency
        #   Byte per channel =    Bytes to read / number of channels two columns and two rows ~ 4 channels
        #           byte per channel / bandwidth = time             
        commtime    = (times-1)*self.opspace(core)/(2*self.tocorebandwidth)

        end = self.compute(core)/self.core_frequency
        
        Mtime = max(
            computetime,
            commtime 
        )
        return start+Mtime+end




    ## estimate ddr_ddr_ time estimate given a memtile subproblem
    ##  MA,NA,KA,atype, btype, ctype = P     main problem
    ##  mm,ma,mk,atype, btype, ctype = Tsub  memtile problem
    ##  cm,ca,ck,atype, btype, ctype = core  core problem
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
        norm    = self.opspace(Tsub) < Mem 
        dbb     = self.opspace_dbb(Tsub) < Mem
        dba     = self.opspace_dba(Tsub) < Mem
        dba_dbb = self.opspace_dba_dbb(Tsub) < Mem 
        Ksplit = Tsub[2]<KA


        TotalTime = 0 
        TT = self.get_time_cluster(Tsub) if core is None else self.get_time_cluster_with_latency(Tsub,core)

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
                        )* (math.ceil(KA/Tsub[2]) -1 ) \
                        + TT
                    
                    TotalTime += self.space_m([Tsub[0],Tsub[1],Tsub[5]])/(self.COLS*self.tocorebandwidth) ## writing C in Memtile
                    TotalTime += self.space_m([Tsub[0],Tsub[1],Tsub[5]])/(self.ddrchannels*self.tomembandwidth)  ## writing C in DDR
                
        elif dba_dbb:
            if v: print("dba_dbb")
            
            ## We double buffer both A and B
            ## first A
            tick = self.space_m([Tsub[0],Tsub[2],Tsub[3]])/(self.ddrchannels*self.tomembandwidth)

            TotalTime = tick
            
            for i in range(0, MA, Tsub[0]):
                # iteration A
                
                ## load B ( compute C , write C, load A) 
                CTime = self.space_m([Tsub[1],Tsub[2],Tsub[4]])/(self.ddrchannels*self.tomembandwidth) + \
                    max(
                        TT,
                        self.space_m([Tsub[1],Tsub[2],Tsub[4]])/(self.ddrchannels*self.tomembandwidth)
                    )* (math.ceil(NA/Tsub[1])-1) + \
                    TT 
        
                TotalTime += max(CTime,tick)
                TotalTime += self.space_m([Tsub[0],Tsub[1],Tsub[5]])/(self.COLS*self.tocorebandwidth) ## writing C in Memtile
                TotalTime += self.space_m([Tsub[0],Tsub[1],Tsub[5]])/(self.ddrchannels*self.tomembandwidth)  ## writing C in DDR
        elif dbb :
            if v: print("dbb")
        
            # We stream A by DB
            TotalTime = 0
            for i in range(0, MA, Tsub[0]):
                ## LOAD A_i once and 
                TotalTime += self.space_m([Tsub[0],Tsub[2],Tsub[3]])/(self.ddrchannels*self.tomembandwidth)
                
                ## We DB B with compute ... 
                ## load B ( compute C , load B) 
                TotalTime += self.space_m([Tsub[1],Tsub[2],Tsub[4]])/(self.ddrchannels*self.tomembandwidth) + \
                    max(
                        TT,
                        self.space_m([Tsub[1],Tsub[2],Tsub[4]])/(self.ddrchannels*self.tomembandwidth)
                    )* (math.ceil(NA/Tsub[1])-1) + \
                    TT
                TotalTime += self.space_m([Tsub[0],Tsub[1],Tsub[5]])/(self.COLS*self.tocorebandwidth) ## writing C in Memtile
                TotalTime += self.space_m([Tsub[0],Tsub[1],Tsub[5]])/(self.ddrchannels*self.tomembandwidth)  ## writing C in DDR
                #print(TotalTime)
            #pdb.set_trace()
        elif  dba :
            if v: print("dba")
            # We stream A by DB
        
            TotalTime = 0
            for j in range(0, NA, Tsub[1]):
                ## load B_j once and 
                TotalTime += self.space_m([Tsub[1],Tsub[2],Tsub[4]])/(self.ddrchannels*self.tomembandwidth)
        
                ## load A ( compute C , load A) 
                TotalTime += self.space_m([Tsub[0],Tsub[2],Tsub[3]])/(self.ddrchannels*self.tomembandwidth) + \
                    max(
                        TT,
                        self.space_m([Tsub[0],Tsub[2],Tsub[3]])/(self.ddrchannels*self.tomembandwidth)
                    )* (math.ceil(MA/Tsub[0])-1) + \
                    TT
                
                TotalTime += self.space_m([Tsub[0],Tsub[1],Tsub[5]])/(self.COLS*self.tocorebandwidth) ## writing C in Memtile
                TotalTime += self.space_m([Tsub[0],Tsub[1],Tsub[5]])/(self.ddrchannels*self.tomembandwidth)  ## writing C in DDR
        
        if TotalTime ==0:
            if v: print("None", norm, dba,dbb, dba_dbb,Ksplit)

            ## non of the above fit in Memtile Time => infinity
            TotalTime = 1000000.0
            #pdb.set_trace()
            #print("Ksplit", Ksplit, KA//Tsub[2], "total time second ",  TotalTime)

        return TotalTime

    
    ##  MA,NA,KA,atype, btype, ctype = P     main problem
    ##  cm,ca,ck,atype, btype, ctype = coresubvolume problem
    ##  Q =1 Tall C
    ##  Q =0 Fat C
    ##  return mem tile subproblem  
    def memsubvolume(
            self,
            coresubvolume : list,
            P : list,
            Q : int
    ):

        MA,NA,KA,_,_,_ = P
        if Q == 0 :
            ## C Tall
            subvolume    =  [ i for i in coresubvolume]
            subvolume[0] =  subvolume[0]*self.ROWS
            subvolume[1] =  subvolume[1]*self.COLS
            subvolume[2] =  KA
        else:
            ## C Fat
            subvolume    =  [ i for i in coresubvolume]
            subvolume[0] =  subvolume[0]*self.COLS
            subvolume[1] =  subvolume[1]*self.ROWS
            subvolume[2] =  KA
        return subvolume


    ##  MA,NA,KA,atype, btype, ctype = P     main problem
    ##  cm,ca,ck,atype, btype, ctype = coresubvolume problem
    ##  Q =1 Tall C
    ##  Q =0 Fat C
    ## time estimate
    def time(
            self,
            coresubvolume : list,
            P             : list,
            Q             : int = 0 ## 0 C is Tall and 1 C is Fat COLS<ROWS
    ):
        
        
        MA,NA,KA, _,_,_ = P
        subvolume = self.memsubvolume(coresubvolume,P,Q)
        
        if subvolume[0]>MA or subvolume[1]>NA or subvolume[2]>KA:
            
            return 10000
        #else:
        #    pdb.set_trace()
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
        time0 = self.time_estimates(subvolume,P,core=coresubvolume)
        
        return time0
            
    ## validation using python 
    def ddr_computation(
            self,
            A : numpy.array, ## operand 
            B : numpy.array, ## operand 
            C : numpy.array, ## operand
            x : list ,       ## problem size
            core : list ,    ## core subvolume
            mem  : list ,    ## memsubvolume
    ):

        M,N,K, Atype, Btype, Ctype = x  
        coresubvolume = [ i for i in core]
        mh = coresubvolume[0]
        nh = coresubvolume[1]
        kh = coresubvolume[2]
      
        Tsub          = [ i for i in mem]
      
        # the alignement for input 
        MA = math.ceil(M/ self.m_align)* self.m_align
        NA = math.ceil(N/self.n_align)* self.n_align
        ali = self.k_align
        KA = math.ceil(K/  ali)* ali

        ## number of aligned blocks 
        mt = MA// self.m_align
        nt = NA// self.n_align
        kt = KA// ali
  
        CR = numpy.matmul(A,B)
        

        
        for j in range(0, NA,Tsub[1]):
            cjr = min(j+Tsub[1], NA)
            cjl = min(j, NA)
            for i in range(0,MA , Tsub[0]):
                cir = min(i+Tsub[0], MA)
                cil = min(i, MA)
                
                k = 0
                ckr = min(k+Tsub[2], KA)
                ckl = min(k, KA)
                C[cil: cir,cjl  : cjr ] = numpy.matmul(A[cil: cir, ckl:ckr], B[ckl:ckr, cjl : cjr])
                for k in range(Tsub[2], KA, Tsub[2]):
                    ckr = min(k+Tsub[2], KA)
                    ckl = min(k, KA)
                    C[cil: cir,cjl  : cjr ] += numpy.matmul(A[cil: cir, ckl:ckr], B[ckl:ckr, cjl : cjr])
                    
                print(sum(sum(C[cil: cir,cjl  : cjr ] -CR[cil: cir,cjl  : cjr ])))
        #pdb.set_trace()
        
        return CR, C
        
        
    
        
                            
    ## Here we create the solution space and use some cost function for its exploration 
    ## M, N, K, atype, btype, ctype = X
    ## C =MxN (ctype). A = MxK (atype).  B = KxN (btype)
    ## exsub is to narrow the search
    ## scaling: to count the number of elements that we can allocate in core for C (extended to 32bit)
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
        print("ALIGNED", P)
        if False and str(P) in  self.perfdictionary:
            Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q, TotalTime        =    self.perfdictionary[str(P)]
            return Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q, [], TotalTime
        ## number of aligned blocks 
        mt = MA// self.m_align
        nt = NA// self.n_align
        kt = KA// ali

        ## all valid stream using ping pong at  core level
        MM = 1
        MMl = []
        scaling = 32/ctype  ## accumulation is done in 32

        #if MA==16:
        #    pdb.set_trace()
        for m in range(1,mt + (1 if mt==1 else 0)) :
            cm = m*self.m_align
            for n in range(1,nt + (1 if nt==1 else 0)): 
                cn = n*self.n_align
                for k in range(1,kt + (1 if kt==1 else 0)):
                    kn = k*ali
                    if  cm*cn <= SC and cm*kn <= SA and kn*cn <= SB :
                        # These are valid solutions, we count the
                        # number of elements per core memory (as
                        # number of elements)
                        q =  [cm, cn,kn, atype,btype,ctype]

                        
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
                        #print(MM, MMl)
        #print(MM,MMl)

        ## maximum computation and then by maximum ratio
        ## Ratio is compute / A+B operand space in bytes 
        #W = sorted( MMl, key= lambda x: ( -self.compute(x)
        #                                  ,-self.ratio(x)
        #                                  #,self.perimeter(x)
        #                                 )
        #           )


        W1 = sorted( MMl, key= lambda x: ( self.time(x,P,1),-self.compute(x),-self.ratio(x) ))
        W0 = sorted( MMl, key= lambda x: ( self.time(x,P,0),-self.compute(x),-self.ratio(x) ))

        #if MA==16:
        #    pdb.set_trace()
        #print("TOP 3 Compute and ratio")
        #for w in W[0:3]: print(self.compute(w),self.ratio(w),self.time(w,P),w)
        print("TOP 3 Q0 Time, compute, ratio ")
        for w in W1[0:3]: print(self.time(w,P,1),self.compute(w),self.ratio(w),w)
        print("TOP 3 Q1 Time, compute, ratio")
        for w in W0[0:3]: print(self.time(w,P,0),self.compute(w),self.ratio(w),w)
        
        WW = [ W0, W1]

        Times = []
        Qs = []
        Cores =[]
        Mems = []
        Ratios = []
        for Q in [0,1] : # Tall and Fat  
            #print(Q)
            coresubvolume = WW[Q][0] 
            
            ## Build the AIE group computation COLSxROWS, for example A0*
            ## is broad cast into column 0 and A1* is broadcast to column
            ## 1, B*0 is broad cast in Row 0 ... C00 is computed in core
            ## 0,0, C01 is computed in core 1,0 ... C03 in core 3,0 Mem
            ## tile Column 0 will have even rows and Column 1 odd rows of
            ## C
            
            Q1 = self.generate_Q(Q)

            subvolume = self.memsubvolume(coresubvolume,P,Q)
            
            Mcomp = self.compute(subvolume)
            Mspace = self.space(subvolume)
            Mratio = self.ratio(subvolume)
            Ratios.append(Mratio)
            
            #Mtime_ = self.time_estimates(subvolume,P)#,True)
            Mtime = self.time(coresubvolume,P,Q)
            #print("MM", Mtime,Mtime_)
            Times.append(Mtime)
            Qs.append(Q1)
            Mems.append(subvolume)
            Cores.append(coresubvolume)

        
        
        #pdb.set_trace()
        mi = min(Times)
        i = Times.index(mi)
        w = WW[i][0]
        print(self.time(w,P,i),self.compute(w),self.ratio(w),w)
        
        FatC = i
        Tsub = Mems[i]
        coresubvolume = Cores[i]
        Q = Qs[i]
        print(Q)

        #pdb.set_trace()
        norm    = self.opspace(Tsub) < Mem 
        dbb     = self.opspace_dbb(Tsub) < Mem
        dba     = self.opspace_dba(Tsub) < Mem
        dba_dbb = self.opspace_dba_dbb(Tsub) < Mem 

        Ksplit = Tsub[2]<KA


        
        TotalTime = Times[i]
        #print("ts/rs/t", Times,Ratios,TotalTime)

        if False and not str(P) in  self.perfdictionary: self.perfdictionary[str(P)] = [  Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q, TotalTime] 
        #print("Ksplit", Ksplit, KA//Tsub[2], "total time second ",  TotalTime)
        return Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q, W1, TotalTime




class BlockLayerNorm():
    def __init__(
            self,
            mat : numpy.array
    ):
        self.A = mat
        self.n = mat.shape[1]
        self.xsq  = numpy.sum(self.A*self.A,1)
        self.x    = numpy.sum(self.A,1)

    def __str__(self):
        return  str(self.n) + " " +str(self.x)

    def reset(self):
        self.n = 0
        self.xsq =  self.xsq*0
        self.x   =  self.x*0
    def mean(self):
        return self.x/self.n
    def var(self):
        n = self.n
        return (self.xsq/n - (self.x/n)**2)#*n/(n-1)

    def value(self):
        mu = self.mean()
        s  = numpy.sqrt(self.var())
        return (self.A-mu[:,None])/s[:,None]
    
    def __add__(self, A) :
        if self.n !=0:
            #pdb.set_trace()
            self.xsq = numpy.concatenate((self.xsq, A.xsq))
            self.x   = numpy.concatenate((self.x, A.x))
            self.n = A.n
        else:
            self.xsq = A.xsq
            self.x   = A.x
            self.n = A.n
        return self

    def __mul__(self, A) :
        self.xsq += A.xsq
        self.x   += A.x
        self.n   += A.n
        
        return self
###
##  Layer Norm by Row of a matrix of size MxN
##
###

    
class LayerNorm(Gemm):
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
            to_mem_channel_bandwidth_gbits  : int = Bandwidth  # 4GBs
    ):

        
        Gemm.__init__(self,
            memtilesizeinbits,ROWS,COLS,
            CoreCSpacebits,CoreASpacebits,CoreBSpacebits,
            m_align,n_align,k_align,frequency,
            to_core_channel_bandwidth_gbits,to_mem_channel_bandwidth_gbits)

    

    def ddr_computation(
            self,
            A : numpy.array, ## operand 
    ):
        v = numpy.sqrt(numpy.var(A,1))
        m = numpy.mean(A,1)
        return (A-m[:,None])/v[:,None]


    def compute(self,x,c : int = 0):
        
        try:
            [cm,cn,ctype]= x[c]
        except:
            [cm,cn,ctype]= x

        ops = 3*cn*cm

        B = 256
        if   ctype == 8 :         B = 256
        elif ctype == 16:         B = 128
        elif ctype == 32 :        B = 64
        
        return math.ceil(ops/B)
    def reduction(self,x,c : int = 0):
        
        try:
            [cm,cn,ctype]= x[c]
        except:
            [cm,cn,ctype]= x

        ops = 2*cm

        B = 256
        if   ctype == 8 :         B = 256
        elif ctype == 16:         B = 128
        elif ctype == 32 :        B = 64
        
        return math.ceil(ops/B)
    def space(self,x, c : int = 0 ):
        
        [cm,cn,ctype]= x[c]
        

        space  = self.space_m([cm,cn,ctype])+2*self.space_m([cm,1,ctype]) 
        return space

    def minimum_computation_time(self,x, c : int = 0 ):
        
        [cm,cn,ctype]= x

        space  = self.space_m([cm,cn,ctype])
        comm   = 2*space/(self.COLS*2*self.tomembandwidth)
        cycles  = self.compute(x)/self.COLS/self.ROWS
        comp = cycles/self.core_frequency

        return max(comp,comm)

    def  time_estimates_one_column(self,x):

        [cm,cn,ctype], [mm,mn,mtype] = x

        
        Tiles = math.ceil(mm/cm)

        Kores = mn//cn

        KK = math.ceil(Kores/self.ROWS)

        ## compute 
        load = self.ROWS*self.space_m([cm,cn,ctype])/self.tocorebandwidth
        TT   = self.compute( [cm,cn,ctype])/self.core_frequency
        write = 2*self.space_m([cm,1,ctype])/self.tocorebandwidth
        Time = load +  max( load,TT)*(KK-1) + TT + write
        
        ## reduction
        load = 2*self.space_m([cm,1,ctype])/self.tocorebandwidth
        red  = self.reduction([cm,cn,ctype])/self.core_frequency
        
        Time += (load+red+write)*(Kores-1)

        
        Time *= Tiles

        return Time

        
    
    def  time_estimates(self,x, P):
        
        [cm,cn,ctype], [mm,mn,mtype]= x
        [mt,nt,ty] = P

        
        TT = self.time_estimates_one_column(x)
        
        Tiles = math.ceil(mt/self.COLS/mm)
        Kores = math.ceil(nt/mn)
        
        
        ## compute 
        load = self.ROWS*self.space_m([mm,mn,ctype])/(2*self.tomembandwidth)
        write = 2*self.space_m([mm,1,ctype])/(2*self.tomembandwidth)
        Time = load +  max( load,TT)*(Kores-1) + TT + write
        
        ## reduction
        load = 2*self.space_m([mm,1,ctype])/(2*self.tomembandwidth)
        red  = self.reduction([mm,mn,ctype])/self.core_frequency
        
        Time += (load+red+write)*(Kores-1)

        Time *=Tiles
        
        return Time

    def ddr_computation_block_one(
            self,
            A : numpy.array, ## operand 
            x : list ,       ## problem size
            mem  : list     ## memsubvolume
    ):
        #pdb.set_trace()
        mm,mn,bt = mem
        mt,nt,tt    = x
        mt = min(mt, A.shape[0]) 
        nt = min(nt, A.shape[1]) 
        
        R = BlockLayerNorm(A[0:mm,0:mn])
        R.reset()
        for mo in range(0,mt,mm):
            M = min(mo+mm,mt)

            RM = BlockLayerNorm(A[0:mm,0:mn])
            RM.reset()
            #print("RM",RM)
            for no in range(0,nt,mn):
                N = min(no+mn,nt)
                AT = A[mo:M,no:N]
                #print(no, no+mn,nt, A.shape)
                ATR = BlockLayerNorm(AT)
                #print("ATR", ATR)
                RM = RM * ATR
            R = R+RM
            
        R.A = A    
        #pdb.set_trace()
        return R

    def ddr_computation_block(
            self,
            A : numpy.array, ## operand 
            x : list ,       ## problem size
            mem  : list ,    ## memsubvolume
            core  : list ,    ## memsubvolume
    ):

        #pdb.set_trace()
        mm,mn,bt = mem
        cm,cn,ct = core
        mt,nt,tt    = x
        
        R = BlockLayerNorm(A[0:mm,0:mn])
        R.reset()
        for mo in range(0,mt,mm):
            M = min(mo+mm,mt)

            RM = BlockLayerNorm(A[0:mm,0:mn])
            RM.reset()
            
            for no in range(0,nt,mn):
                N = min(no+mn,nt)
                
                T = self.ddr_computation_block_one(
                    A[mo:M,no:N],
                    mem,
                    core)
                RM = RM * T
            R = R+RM
        #pdb.set_trace()
        R.A = A    
        
        return R
    

        
        
    ###
    ##  1) the basic computation is by h in incremental steps streaming
    ##  from memtile
    ##
    ##  2) the input and output are split by column by the w dimension
    ###
    def gen_fm_par_fm_(self, X, extype : int = 32, multiple : bool =False):
        M,N,ty = X

        # space in Bytes
        Mem = self.Mem//8

        SY = self.cspace//ty  ## ping output
        SX = 2*self.aspace//ty//2  ## ping input 
        #SW = self.bspace//w[-1]     ## whole

        ## aligned to MemTile requirements
        M = math.ceil(M/self.m_align)*self.m_align
        N = math.ceil(N/self.n_align)*self.n_align

        P = [M,N,ty]
        W = []

        MC = math.ceil(M/self.COLS)
        
        for n in range(self.n_align,N,self.n_align,):
            ## we split the computatio by column by M one core will
            ## see only half the number of rows but the reduction will
            ## be per column

            for m in range(self.m_align,MC,self.m_align):

                ## min block per core Each Core will write a sub group
                ## of channels: this is because the computation must
                ## be independent.
                
                
                core = [m, n,ty]
                mu   = [m, 1,max(ty,extype)]
                vr   = [m, 1,max(ty,extype)]
                nn    = [1, 1,ty]
                
                
                if self.space_m(core) < SX and \
                   self.space_m(mu)+self.space_m(vr)+\
                   self.space_m(nn)< SX: # MEM
                    
                    
                    ## W.append([core_problem, mem_problem]) now let's
                    ## us find the largest subproblem that fit memtile
                    ## by a multiple of the core so the communications
                    ## and computations are balanced.
                    for mm in range(m,MC,
                                    m if multiple else self.m_align
                                    ) :
                        for nn  in range(n,N,
                                         n if multiple else self.n_align
                                         ):
                            mem_problem  = [ mm , nn, ty]
                            mmu   = [mm, 1,max(ty,extype)]
                            mvr   = [mm, 1,max(ty,extype)]
                            mnn   = [1, 1,ty]
                            
                            S = self.space_m(mem_problem)+\
                                self.space_m(mmu)+\
                                self.space_m(mvr)+self.space_m(mnn)
                            
                            if S < Mem/self.COLS: # one column one memtile
                                W.append([core, mem_problem])
                            
                                
        
                                        
        if (len(W))==0 : return W

        if False :
            pdb.set_trace()
            S = sorted(
                W,
                key = lambda x:
                ( -self.compute(x,0), -self.space(x,0))
            )
            
            pdb.set_trace()
        
            print("TOP 3 by the fattest core computation")
            for i in range(min(3,len(S))):
                time = self.time_estimates(S[i],P)
                #print(self.compute(S[i]), "y, x,  wc, b, p,s", S[i])
                print("time", time,"Ref",Ref, "slowd", time/Ref,
                      "ycore", S[i][0],"ymem", S[i][1])
        #pdb.set_trace()   
        Ref = self.minimum_computation_time(P)
        S1 = sorted(
            W,
            key = lambda x:
            ( self.time_estimates(x,P),-self.compute(x))
        )
        
        print("TOP 3 by the fastest overal computation")
        for i in range(min(3,len(S1))):
            time = self.time_estimates(S1[i],P)
            #print(self.compute(S[i]), "y, x,  wc, b, p,s", S[i])
            print("time", time,"Ref",Ref, "slowd", time/Ref,
                  "ycore", S1[i][0],"ymem", S1[i][1])
        
        
        
        
        return S1
        





        
        

        

        

    
###
##     The problem size for a MHA is [d,L,type)
##     Q = L x d  type
##     K = d x L  type in bits
##     V = L x r  type in bits
##     G = L x L  Gated
##     Here we assume all operand of the same precision/type
##     Prefix computation with Lxd matrices
##     See MHALib for the more general [d,Lo, L1, r] problem for the Token computation
###
class MHA(Gemm):

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
            to_mem_channel_bandwidth_gbits  : int = Bandwidth  # 4GBs
    ):

        
        Gemm.__init__(self,
            memtilesizeinbits,ROWS,COLS,
            CoreCSpacebits,CoreASpacebits,CoreBSpacebits,
            m_align,n_align,k_align,frequency,
            to_core_channel_bandwidth_gbits,to_mem_channel_bandwidth_gbits)
        
    ## creation of the valid solution without gated computation    
    def Q_i(self, Q, K, V, exptype : int = 16, Qping : int = 1 ):
            
        # what is the size of Q_0, K_i, V_i so that exp(Q_0*K_i) * V_i
        # fit in a core?
        
        W = []
        it = math.ceil(K[1]/self.k_align)

        for i in range(1,it):
            for j in range(1,it):
                qm = i*self.k_align
                kn = j*self.k_align

                q0 = [qm, Q[1], Q[2]]
                k0 = [K[0], kn, K[2]]
                v0 = [kn, V[1], V[2]]
                
                q0k0 = [qm,kn, exptype]
                expq0k0 = [qm,kn, exptype]
                expq0k0v0 = [qm,V[1], exptype]

                N_t = [qm,V[1], exptype]
                D_t = [qm,1, exptype]
                M_t = [qm,1, exptype]
                M_t_1 = [qm,1, exptype]
                
                space =self.space_m(q0)*Qping + \
                    self.space_m(k0) + self.space_m(v0) + \
                    self.space_m(expq0k0) +  self.space_m(N_t)+ self.space_m(N_t)+ \
                    self.space_m(M_t) + self.space_m(M_t_1)

                if space < (self.cspace + self.aspace + self.bspace)//8:
                    W.append([qm,kn, space])
        return W

    ## creation of the valid solution with  gated computation    
    def Q_i_gated(self, Q, K, V, exptype : int = 16, Qping : int = 1 ):
            
        # what is the size of Q_0, K_i, V_i so that exp(Q_0*K_i) * V_i
        # fit in a core?
        
        W = []
        it = math.ceil(K[1]/self.k_align)

        for i in range(1,it):
            for j in range(1,it):
                qm = i*self.k_align
                kn = j*self.k_align

                q0 = [qm, Q[1], Q[2]]
                k0 = [K[0], kn, K[2]]
                v0 = [kn, V[1], V[2]]
                
                q0k0 = [qm,kn, exptype]
                gated = [qm,kn, exptype]
                expq0k0 = [qm,kn, exptype]
                expq0k0v0 = [qm,V[1], exptype]

                N_t = [qm,V[1], exptype]
                D_t = [qm,1, exptype]
                M_t = [qm,1, exptype]
                M_t_1 = [qm,1, exptype]
                
                space =self.space_m(q0)*Qping + \
                    self.space_m(k0) + self.space_m(v0) + \
                    self.space_m(expq0k0) + self.space_m(gated) +  self.space_m(N_t)+ self.space_m(N_t)+ \
                    self.space_m(M_t) + self.space_m(M_t_1)

                if space < (self.cspace + self.aspace + self.bspace)//8:
                    W.append([qm,kn, space])
        return W
                

    
    ###
    ##  How you would compute the softmax(QK)*V using only scipy and
    ##  numpy but using the definition.
    ## 
    ###
    def shead(
            self,
            Q : numpy.array, ## operand 
            K : numpy.array, ## operand 
            V : numpy.array ## operand
    ):
        return numpy.matmul(
            scipy.special.softmax(
                numpy.matmul(Q,K),
                1),
            V
        )
        

    ###
    ## we explicitly split the computation in long form
    ## T = Q*K
    ## T = exp(T -max_r(T)) ## normalization necessary for small bits
    ## N = T*V 
    ## D = sum(T)
    ## R = N/D
    ## 
    ###
    def ddr_computation_(
            self,
            Q : numpy.array, ## operand 
            K : numpy.array, ## operand 
            V : numpy.array, ## operand
            x : list        ## problem size
    ):
        #pdb.set_trace()
        N = numpy.zeros((Q.shape[0], V.shape[1]))
        D = numpy.zeros((Q.shape[0]))

        T = numpy.matmul(Q,K)
        T = numpy.exp(T - numpy.max(T,1)[:,None])
        D = sum(T.transpose())
        N = numpy.matmul(T,V)
        return N/D[:,None]

    ###
    ##  This is the blocked computation we would do using AIE.
    ##
    ##  The tiling suggests [m,n,Qtime, KVtime]= x. The parameter m
    ##  specify the number of row of Q and thus the row of R we
    ##  compute in one iteration. the parameter n specifies the number
    ##  of column of K (the row of V) we use for the block computation of (Q0Ki)Vi
    ## 
    ##  for qi in Q
    ##    N = 0
    ##    D = 0
    ##    M = 0
    ##    for kj,vj  in K,V
    ##        t = qi*kj
    ##        M1 = max(t)
    ##        M1 = max(M,M1)
    ##        t = exp(t - M1)
    ##        s = exp(M1-M), M = M1  (if we could avoid the division by s and transform it 
    ##        D = D/S + sum(T)
    ##        N = N/S + t*vi
    ###
    def ddr_computation_block(
            self,
            Q : numpy.array, ## operand 
            K : numpy.array, ## operand 
            V : numpy.array, ## operand
            x : list        ## problem size
    ):

        [m,n,Qtime, KVtime]= x
        R = Q*0
        N = numpy.zeros((m,V.shape[1]),dtype=Q.dtype)
        D = numpy.zeros((m),dtype=Q.dtype)
        M = numpy.zeros((m),dtype=Q.dtype)
        M1 = numpy.zeros((m),dtype=Q.dtype)

        ## qi
        for q in range(0,Q.shape[0],m):
            N = N*0
            D = D*0
            M = M*0 
            Q0 = Q[q:min(q+m, Q.shape[0]) , :]

            ## ki, vi 
            for k in range(0,K.shape[1],n):
                #pdb.set_trace()
                KI =  K[:, k:min(k+n, K.shape[1]) ]
                VI =  V[ k:min(k+n, K.shape[1]),: ]
                T = numpy.matmul(Q0,KI)
                ## normalization of the current and previous terms
                M1 =  numpy.max(T,1)
                M1 = numpy.maximum(M,M1)
                T = numpy.exp(T-M1[:,None])
                S = numpy.exp(M1-M)
                M = M1
                
                if k>0 :
                    # we could use D and N with better bit or
                    # accumulation to improve further the accuracy of
                    # the block computation.
                    D = D/S  + sum(T.transpose())
                    N = N/S[:,None] + numpy.matmul(T,VI)
                else:
                    D = sum(T.transpose())
                    N = numpy.matmul(T,VI)

            #pdb.set_trace()
            # blocked result
            R[q:min(q+m, Q.shape[0]) , :] = N/D[:,None]
        return R
            
    def minimum_computation_time(
            self,
            P : list,
            gated : bool = False
    ):
        
        
        d, L, r, dtype = P
        Space = 0
        
        cycles  = self.compute([L,L,d,dtype,dtype,dtype] ) #T = numpy.matmul(Q,K)
        Space   += 2*self.space_m([d,L,dtype])+self.space_m([L,L,dtype])

        if gated:
            cycles += self.element_compute([L,L,dtype]) # K + Gated
            Space += 3*self.space_m([L,L,dtype])

        cycles += 3*self.element_compute([L,L,dtype])# T = numpy.exp(T - numpy.max(T,1)[:,None])
        Space  += 3*self.space_m([L,L,dtype])

        cycles += self.element_compute([L,L,dtype])  # D = sum(T.transpose())
        Space += self.space_m([L,L,dtype])

        cycles  += self.compute([L,r,L,dtype,dtype,dtype] ) #T = numpy.matmul(T,V)
        Space += self.space_m([d,L,dtype])+self.space_m([L,r,dtype])+self.space_m([L,L,dtype])
        
        cycles += self.element_compute([L,r,dtype])   ## N/D 
        Space += 2*self.space_m([L,r,dtype])

        #print("cycles", cycles, "space", Space)
        Ncores = self.COLS*self.ROWS

        comptime = cycles/Ncores/self.core_frequency
        comttime = Space/(2*(min(self.COLS, self.ROWS))*self.tomembandwidth) + \
            Space/(2*(min(self.COLS, self.ROWS))*self.tocorebandwidth) 
        
        #pdb.set_trace()
        return max(comptime,comttime)

    def ops(
            self,
            P : list,
            gated : bool = False
    ):
        
        
        d, L0, L1, r, dtype = P
        Space = 0

        ops = 2*L0*L1*d  # T = numpy.matmul(Q,K)
        
        if gated:
            cycles += L0*L1 # K + Gated

        ops += 3*L0*L1   # T = numpy.exp(T - numpy.max(T,1)[:,None])
        

        ops += L0*L1  # D = sum(T.transpose())


        ops  += 2*L0*L1*r #T = numpy.matmul(T,V)

        
        ops += L0*r   ## N/D 


        
        #pdb.set_trace()
        return ops

    def minimum_computation_time_2(
            self,
            P : list,
            gated : bool = False
    ):
        d, L, r,dtype = P
        Space = 0

        gem = Gemm()
        RR0 = gem.gen_fm_par_fm_([L,L,d,dtype,dtype,dtype])
        Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q,  W,time0  = RR0
        Space   += 2*self.space_m([d,L,dtype])+self.space_m([L,L,dtype])

        
        RR1 = gem.gen_fm_par_fm_([L,r,L,dtype,dtype,dtype])
        Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q,  W,time1  = RR1
        Space += 2*self.space_m([d,L,dtype])+self.space_m([L,L,dtype])
        
        cycles = 0

        if gated:
            Space += 3*self.space_m([L,L,dtype])
            cycles += self.element_compute([L,L,dtype]) # K + Gated
        cycles += 3*self.element_compute([L,L,dtype])# T = numpy.exp(T - numpy.max(T,1)[:,None])
        Space += 3*self.space_m([L,L,dtype])
        cycles += self.element_compute([L,L,dtype])  # D = sum(T.transpose())
                                 

        cycles += self.element_compute([L,r,dtype]) ## N/D
        Space += 2*self.space_m([L,r,dtype])
        
        

        Ncores = self.COLS*self.ROWS

        comptime = cycles/Ncores/self.core_frequency + time0 + time1
        comttime = Space/(2*(min(self.COLS, self.ROWS))*self.tomembandwidth) + \
            Space/(2*(min(self.COLS, self.ROWS))*self.tocorebandwidth) 
        
        #pdb.set_trace()
        return max(comptime,comttime)


    ## estimate ddr_ddr_ time estimate given a memtile subproblem
    ##  m,n,Qtime, KVtime= X
    ##  d, L, r,dtype = P
    def time_estimates(
            self,
            X : list , ## memtile subproblem
            P      : list , ## aligned problem size
            gated : bool = False
            
    ):
        #pdb.set_trace()
        m,n,Qtime, KVtime= X
        d, L, r,dtype = P

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
            load = (self.space_m(q0)+ self.space_m(gated))/self.tocorebandwidth

            core_time = []
            for k in range(KVtime):
                
                cycles = self.compute([m,n,d,dtype,dtype,dtype]) # numpy.matmul(Q0,KI)
                if gated: cycles += self.element_compute([m,n,dtype])              # + gated
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

            load = (self.space_m(q0)+ self.space_m(gated))/self.tomembandwidth

            write = (self.space_m(expq0k0v0))/self.tomembandwidth
            #pdb.set_trace()
            if KVtime <= self.ROWS:
                Qtime /= self.COLS
            WQ = math.ceil(L/m)
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

    
    def counts_only_multiples(self, x : list, M : int , N : int ):
        #print(x,M,N)
        q0 = M%x[0]
        q1 = M%x[1]
        q2 = 0 #(M//x[1]) % N
        

        if q0==0 and  q1==0 and q2 == 0: return 1
        else         : return -1


    ## tiling code geneation we have the problem to solve
    ## [d,l,type in bits]  = X
    def gen_fm_par_fm_(
            self,
            X : list,
            gated : bool= False 
            
    ) -> list :

        d, L, r, dtype = X
        
        d = math.ceil(d/self.n_align)*self.n_align
        L = math.ceil(L/self.n_align)*self.n_align
        r = math.ceil(r/self.n_align)*self.n_align
        
        Q = [L, d , dtype ]  ## matrix shapes
        K = [d, L , dtype ]  
        V = [L, r , dtype ]  
        
        S = self.Q_i(Q,K,V) if not gated else self.Q_i_gated(Q,K,V)## all solutions 
        if False:
            W = sorted(
                S,
                key = lambda x: (
                    -x[2]*self.counts_only_multiples(x,L,self.ROWS), -x[1]
                )
            )

            print("Top 5")
            for x in W[0:10]:
                print(x,[x[0],x[1],math.ceil(L/x[0]),math.ceil(L/x[1])])
                print(x,
                      self.time_estimates([x[0],x[1],math.ceil(L/x[0]),math.ceil(L/x[1])],X,gated),
                      (x[1], x[2]*self.counts_only_multiples(x,L,self.ROWS)
                       )
                      )

        W1 = sorted(
            S,
            key = lambda x: (
                self.time_estimates(
                    [x[0],x[1],math.ceil(L/x[0]),math.ceil(L/x[1])],X,gated)
            )
        ) 
        
        #print("Top 5")
        #for x in W1[0:10]: print(x, self.time_estimates([x[0],x[1],math.ceil(L/x[0]),math.ceil(L/x[1])],X,gated))
            
        
       
        m, n, t = W1[0]
        Qtime  = math.ceil(L/m)
        KVtime = math.ceil(L/n)
        print(W1[0])
        
        return [m,n,Qtime, KVtime]
        

###
##    y = CONV(x)+b  
##    x = [-1, hx,wx,cx,type]  Input tensor 
##    y = [-1, hy,wy,cy,type]  Output tensor
##    b = [-2, 1, 1, cy,type]  Bias
##    w = [cy,h, w, cx,type],  Weights 
##    p = [0,hp, wp, 0, 0],    Padding 
##    s = [0,hs,ws,0,0]        Stride 
##
##    Problem definition [y,x,w,b,p,s]
###
class CONV(Gemm):

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
            to_mem_channel_bandwidth_gbits  : int = Bandwidth  # 4GBs
    ):

        
        Gemm.__init__(self,
            memtilesizeinbits,ROWS,COLS,
            CoreCSpacebits,CoreASpacebits,CoreBSpacebits,
            h_align,w_align,c_align,frequency,
            to_core_channel_bandwidth_gbits,to_mem_channel_bandwidth_gbits)
        
        self.h_align = 1
        self.w_align = COLS
        self.c_align = c_align
        
            
        
    def ddr_computation(
            self,
            X : numpy.array, ## operand 
            W : numpy.array, ## operand 
            B : numpy.array,
            Y : numpy.array,
            s : list 
    ):

        bx,hx,wx,cx = X.shape
        by,hy,wy,cy = Y.shape
        bw,hw,ww,cw = W.shape
        bb,hb,wb,cb = B.shape
        bs,hs,ws,cs = s

        
        ph = (hy-1)*hs +hw - hx 
        pw = (wy-1)*ws +ww - wx 
        #pdb.set_trace()
        
        if ph>0 or pw>0: 
            pt = math.ceil(ph/2) if ph>0 else 0
            pl = math.ceil(pw/2) if pw>0 else 0
            
            XP = numpy.zeros((bx,hx+ph,wx+pw,cx))
            XP[:,pt:(hx+ph-(ph-pt)), pl:(wx+pw-(pw-pl)),:] = X
            X = XP*1
        
        for bo in range(by):    
            for ho in range(hy):
                for wo in range(wy):
                    for co in range(cy):
                        #pdb.set_trace()
                        T = X[bo,ho*hs:(ho*hs+hw),wo*ws:(wo*ws+ww),:]*W[co,:,:,:]
                        Y[bo,ho,wo,co] = sum(T.flatten()) + B[bo,0,0,co]

        #
        return Y

    def ddr_computation_blocked(
            self,
            X : numpy.array, ## operand 
            W : numpy.array, ## operand 
            B : numpy.array,
            Y : numpy.array,
            Xs : list,
            P : list,
            
    ):
        #pdb.set_trace()
        y,x,w,b,p,s  = P
        core, memsub = Xs
        yc,xc,wc,bc,pc,sc  = core
        ym,xm,wm,bm,pm,sm  = memsub
  
        
        HT,WT,COT,HCT,WCT,CCOT = self.iterations_counts(P,Xs)
        
        bx,hx,wx,cx = X.shape
        by,hy,wy,cy = Y.shape
        bw,hw,ww,cw = W.shape
        bb,hb,wb,cb = B.shape
        bs,hs,ws,cs = s

        
        
        ph = (hy-1)*hs +hw - hx 
        pw = (wy-1)*ws +ww - wx 
        #pdb.set_trace()
        
        if ph>0 or pw>0: 
            pt = math.ceil(ph/2) if ph>0 else 0
            pl = math.ceil(pw/2) if pw>0 else 0
            
            XP = numpy.zeros((bx,hx+ph,wx+pw,cx))
            XP[:,pt:(hx+ph-(ph-pt)), pl:(wx+pw-(pw-pl)),:] = X
            X = XP*1
            
        for bo in range(by):  ## outers  memtiles 
            for hoo in range(0,hy,ym[1]):
                for woo in range(0,wy,ym[2]):
                    for coo in range(0,cy,ym[3]):

                        #inners memtiles 
                        for hoi in range(ym[1]):
                            ho = min(hy, hoo+hoi)
                            for woi in range(ym[2]):
                                wo = min(wy, woo+woi)
                                for coi in range(ym[3]):
                                    co =min(cy,coo+coi)

                                    # one point, the blocked
                                    #computation will break the
                                    #summation making more pleasurable
                                    # pdb.set_trace()
                                    T = X[bo,ho*hs:(ho*hs+hw),wo*ws:(wo*ws+ww),:]*W[co,:,:,:]
                                    Y[bo,ho,wo,co] = sum(T.flatten()) + B[bo,0,0,co]

        #pdb.set_trace()
        return Y
        
    def element_compute(self,x):
        b,m,n,c,ctype = x

        ops = m*n*c

        B = 256
        if   ctype == 8 :         B = 256
        elif ctype == 16:         B = 128
        elif ctype == 32 :        B = 64
        
        return math.ceil(ops/B)
    ## space in Bytes for C, A, and B 
    def space(self,X):
        y,x,w,b,p,s = X
        return self.space_m(y)+self.space_m(x) + self.space_m(w) + self.space_m(b)
    def space_i(self,X):
        y,x,w,b,p,s = X
        return self.space_m(x) + self.space_m(w) + self.space_m(b)
    def space_o(self,X):
        y,x,w,b,p,s = X
        return self.space_m(y) 

    def space_m(self,A):
        b,h,w,c,atype = A
        return atype*abs(b)*h*w*c//8
    def space_e(self,A):
        b,h,w,c,atype = A
        return abs(b)*h*w*c
    
    ## CIN SPLIT
    def opspace_dbw(self,X):
        y,x,w,b,p,s = X
        return self.space_m(x) + 2*(self.space_m(w)+self.space_m(b))
    ## I DONOT KNOW
    def opspace_dbx(self,X):
        y,x,w,b,p,s = X
        return (2*self.space_m(x))# + (self.space_m(w)+self.space_m(b)))
                
    ## W reuse.
    def opspace_dby_dbx(self,X):
        y,x,w,b,p,s = X
        return (2*self.space_m(y) +  2*self.space_m(x) + (self.space_m(w)+self.space_m(b)))

    ## Only inputs
    def opspace(self,X):
        y,x,w,b,p,s = X
        return self.space_m(x) + self.space_m(w) + self.space_m(b)
    ## Only inputs
    def opspace_x(self,X):
        y,x,w,b,p,s = X
        return self.space_m(x) 
    def opspace_y(self,X):
        y,x,w,b,p,s = X
        return self.space_m(y) 
    def opspace_w(self,X):
        y,x,w,b,p,s = X
        return self.space_m(w) +  self.space_m(b)


    def align_t(self,x):
        b,h,w,c,atype = x
        if b==-1: ## tensor
            return [
                b,
                math.ceil(h/self.h_align)*self.h_align,
                math.ceil(w/self.w_align)*self.w_align,
                math.ceil(c/self.c_align)*self.c_align,
                atype
            ]
        elif b==-2: # bias
            return [
                b,
                h,
                w,
                math.ceil(c/self.c_align)*self.c_align,
                atype
            ]
        elif b>0: #weight
            return [
                math.ceil(b/self.c_align)*self.c_align,
                h,
                w,
                math.ceil(c/self.c_align)*self.c_align,
                atype
            ]
        return x

    def projection(self, u : list , X : list)-> list :
        y,x,w,b,p,s   = X
        wy,hw,ww,wx,_  = w
        _,sh, sw, _  = s
        _,ph, pw, _  = p
         
        yb,yh,yw,yc,yatype = u
        xb,xh,xw,xc,xatype = u
        
         
        xh = (yh-1)*sh + hw
        xw = (yw-1)*sw + ww
        
        return [ xb,xh,xw,xc,xatype]
     

    def projection_c_m_o(self, u : list, y : list ) -> list:
        b,h,w,c,atype = u
        rb,rh,rw,rc,ratype = y
        
        return [ b,
                 h,
                 min(w*self.COLS,rw),
                 min(c*self.ROWS,rc),
                 atype]
     
         
    def compute(self,x):
        try: 
            [y,x,w,b,p,s],[ym,xm,wm,bm,pm,sm]= x
        except:
            [y,x,w,b,p,s] =  x
            #pdb.set_trace()
            #return 0
        ctype = y[-1]

        ops = 2*y[1]*y[2]*y[3]* w[1]*w[2]*w[3]+b[3]

        B = 256
        if   ctype == 8 :         B = 256
        elif ctype == 16:         B = 128
        elif ctype == 32 :        B = 64
        
        return math.ceil(ops/B)

    def ops(self,x):
        try: 
            [yc,xc,wc,bc,pc,sc],[y,x,w,b,p,s]= x
        except:
            [y,x,w,b,p,s] =  x
            #pdb.set_trace()
            #return 0
        ops = 2*y[1]*y[2]*y[3]* w[1]*w[2]*w[3]+b[3]
        return ops

    def fat_mem(self,x):
        try: 
            [y,x,w,b,p,s],[ym,xm,wm,bm,pm,sm]= x
        except:
            pdb.set_trace()
            return 0
        return (self.space_m(ym) +self.space_m(xm) +self.space_m(wm) +self.space_m(bm))


    
    
    def minimum_computation_time(
            self,
            P : list
    ):

        Comp = self.compute(P)

        Space= self.space(P)
        Space_2= self.space_o(P)

        Ncores = self.COLS*self.ROWS

        comptime = Comp/Ncores/self.core_frequency
        comttime = Space/(2*(min(self.COLS, self.ROWS))*self.tomembandwidth) + \
            Space/(2*(min(self.COLS, self.ROWS))*self.tocorebandwidth) 
        
        #pdb.set_trace()
        return max(comptime,comttime)

    ##  get time estimates for convolutions in memtile
    ##  core, memsub = X
    ##  yc,xc,wc,bc,pc,sc  = core
    ##  ym,xm,wm,bm,pm,sm  = memsub
    def get_time_cluster_with_latency(
            self, X : list,
            HCT,WCT,CCOT
    ):

        core, memsub = X
        
        # each core compute independently a sub volume, we coutn the
        # number of cores
        Ncores = self.COLS*self.ROWS
        yc,xc,wc,bc,pc,sc  = core
        ym,xm,wm,bm,pm,sm  = memsub

    
        MM = 2*ym[1]*ym[2]*ym[3]* wm[1]*wm[2]*wm[3]

        CM = 2*yc[1]*yc[2]*yc[3]* wc[1]*wc[2]*wc[3]

        
        start = self.opspace_x(core)/(self.tocorebandwidth)
        
        computetime = (HCT-1)*self.compute(X)/self.core_frequency
        read    = (HCT-1)*self.opspace_x(core)/(self.tocorebandwidth)
        write   = (HCT-1)*self.opspace_y(core)/(self.tocorebandwidth)
        
        end = self.compute(X)/self.core_frequency
        end_w = self.opspace_y(core)/(self.tocorebandwidth)
        
        Mtime = max(
            computetime,
            read,
            write 
        )
        time = (start+Mtime+end+end_w) * WCT 
        

        ops = CM*(ym[1]/yc[1])*(ym[2]/yc[2])*(ym[3]/yc[3])*(wm[3]/wm[3])
        
        #pdb.set_trace()
        if self.compute(memsub)/Ncores/self.core_frequency > time:
            print("ERERR", CM, HCT, WCT, self.compute(X))
            pdb.set_trace()
            
            
        return time,  MM/Ncores/256/self.core_frequency

    ## estimate ddr_ddr_ time estimate given a memtile subproblem
    ## y,x,w,b,p,s  = P
    ## core, memsub = X
    ## yc,xc,wc,bc,pc,sc = core 
    ## ym,xm,wm,bm,pm,sm = memsub
    def time_estimates(
            self,
            X : list , ## memtile subproblem
            P      : list , ## aligned problem size
            
    ):
        #pdb.set_trace()
        y,x,w,b,p,s  = P
        core, memsub = X
        yc,xc,wc,bc,pc,sc  = core
        ym,xm,wm,bm,pm,sm  = memsub
         
        
        ## this is Memtile space in Bytes (the unit of every opspace
        Mem = self.Mem//8 # bytes

        MA,NA,KA,atype, btype, ctype = P
        #pdb.set_trace()
        norm    = self.opspace(memsub) < Mem 
        dbx     = self.opspace_dbx(memsub) < Mem

        CC = self.compute(P)
        
        CM = self.compute(memsub)

        # input + weights 
        load_to_core  =   self.opspace(core)/(2*self.tocorebandwidth)
        load_to_core +=   self.opspace(core)/(2*self.tomembandwidth)


        HT,WT,COT,HCT,WCT,CCOT = self.iterations_counts(P,X)
        
        #pdb.set_trace()
        TT,TM = self.get_time_cluster_with_latency(X,HCT,WCT,CCOT)

        TotalTime = 0

        CINS = (math.ceil(x[3]/xm[3]))
        
        ## CIN SPLIT means that we need to combine the results 
        
        
        
        ## for every CINs and CT  we send in the weight and bias 
        load_to_mem_W = self.opspace_w(memsub)/(2*(self.COLS)*self.tomembandwidth)   ## load weights memtile
        load_to_core_w = self.opspace_w(core)/(self.tocorebandwidth) ## load weights to core

        
        ## we stream x by height but we may need to do multiple pass by W         
        load_to_mem =  self.opspace_x(memsub)/(self.tocorebandwidth)  ## load x into memtile
        write_to_mem =  self.opspace_y(memsub)/(self.tocorebandwidth) ## write y into memtile
        TotalTime += (load_to_mem + TT + write_to_mem)*HT*WT

        TotalTime +=load_to_mem_W  +load_to_core_w
                
        TotalTime*=COT

        TotalTime *= CINS

                    
        ## cost of the final reduction by CIN: read + add + write
        TotalTime += (
            self.opspace_y(memsub)/(2*self.tomembandwidth) + \
            self.element_compute(ym)/self.core_frequency + \
            self.opspace_y(memsub)/(2*self.tomembandwidth)
        ) * CINS
            
        
        #pdb.set_trace()
                              
        #print("Total asdasd",
        #      TotalTime,CC,CM*Hsplit*Wsplit*math.ceil(y[2]/ym[2])*math.ceil(x[2]/xm[2]),
        #      CC/8/256/self.core_frequency)
        if TotalTime ==0:
            if v: print("None", norm, dba,dbb, dba_dbb,Ksplit)

            ## non of the above fit in Memtile Time => infinity
            TotalTime = 1000000.0
            #pdb.set_trace()
            #print("Ksplit", Ksplit, KA//Tsub[2], "total time second ",  TotalTime)

        return TotalTime



    ## The original problem y,x,w,b,p,s = X
    ## is broken up in memtiles problems
    ## y -> ym  HT = y[1]/ym[1]
    ##          WT = y[2]/ym[2]
    ##          COT= y[3]/ym[3]
    ## Now we have ym[2]/COLS per columns
    ##             ym[3]/ROWS per core
    ## Thus we have these may core call 
    ##          HCT = ym[1]/yc[1]
    ##          WCT = ym[2]/COLS/yc[2]
    ##          CCOT =   ym[3]/ROWS/yc[3]

    def iterations_counts(
            self,
            P : list,
            X : list
    ):
        y,x,w,b,p,s = P
        core, memsub = X
        yc,xc,wc,bc,pc,sc  = core
        ym,xm,wm,bm,pm,sm  = memsub


        HT = math.ceil(y[1]/ym[1])
        WT = math.ceil(y[2]/ym[2])
        COT= math.ceil(y[3]/ym[3])

        percol  =  math.ceil(ym[2]/self.COLS)
        percore =  math.ceil(ym[3]/self.ROWS)
        HCT =      math.ceil(ym[1]/yc[1])
        WCT =      math.ceil(percol/yc[2])
        CCOT =     math.ceil(percore/yc[3])
        
        return HT,WT,COT, HCT, WCT, CCOT
    
    ###
    ##  1) the basic computation is by h in incremental steps streaming
    ##  from memtile
    ##
    ##  2) the input and output are split by column by the w dimension
    ##
    ##  2) the COUT is split by Rows 
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


        P = [y,x,ww,b,p,s]
        
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

                            W.append([core_problem, mem_problem])
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
                                        W.append([core_problem, mem_problem])
                            

        
                                        
        if (len(W))==0 : return W
        #S = sorted(
        #    W,
        #    key = lambda x:
        #    ( -self.compute(x), -self.fat_mem(x))
        #)
        
        
        #Ref = self.minimum_computation_time(P)
        #print("expsub", expsub,"TOP 3 by the fattest core computation")
        #for i in range(min(3,len(S))):
        #    time = self.time_estimates(S[i],P)
        #    #print(self.compute(S[i]), "y, x,  wc, b, p,s", S[i])
        #    print("time", time,"Ref",Ref, "slowd", time/Ref,"ycore", S[i][0][0],"ymem", S[i][1][0])
        #pdb.set_trace()   
        Ref = self.minimum_computation_time(P)
        S1 = sorted(
            W,
            key = lambda x:
            ( self.time_estimates(x,P),-self.compute(x), -self.fat_mem(x))
        )
        if Ref> self.time_estimates(S1[0],P):
            pdb.set_trace()
            Ref = self.minimum_computation_time(P)
            self.time_estimates(S1[0],P)
        #print("TOP 3 by the fastest overal computation")
        #for i in range(min(3,len(S))):
        #    time = self.time_estimates(S1[i],P)
        #    #print(self.compute(S[i]), "y, x,  wc, b, p,s", S[i])
        #    print("time", time,"Ref",Ref, "slowd", time/Ref,"ycore", S1[i][0][0],"ymem", S1[i][1][0])
        
        
        
        
        return S1


    def cinsplit(self, X : list, T : list ) :
        y,x,w,b,p,s = X

        xb,xh,xw,xc,xtype = x
        wb,wh,ww,wc,wtype = w
        ref = None
        if len(T)>0:
            ref =  self.time_estimates(T[0],X)
        
    
        for c in range(4):
            nec = math.ceil(xc/(2**c))
            if nec<2: continue
            x = xb,xh,xw,nec,xtype
            w = wb,wh,ww,nec,xtype
            P = [y,x,w,b,p,s]
            T = self.gen_fm_par_fm_(P)
            if len(T)>0:
                time =  self.time_estimates(T[0],P)*(2**c)
                #time +=  self.space_m(y)*( 2**c) -1)*(2*self.tocorebandwidth) + self.space_m(y)*( 2**c) -1)*(2*self.tomembandwidth)
                if ref is None :
                    ref = time
                    W = [ T,(2**c)]
                elif time >= ref:
                    break
                else: 
                    ref = time
                    W = [ T,(2**c)]
                    
        return W

    def coutsplit(self, X : list, T : list ) :
        y,x,w,b,p,s = X

        ref = None
        if len(T)>0:
            ref =  self.time_estimates(T[0],X)

        yb,yh,yw,yc,ytype = y
        wb,wh,ww,wc,wtype = w

        Ws  = [] 
        for c in range(4):
            nec = math.ceil(xc/(2**c))
            if nec<2: continue
            y = yb,yh,yw,nec,xtype
            w = nec,wh,ww,wc,wtype
            T = self.gen_fm_par_fm_([y,x,w,b,p,s])
            if len(T)>0:
                time =  self.time_estimates(T[0],P)*(2**c)
                if ref is None :
                    ref = time
                    W = [ T,(2**c)]
                elif time >= ref:
                    break
                else: 
                    ref = time
                    W = [ T,(2**c)]
            


        return W
            
        
        
def test_layernorm():
    ln = LayerNorm()

    
    Q = numpy.random.rand(768,768)

    R0 = ln.ddr_computation(Q)
    #pdb.set_trace()
    RR1 = ln.ddr_computation_block(Q, [768,768,8], [384,384,8], [96,96,8] )
    R1 = RR1.value()

    pdb.set_trace()

    S= ln.gen_fm_par_fm_([768,768,8], multiple = True)
    pdb.set_trace()
    Ref = ln.minimum_computation_time([768,768,8])    
    print("TOP 3 by the fastest overal computation")
    for i in range(min(3,len(S))):
        time = ln.time_estimates(S[i],[768,768,8])
        #print(self.compute(S[i]), "y, x,  wc, b, p,s", S[i])
        print("time", time,"Ref",Ref, "slowd",
              time/Ref,"ycore", S[i][0],"ymem", S[i][1])
    

def test_conv():
    conv = CONV()

    Q = []
    
    x = [-1,224, 224, 3, 8]
    y = [-1,112, 112, 32,8]
    w = [32,  7,   7, 3 ,8]
    b = [-2,  1,   1,16 ,8]
    s = [0,2,2,0]
    p = [0,0,0,0]

    P = [y,x,w,b,p,s]
    Q.append(P)
    
    x = [-1,112, 112, 32, 8]
    y = [-1,56, 56, 32,8]
    w = [32, 3,  3, 32,8]
    b = [-2,  1, 1, 32 ,8]
    s = [0,2,2,0]
    p = [0,0,0,0]

    P = [y,x,w,b,p,s]
    Q.append(P)

    x = [-1,64, 64, 128, 8]
    y = [-1,64, 64, 128,8]
    w = [128, 1,  1, 128,8]
    b = [-2,  1, 1, 128 ,8]
    s = [0,1,1,0]
    p = [0,0,0,0]

    P = [y,x,w,b,p,s]
    Q.append(P)


    for P in Q:
        Ref = conv.minimum_computation_time(P)
        print("Reference without alignments", Ref)
    
        S = conv.gen_fm_par_fm_(P)
        i=0
        expsub =1
        times_ = 1
        while len(S)==0:
            i+=1
            expsub = w[3]//2**i
            S = conv.gen_fm_par_fm_(P,expsub)
            times_ = 2**i
        print(len(S))
        for i in range(3):
            time =conv.time_estimates(S[i],P)
            print(times_, "Time", time, "miimum", Ref, "slowdown", (time)/Ref)


     
        pdb.set_trace()
    
    pdb.set_trace()

def test_conv_2():
    conv = CONV()
    gemm = Gemm()

    

    x = [-1,32, 32, 1024, 8]
    y = [-1,32, 32, 1024, 8]
    w = [1024, 1,  1, 1024,8]
    b = [-2,  1, 1, 1024 ,8]
    s = [0,1,1,0]
    p = [0,0,0,0]

    P = [y,x,w,b,p,s]
    
    Ref = conv.minimum_computation_time(P)
    print("Reference without alignments", Ref)
    pdb.set_trace()
    S = conv.gen_fm_par_fm_(P)
    i=0
    expsub =1
    times_ = 1
    while len(S)==0:
        i+=1
        expsub = w[3]//2**i
        S = conv.gen_fm_par_fm_(P,expsub)
        times_ = 2**i
    print(len(S))
    for i in range(3):
        time =conv.time_estimates(S[i],P)
        print(times_, "Time", time, "miimum", Ref, "slowdown", (time)/Ref)

    pdb.set_trace()
    P = [1024,1024,1024,8,8,8]
    Ref = gemm.minimum_computation_time(P)
    print("Reference without alignments", Ref)
    
    RR = gemm.gen_fm_par_fm_(P)
    Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q,  S,time  = RR
    print(times_, "Time", time, "miimum", Ref, "slowdown", (time)/Ref)
    print(len(S))
    for i in range(3):
        time =gemm.time(coresubvolume,P,Q)
        print(times_, "Time", time, "miimum", Ref, "slowdown", (time)/Ref)
    
        
        
    
    pdb.set_trace()

    
def test_mha():
    mha = MHA()

    P = [77, 768, 16]
    Ref = mha.minimum_computation_time(P)
    RT = mha.gen_fm_par_fm_(P)
    time = mha.time_estimates(RT,P)
    print(P,"time", time, "ref", Ref, "slowdown", time/Ref)

    P = [77, 768, 8]
    Ref = mha.minimum_computation_time(P,True)
    RT = mha.gen_fm_par_fm_(P,True)
    time = mha.time_estimates(RT,P,True)
    print(P,"time", time, "ref", Ref, "slowdown", time/Ref)

    for t in range(1) : 
        Q = numpy.random.rand(768,77)
        K = numpy.random.rand(77,768)
        V = numpy.random.rand(768,77)

        Q32 = numpy.ndarray.astype(Q,numpy.float16)
        K32 = numpy.ndarray.astype(K,numpy.float16)
        V32 = numpy.ndarray.astype(V,numpy.float16)

        #pdb.set_trace()
        One = mha.shead(Q,K,V)
        one = mha.shead(Q32,K32,V32)
        two = mha.ddr_computation_(Q32,K32,V32,[])
        three = mha.ddr_computation_block(Q32,K32,V32,RT)
        
        print("scipy L1 %1.3e" % ( sum(sum(numpy.fabs(One-one)))/One.shape[0]/One.shape[1]))
        print("separ L1 %1.3e" % (sum(sum(numpy.fabs(One-two)))/One.shape[0]/One.shape[1]))
        print("block L1 %1.3e" % (sum(sum(numpy.fabs(One-three)))/One.shape[0]/One.shape[1]))
        
    pdb.set_trace() 




def test_mha_L(btype : int = 8):
    mha = MHA()
    
    P = [77, 768, 77, btype]
    Ref = mha.minimum_computation_time(P,True)
    Ref2 = mha.minimum_computation_time_2(P,True)
    RT = mha.gen_fm_par_fm_(P,True)
    time = mha.time_estimates(RT,P,True)
    print(P,RT,"time", time, "ref", Ref, Ref2,"slowdown", time/Ref, time/Ref2)
    return 0
    pdb.set_trace()
    for k in range(0,768,16):
        P = [1, 768+k, 77, btype]
        Ref = mha.minimum_computation_time(P,True)
        RT = mha.gen_fm_par_fm_(P,True)
        time = mha.time_estimates(RT,P,True)
        print(P,RT,"time", time, "ref", Ref, "slowdown", time/Ref)


    

def test_gemm_2(cols=2,actbit=16,weibit=8) :
    gem = Gemm(COLS=cols)
    Times = []

    
    M,N,K = [512,768,768]
    Ref = gem.minimum_computation_time([M,N,K,actbit,weibit,actbit])

    pdb.set_trace()
    RR = gem.gen_fm_par_fm_([M,N,K,actbit,weibit,actbit])
    Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q,  W,time  = RR
    print("## P", [M,N,K],Ref)
    print("## Time", time, "padding M", M%coresubvolume[0], "N", N % coresubvolume[1], "K", K % coresubvolume[2])
    print("## xslowdown", time/Ref, Tsub, coresubvolume)
    print("## TFLOPS", 2*M*N*K/time/10**12)
    Mod = M%coresubvolume[0]!=0 or  N % coresubvolume[1] != 0 or  K % coresubvolume[2]!=0
    Times.append([[M,N,K],Tsub, coresubvolume,Mod,Ksplit, FatC,time])

    if Mod:
        RR = gem.gen_fm_par_fm_([M,N,K,actbit,weibit,actbit],[2])
        Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q,  W,time  = RR
        print("## P", [M,N,K],Ref)
        print("## Time", time, "padding M", M%coresubvolume[0], "N", N % coresubvolume[1], "K", K % coresubvolume[2])
        print("## xslowdown", time/Ref)
        Mod = M%coresubvolume[0]!=0 or  N % coresubvolume[1] != 0 or  K % coresubvolume[2]!=0
        Times.append([[M,N,K],Tsub, coresubvolume,Mod,Ksplit, FatC,time])
    for t in Times:
        print(t)
def test_gemm(cols=2,actbit=16,weibit=8) :
    gem = Gemm(COLS=cols)
    Times = []

    
    M,N,K = [512,768,768]
    Ref = gem.minimum_computation_time([M,N,K,actbit,weibit,actbit])

    #pdb.set_trace()
    RR = gem.gen_fm_par_fm_([M,N,K,actbit,weibit,actbit])
    Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q,  W,time  = RR
    print("## P", [M,N,K],Ref)
    print("## Time", time, "padding M", M%coresubvolume[0], "N", N % coresubvolume[1], "K", K % coresubvolume[2])
    print("## xslowdown", time/Ref, Tsub, coresubvolume)
    Mod = M%coresubvolume[0]!=0 or  N % coresubvolume[1] != 0 or  K % coresubvolume[2]!=0
    Times.append([[M,N,K],Tsub, coresubvolume,Mod,Ksplit, FatC,time])

    if Mod:
        RR = gem.gen_fm_par_fm_([M,N,K,actbit,weibit,actbit],[2])
        Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q,  W,time  = RR
        print("## P", [M,N,K],Ref)
        print("## Time", time, "padding M", M%coresubvolume[0], "N", N % coresubvolume[1], "K", K % coresubvolume[2])
        print("## xslowdown", time/Ref)
        Mod = M%coresubvolume[0]!=0 or  N % coresubvolume[1] != 0 or  K % coresubvolume[2]!=0
        Times.append([[M,N,K],Tsub, coresubvolume,Mod,Ksplit, FatC,time])

    
    A = numpy.ones((512,768), dtype=numpy.int16) 
    B = numpy.ones((768,768), dtype=numpy.int8)*2 
    C = numpy.ones((512,768), dtype=numpy.int16)*0
    
    gem.ddr_computation(A,B,C, [M,N,K,actbit,weibit,actbit], coresubvolume, Tsub)
    
    M,N,K = [512,768*4,768]
    Ref = gem.minimum_computation_time([M,N,K,actbit,weibit,actbit])
    RR = gem.gen_fm_par_fm_([M,N,K,actbit,weibit,actbit])
    Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q,  W,time = RR
    Mod = M%coresubvolume[0]!=0 or  N % coresubvolume[1] != 0 or  K % coresubvolume[2]!=0
    print("## Time", time,"padding M", M%coresubvolume[0], "N", N % coresubvolume[1], "K", K % coresubvolume[2])
    print("## xslowdown", time/Ref)
    Times.append([[M,N,K],Tsub, coresubvolume,Mod,Ksplit, FatC,time])
    if Mod:
        RR = gem.gen_fm_par_fm_([M,N,K,actbit,weibit,actbit],[2])
        Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q,  W,time  = RR
        print("## Time", time, "padding M", M%coresubvolume[0], "N", N % coresubvolume[1], "K", K % coresubvolume[2])
        print("## xslowdown", time/Ref)
        Mod = M%coresubvolume[0]!=0 or  N % coresubvolume[1] != 0 or  K % coresubvolume[2]!=0
        Times.append([[M,N,K],Tsub, coresubvolume,Mod,Ksplit, FatC,time])
    #pdb.set_trace()

    M,N,K = [512,768,768*4]
    Ref = gem.minimum_computation_time([M,N,K,actbit,weibit,actbit])
    #pdb.set_trace()
    RR =  gem.gen_fm_par_fm_([M,N,K,actbit,weibit,actbit])
    Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q,  W,time = RR
    
    Mod = M%coresubvolume[0]!=0 or  N % coresubvolume[1] != 0 or  K % coresubvolume[2]!=0
    print("## Time", time,"padding M", M%coresubvolume[0], "N", N % coresubvolume[1], "K", K % coresubvolume[2])
    print("## xslowdown", time/Ref)
    Times.append([[M,N,K],Tsub, coresubvolume,Mod,Ksplit, FatC,time])
    if Mod:
        RR = gem.gen_fm_par_fm_([M,N,K,actbit,weibit,actbit],[2])
        Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q,  W,time  = RR
        print("## Time", time, "padding M", M%coresubvolume[0], "N", N % coresubvolume[1], "K", K % coresubvolume[2])
        print("## xslowdown", time/Ref)
        Mod = M%coresubvolume[0]!=0 or  N % coresubvolume[1] != 0 or  K % coresubvolume[2]!=0
        Times.append([[M,N,K],Tsub, coresubvolume,Mod,Ksplit, FatC,time])
    #pdb.set_trace()
    if Ksplit:
        RR =  gem.gen_fm_par_fm_([M,N,K,actbit,weibit,actbit], expsub =[FatC  ] )
        Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q,  W,time = RR
        print("## Time", time,"padding M", M%coresubvolume[0], "N", N % coresubvolume[1], "K", K % coresubvolume[2])
        print("## xslowdown", time/Ref)
        Mod = M%coresubvolume[0]!=0 or  N % coresubvolume[1] != 0 or  K % coresubvolume[2]!=0
        Times.append([[M,N,K],Tsub, coresubvolume,Mod,Ksplit, FatC,time])
        if Mod:
            RR = gem.gen_fm_par_fm_([M,N,K,actbit,weibit,actbit],[2])
            Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q,  W,time  = RR
            print("## Time", time, "padding M", M%coresubvolume[0], "N", N % coresubvolume[1], "K", K % coresubvolume[2])
            print("## xslowdown", time/Ref)
            Mod = M%coresubvolume[0]!=0 or  N % coresubvolume[1] != 0 or  K % coresubvolume[2]!=0
            Times.append([[M,N,K],Tsub, coresubvolume,Mod,Ksplit, FatC,time])
        #pdb.set_trace()

    print("COLS %d" %(cols))
    for t in Times:
        print(t)

def estimate_head_psj():
    mha = MHA()
    gem = Gemm()
    ln  = LayerNorm()

    bits   = 8
    acbits = 8
    webit  = 8
    d = d
    L = 768
    h = 12
        
    heads = 12

    Times = []
    
    print("Pre multiplications")
    P =[77,77  ,1152,acbits,webit,acbits ]
    Ref = gem.minimum_computation_time(P)
    RR = gem.gen_fm_par_fm_(P)
    Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q,  W,time  = RR
    print("GEMM Time ",P,time,"## xslowdown", time/Ref,"MemTile", Tsub, "coretile",coresubvolume)
    Times.append([P, time])
    
    
    P =[77,1152,1152,acbits,webit,acbits ]
    Ref = gem.minimum_computation_time(P)
    RR = gem.gen_fm_par_fm_(P)
    Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q,  W,time  = RR
    print("GEMM Time ",P,time,"## xslowdown", time/Ref,"MemTile", Tsub, "coretile",coresubvolume)
    Times.append([P, time])
    
    
    P =[77,1152,1152,acbits,webit,acbits ]
    Ref = gem.minimum_computation_time(P)
    RR = gem.gen_fm_par_fm_(P)
    Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q,  W,time  = RR
    print("GEMM Time ",P,time,"## xslowdown", time/Ref,"MemTile", Tsub, "coretile",coresubvolume)
    Times.append([P, time])
    
    P =[77,1152,768,acbits,webit,acbits ]
    Ref = gem.minimum_computation_time(P)
    RR = gem.gen_fm_par_fm_(P)
    Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q,  W,time  = RR
    print("GEMM Time ",P,time,"## xslowdown", time/Ref,"MemTile", Tsub, "coretile",coresubvolume)
    Times.append([P, time])

    print("MHA with %d heads ", heads)
    P = [96, 77, 64, bits]
    print(P)
    Ref = mha.minimum_computation_time(P,True)
    Ref2 = mha.minimum_computation_time_2(P,True)
    RT = mha.gen_fm_par_fm_(P,True)
    print(RT)
    time = mha.time_estimates(RT,P,True)
    print(P,"time", time, "ref", Ref, Ref2, "slowdown", time/Ref,time/Ref2)
    time = heads*time
    print("Total Time", time)
    Times.append([P, time])
    
    print("Post GEMM")
    
    P =[77,768,768,acbits,webit,acbits ]
    Ref = gem.minimum_computation_time(P)
    RR = gem.gen_fm_par_fm_(P)
    Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q,  W,time  = RR
    print("GEMM Time ",P,time,"## xslowdown", time/Ref,"MemTile", Tsub, "coretile",coresubvolume)
    Times.append([P, time])
    
    S= ln.gen_fm_par_fm_([77,768,8], multiple = True)
    Ref = ln.minimum_computation_time([77,768,8])    
    time = ln.time_estimates(S[0],[77,768,8])
    #print(self.compute(S[i]), "y, x,  wc, b, p,s", S[i])
    print("Layer Norm time", time,"Ref",Ref, "slowd",
          time/Ref,"ycore", S[0][0],"ymem", S[0][1])
    Times.append([[77,768,8], time])
    
    
    P =[77,768,768*4,acbits,webit,acbits ]
    Ref = gem.minimum_computation_time(P)
    RR = gem.gen_fm_par_fm_(P)
    Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q,  W,time  = RR
    print("GEMM Time ",P,time,"## xslowdown", time/Ref,"MemTile", Tsub, "coretile",coresubvolume)
    Times.append([P, time])
    
    P = [77,768*4,768, acbits,webit,acbits]
    Ref = gem.minimum_computation_time(P)
    RR = gem.gen_fm_par_fm_(P)
    Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q,  W,time  = RR
    print("GEMM Time ",P,time,"## xslowdown", time/Ref,"MemTile", Tsub, "coretile",coresubvolume)
    Times.append([P, time])
    
    S= ln.gen_fm_par_fm_([77,768,8], multiple = True)
    Ref = ln.minimum_computation_time([77,768,8])    
    time = ln.time_estimates(S[0],[77,768,8])
    #print(self.compute(S[i]), "y, x,  wc, b, p,s", S[i])
    print("Layer Norm time", time,"Ref",Ref, "slowd",
          time/Ref,"ycore", S[0][0],"ymem", S[0][1])

    Times.append([[77,768,8], time])

    for t in Times:
        print(t)
    return Times


def estimate_head_psf():
    mha = MHA()
    gem = Gemm()
    ln  = LayerNorm()

    bits   = 8
    acbits = 8
    webit  = 8
        
    heads = 12

    Times = []
    
    print("Pre multiplications")
    P =[512,512  ,1152,acbits,webit,acbits ]
    Ref = gem.minimum_computation_time(P)
    RR = gem.gen_fm_par_fm_(P)
    Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q,  W,time  = RR
    print("GEMM Time ",P,time,"## xslowdown", time/Ref,"MemTile", Tsub, "coretile",coresubvolume)
    Times.append([P, time])
    
    
    P =[512,1152,1152,acbits,webit,acbits ]
    Ref = gem.minimum_computation_time(P)
    RR = gem.gen_fm_par_fm_(P)
    Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q,  W,time  = RR
    print("GEMM Time ",P,time,"## xslowdown", time/Ref,"MemTile", Tsub, "coretile",coresubvolume)
    Times.append([P, time])
    
    
    P =[512,1152,1152,acbits,webit,acbits ]
    Ref = gem.minimum_computation_time(P)
    RR = gem.gen_fm_par_fm_(P)
    Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q,  W,time  = RR
    print("GEMM Time ",P,time,"## xslowdown", time/Ref,"MemTile", Tsub, "coretile",coresubvolume)
    Times.append([P, time])
    
    P =[512,1152,768,acbits,webit,acbits ]
    Ref = gem.minimum_computation_time(P)
    RR = gem.gen_fm_par_fm_(P)
    Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q,  W,time  = RR
    print("GEMM Time ",P,time,"## xslowdown", time/Ref,"MemTile", Tsub, "coretile",coresubvolume)
    Times.append([P, time])

    print("MHA with %d heads ", heads)
    P = [96, 512, 64,bits]
    print(P)
    Ref = mha.minimum_computation_time(P,True)
    Ref2 = mha.minimum_computation_time_2(P,True)
    RT = mha.gen_fm_par_fm_(P,True)
    print("RT", RT)
    time = mha.time_estimates(RT,P,True)
    print(P,"time", time, "ref", Ref, Ref2, "slowdown", time/Ref,time/Ref2)
    time = heads*time
    print("Total Time", time)
    Times.append([P, time])
    
    print("Post GEMM")
    
    P =[512,768,768,acbits,webit,acbits ]
    Ref = gem.minimum_computation_time(P)
    RR = gem.gen_fm_par_fm_(P)
    Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q,  W,time  = RR
    print("GEMM Time ",P,time,"## xslowdown", time/Ref,"MemTile", Tsub, "coretile",coresubvolume)
    Times.append([P, time])
    
    S= ln.gen_fm_par_fm_([512,768,8], multiple = True)
    Ref = ln.minimum_computation_time([512,768,8])    
    time = ln.time_estimates(S[0],[512,768,8])
    #print(self.compute(S[i]), "y, x,  wc, b, p,s", S[i])
    print("Layer Norm time", time,"Ref",Ref, "slowd",
          time/Ref,"ycore", S[0][0],"ymem", S[0][1])
    Times.append([[512,768,8], time])
    
    
    P =[512,768,768*4,acbits,webit,acbits ]
    Ref = gem.minimum_computation_time(P)
    RR = gem.gen_fm_par_fm_(P)
    Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q,  W,time  = RR
    print("GEMM Time ",P,time,"## xslowdown", time/Ref,"MemTile", Tsub, "coretile",coresubvolume)
    Times.append([P, time])
    
    P = [512,768*4,768, acbits,webit,acbits]
    Ref = gem.minimum_computation_time(P)
    RR = gem.gen_fm_par_fm_(P)
    Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q,  W,time  = RR
    print("GEMM Time ",P,time,"## xslowdown", time/Ref,"MemTile", Tsub, "coretile",coresubvolume)
    Times.append([P, time])
    
    S= ln.gen_fm_par_fm_([512,768,8], multiple = True)
    Ref = ln.minimum_computation_time([512,768,8])    
    time = ln.time_estimates(S[0],[512,768,8])
    #print(self.compute(S[i]), "y, x,  wc, b, p,s", S[i])
    print("Layer Norm time", time,"Ref",Ref, "slowd",
          time/Ref,"ycore", S[0][0],"ymem", S[0][1])

    Times.append([[512,768,8], time])

    for t in Times:
        print(t)
    return Times
    
if __name__ == "__main__":

#    estimate_head_psj()
#    estimate_head_psf()
#    test_conv()
    test_conv_2()
#    test_mha()
#    test_gemm_2(4,8,8)
#     test_mha_L(8)
#    test_layernorm()
