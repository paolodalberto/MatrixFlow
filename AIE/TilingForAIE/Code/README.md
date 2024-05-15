# Computing optimal tiles: codes and examples. 


This is not a set of python books it is a set of files where there is
a __main__ collecting examples how to run and use these classes.  The
idea is straightforward: one class one operator, then we give a
problem size, by a simplified tuple describing the minimum shapes, and
it will provide the optimal tiles (using the same tuple formalism),
the set of valid solutions, and example of cost functions.

## HW a simplified world

The Overlay is an array of AIE organized by ROWs and COLS.

```python
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
```

There is not description of channel connections or cascade. We assume
row and column connections and DDR memtile connection. For example we
provide three classes for GEMM: Gemm, GemmLib, GemmCascade.

- Gemm: it assumes a spatial division of the computation so that C is
  written only once, there is no cascade, and the accumulation is done
  in AIE core.

- GemmLib: it assumes the connection above but we would like to
  remember and reuse the valid solutions to a set of GEMMs. This is
  truly an exploration space not only for the fastest configurations
  but this can give an idea of the most common.

- GemmCascade: this is to explore the solution when AIE core are
  connected in the columns and thus it implies the split by "k" across
  row and then accumulate. There is something fuzzy here, we can find
  a reproduce the current valid solutions provided in practice.

We provide similar solutions for Convolution, MHA, and Layer Norm. You
can use these to have full estimate and optimal tiling for a complete
HEAD: Pre-Multiplication preparing Q,K,V, 12 MHA, Projections, and
Layer Norms.


## Optimal Tiling

We have a simple problem description 
``` python 

    gem = Gemm()

    bits   = 8
    acbits = 8
    webit  = 8

    ##     The problem size for a GEMM is [m,n,k,atypem, btype, ctype)
    ##     C = m x n  ctype in bits
    ##     A = m x k  atype in bits
    ##     B = k x n  atype in bits 

    P =[512,512  ,1152,acbits,webit,acbits ]
    Ref = gem.minimum_computation_time(P)
    RR = gem.gen_fm_par_fm_(P)

    ## MemTile = Tsub = [mm,mn,mk,atype, btype,ctype] ,
    ## coresubvolume = core = [cm,cn,ck,atype, btype,ctype] 
    Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q,  W,time  = RR
    print("GEMM Time ",P,time,"## xslowdown", time/Ref,"MemTile", Tsub, "coretile",coresubvolume)

```

The idea is to provide a reference minimum time considering
computation and communication limitations. Given a problem we provide
the optimal solution. We provide also tools to execute the tiled
computation using Python to have a naive correctness attempt. 

In general we provide a solution set (W) and then we can just use a cost function of the single valid solution 
```python
gemm = Gemm() 
...
for w in W[0:3]: print(gemm.time(w,P,1),gemm.compute(w),gemm.ratio(w),w)
```

## Time estimates and cost functions

The design of the time estimate, thus the practical cost function, is
at the core of the exploration. Our HW constraints are simple to
understand and they are space requirements basically (i.e., basic
understanding of algorithms and HW). Some solutions may be considered
valid by our constraints but we are not certain. We can choose the
cost function so that time will be prohibitive.

For example in Gemm class, for a given valid solution, we may have
multiple algorithms, and multiple schedules. The time-estimate
function considers these choices and makes context based decision to
represent the execution. For example, we may have to choose double
buffering both A and B, or either one (we prefer to double buffer B
because in this context is larger)