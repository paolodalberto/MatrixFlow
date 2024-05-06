"""
// IFM L2
std::vector<uint32_t> ifmL2_dim = {8, 8, 512/8};
uint32_t ifmL2_ping_addr = 256*1024;
std::vector<adf::access_pattern> ifm_pattern_in = {
    adf::tiling({.buffer_dimension={8, 8, 512/8}, .tiling_dimension={8, 1, 1}, .offset={0, 0, 0}, .tile_traversal={{.dimension=2, .stride=1, .wrap=512/8}, {.dimension=1, .stride=1, .wrap=8}}})
};
std::vector<adf::access_pattern> ifm_pattern_out = {
    adf::tiling(
    {
      .buffer_dimension={8, 8, 512/8}, 
      .tiling_dimension={8, 8, 128/8}, 
      .offset={0, 0, 0}, 
      .tile_traversal={
           {.dimension=2, .stride=128/8, .wrap=itersL2I}
      }, 
      .packet_port_id=-1, .repetition=itersL2W
    })
};

// IFM L3
std::vector<adf::access_pattern> ifm_ddr_pattern_out = {
    adf::tiling({.buffer_dimension={mxI, mxH}, .tiling_dimension={512, 8}, .offset={0, 0}, .tile_traversal={{.dimension=1, .stride=8, .wrap=itersL2H}}}),
    adf::tiling({.buffer_dimension={mxI, mxH}, .tiling_dimension={512, 8}, .offset={512, 0}, .tile_traversal={{.dimension=1, .stride=8, .wrap=itersL2H}}}),
    adf::tiling({.buffer_dimension={mxI, mxH}, .tiling_dimension={512, 8}, .offset={1024, 0}, .tile_traversal={{.dimension=1, .stride=8, .wrap=itersL2H}}}),
    adf::tiling({.buffer_dimension={mxI, mxH}, .tiling_dimension={512, 8}, .offset={1536, 0}, .tile_traversal={{.dimension=1, .stride=8, .wrap=itersL2H}}})
};

// WTS L2
std::vector<uint32_t> wtsL2_dim = {mxI*64};
uint32_t wtsL2_ping_addr = 0;
uint32_t wtsL2_pong_addr = 128*1024;
std::vector<adf::access_pattern> wts_pattern_in = {
    adf::tiling({.buffer_dimension={mxI*64}, .tiling_dimension={(mxI/2)*64}, .offset={0}}),
    adf::tiling({.buffer_dimension={mxI*64}, .tiling_dimension={(mxI/2)*64}, .offset={(mxI/2)*64}})
};
std::vector<adf::access_pattern> wts_pattern_out = {
    adf::tiling({.buffer_dimension={mxI*64}, .tiling_dimension={128*64}, .offset={0}, .tile_traversal={{.dimension=0, .stride=128*64, .wrap=itersL2I}}}),
    adf::tiling({.buffer_dimension={mxI*64}, .tiling_dimension={128*64}, .offset={512*64}, .tile_traversal={{.dimension=0, .stride=128*64, .wrap=itersL2I}}}),
    adf::tiling({.buffer_dimension={mxI*64}, .tiling_dimension={128*64}, .offset={1024*64}, .tile_traversal={{.dimension=0, .stride=128*64, .wrap=itersL2I}}}),
    adf::tiling({.buffer_dimension={mxI*64}, .tiling_dimension={128*64}, .offset={1536*64}, .tile_traversal={{.dimension=0, .stride=128*64, .wrap=itersL2I}}})
};

// WTS L3
#if 0   // Original W8 format version
std::vector<adf::access_pattern> wts_ddr_pattern_out = {
    adf::tiling({.buffer_dimension={8, mxI, mxW/8}, .tiling_dimension={8, mxI/2, 64/8}, .offset={0, 0, 0}, .tile_traversal={{.dimension=2, .stride=64/8, .wrap=itersL2W}}, .packet_port_id=-1, .repetition=itersL2H}),
    adf::tiling({.buffer_dimension={8, mxI, mxW/8}, .tiling_dimension={8, mxI/2, 64/8}, .offset={0, (int)mxI/2, 0}, .tile_traversal={{.dimension=2, .stride=64/8, .wrap=itersL2W}}, .packet_port_id=-1, .repetition=itersL2H}),
    adf::tiling({.buffer_dimension={8, mxI, mxW/8}, .tiling_dimension={8, mxI/2, 64/8}, .offset={0, 0, (512/8)}, .tile_traversal={{.dimension=2, .stride=64/8, .wrap=itersL2W}}, .packet_port_id=-1, .repetition=itersL2H}),
    adf::tiling({.buffer_dimension={8, mxI, mxW/8}, .tiling_dimension={8, mxI/2, 64/8}, .offset={0, (int)mxI/2, (512/8)}, .tile_traversal={{.dimension=2, .stride=64/8, .wrap=itersL2W}}, .packet_port_id=-1, .repetition=itersL2H}),
    adf::tiling({.buffer_dimension={8, mxI, mxW/8}, .tiling_dimension={8, mxI/2, 64/8}, .offset={0, 0, (512/8)*2}, .tile_traversal={{.dimension=2, .stride=64/8, .wrap=itersL2W}}, .packet_port_id=-1, .repetition=itersL2H}),
    adf::tiling({.buffer_dimension={8, mxI, mxW/8}, .tiling_dimension={8, mxI/2, 64/8}, .offset={0, (int)mxI/2, (512/8)*2}, .tile_traversal={{.dimension=2, .stride=64/8, .wrap=itersL2W}}, .packet_port_id=-1, .repetition=itersL2H}),
    adf::tiling({.buffer_dimension={8, mxI, mxW/8}, .tiling_dimension={8, mxI/2, 64/8}, .offset={0, 0, (512/8)*3}, .tile_traversal={{.dimension=2, .stride=64/8, .wrap=itersL2W}}, .packet_port_id=-1, .repetition=itersL2H}),
    adf::tiling({.buffer_dimension={8, mxI, mxW/8}, .tiling_dimension={8, mxI/2, 64/8}, .offset={0, (int)mxI/2, (512/8)*3}, .tile_traversal={{.dimension=2, .stride=64/8, .wrap=itersL2W}}, .packet_port_id=-1, .repetition=itersL2H})
};
#else  // prearranged 1D WTS version
std::vector<adf::access_pattern> wts_ddr_pattern_out = {
    adf::tiling({.buffer_dimension={mxI*mxW}, .tiling_dimension={(mxI/2)*512}, .offset={0}}),
    adf::tiling({.buffer_dimension={mxI*mxW}, .tiling_dimension={(mxI/2)*512}, .offset={(mxI/2)*512}}),
    adf::tiling({.buffer_dimension={mxI*mxW}, .tiling_dimension={(mxI/2)*512}, .offset={(mxI/2)*512*2}}),
    adf::tiling({.buffer_dimension={mxI*mxW}, .tiling_dimension={(mxI/2)*512}, .offset={(mxI/2)*512*3}}),
    adf::tiling({.buffer_dimension={mxI*mxW}, .tiling_dimension={(mxI/2)*512}, .offset={(mxI/2)*512*4}}),
    adf::tiling({.buffer_dimension={mxI*mxW}, .tiling_dimension={(mxI/2)*512}, .offset={(mxI/2)*512*5}}),
    adf::tiling({.buffer_dimension={mxI*mxW}, .tiling_dimension={(mxI/2)*512}, .offset={(mxI/2)*512*6}}),
    adf::tiling({.buffer_dimension={mxI*mxW}, .tiling_dimension={(mxI/2)*512}, .offset={(mxI/2)*512*7}})
};
#endif

// OFM L2
std::vector<uint32_t> ofmL2_dim = {64, 8};
uint32_t ofmL2_ping_addr = 260*1024;
uint32_t ofmL2_pong_addr = 262*1024;
std::vector<adf::access_pattern> ofm_pattern_in = {
    adf::tiling({.buffer_dimension={8, 64/8, 8}, .tiling_dimension={8, 1, 1}, .offset={0, 0, 0}, .tile_traversal={{.dimension=2, .stride=1, .wrap=8}, {.dimension=1, .stride=1, .wrap=64/8}}})
};
std::vector<adf::access_pattern> ofm_pattern_out = {
    adf::tiling({.buffer_dimension={64, 8}, .tiling_dimension={64, 8}, .offset={0, 0}})
};

// OFM L3
std::vector<uint32_t> ofm_ddr_dim = {2048, 8};
std::vector<adf::access_pattern> ofm_ddr_pattern_in = {
    adf::tiling({.buffer_dimension={mxW, mxH}, .tiling_dimension={64, 8}, .offset={0, 0}, .tile_traversal={{.dimension=0, .stride=64, .wrap=itersL2W}, {.dimension=1, .stride=8, .wrap=itersL2H}}}),
    adf::tiling({.buffer_dimension={mxW, mxH}, .tiling_dimension={64, 8}, .offset={512, 0}, .tile_traversal={{.dimension=0, .stride=64, .wrap=itersL2W}, {.dimension=1, .stride=8, .wrap=itersL2H}}}),
    adf::tiling({.buffer_dimension={mxW, mxH}, .tiling_dimension={64, 8}, .offset={1024, 0}, .tile_traversal={{.dimension=0, .stride=64, .wrap=itersL2W}, {.dimension=1, .stride=8, .wrap=itersL2H}}}),
    adf::tiling({.buffer_dimension={mxW, mxH}, .tiling_dimension={64, 8}, .offset={1536, 0}, .tile_traversal={{.dimension=0, .stride=64, .wrap=itersL2W}, {.dimension=1, .stride=8, .wrap=itersL2H}}})
};
"""

# Python code to demonstrate namedtuple()
from collections import namedtuple
from  Matrices.matrices import Matrix, PartitionMatrix
import math


Traversal = namedtuple("Traversal", ['dimension', 'stride', 'wrap'])
Tiling    = namedtuple(
    "Tiling",
    [
        'buffer_dimension', # size per dimension
        'tiling_dimension', # tile size per dimension  
        'offset',           # offset per dimension    
        'tile_traversal',   # per dimension Traversal
        'packet_port_id' ,
        'repetition'        # how many time this will be executed (synch ? DB?)  
    ]
)



class Level:
    def __init__(
            self,
            name : str,
            level : int,
            size_per_part   : int = 512*1024**3 , # 512MB * 4 = 2 GB 
            parts           : int = 4 ,
            input_channels  : int = 2 ,
            output_channels : int = 2,
             
    ):
        self.name = name
        self.level = level
        self.mem = [ size_per_part for i in range(parts) ]
        self.inp = [ [{} for i in range(input_channels)  ] for i in range(parts) ]   
        self.out = [ [{} for i in range(output_channels) ] for i in range(parts) ]   

    def __str__(self):
        return self.name+ (" L: %d " % self.level) +str(self.mem)

    def copy(self, extra:str='x'):
        R = Level(self.name,self.level)
        R.mem = [ i for i in self.mem]
        return R
    
    def parts_number(self) : return len(self.mem)

    def strides(self, shapes:list )-> list:
        s =1
        R = []
        for i in range(len(shapes)):
            R.append(s)
            s *= shapes[i]

        
        return R
    def offsets(self, dim, shapes:list, full: int =2 )-> list:
        S = self.strides(shapes)
        R = []
        B=1
        for b in range(dim):
            B*=S[b]
        D = shapes[dim]

        Os = full*(len(self.mem))
        Qs = math.ceil(D/Os)
        for i in range(Os):
            R.append(B*i*Qs)
        
        return R
    ## 
    def traversal(
            self,
            shapes : list, ## sizes of the main matrix 
            tile : list    ## sizes of one tile.
    ) -> list:
        R = []
        s =1
        for i in range(len(shapes)):
            R.append(Traversal(i,tile[i], math.ceil(shapes[i]/tile[i])))

            
        return R

        



               
L3 = Level("DDR", 3)   
L2 = Level("MemTile", 2, 512*1024, 4) # 4 MemTiles 2MB   
L1 = Level("DDR", 1, 16*1024,2)   
    


class MemoryHierarchTensors:
    def __init__(
            self,
            name : str,
            MP : PartitionMatrix, ## this is a logical partition 
    ):
        self.name = name
        self.MP = MP
        
        matshape = [ i for i in reversed(MP.logicalshape)] 
        dimensions = len(matshape)
        A = MP.original
        Shape = [ A.max[i] -A.min[i] for i in reversed(range(dimensions))]
        
        buffer_dimension = Shape
        repetition       = len(MP.l)*len(MP.l[0]) 
        tiling_dimension = matshape
        offset           = 0 #[ 0 for i in matshape] 
        tile_traversal   = []#Level.traversal(Level,Shape,matshape)
        
        self.T = Tiling(buffer_dimension,tiling_dimension,offset,tile_traversal,-1,repetition)
        
    
    ## this means that one partition will be split by parts in the 
    def read_tiling_by_parts(
            self,
            dim : int,           # dimension we split into parts
            level : Level,       # Level memory 
            channels : int = 2   # all channnels 
    ) -> Tiling:
        
        matshape =  [ i for i in reversed(self.MP.logicalshape)]
        dimensions = len(matshape)
        parts = level.parts_number()
        A = self.MP.original
        Shape = [ A.max[i] -A.min[i] for i in reversed(range(dimensions))]

        matshape[dim] //= parts
        
        
        buffer_dimension = [i for i in Shape]
        buffer_dimension[dim] //= parts

               
        repetition       = len(self.MP.l)*len(self.MP.l[0]) 
        tiling_dimension = matshape
        offset           = level.offsets(dim,  Shape, channels)
        tile_traversal   = level.traversal(buffer_dimension,matshape)

        R = [] 
        for i in range(len(level.mem)*channels):
            if i%channels==0:
                R.append("##### %d" % (i//channels))
            R.append(     
                Tiling(
                    buffer_dimension,
                    tiling_dimension,
                    offset[i],
                    tile_traversal,
                    -1,
                    repetition)
            )
        

        return R


