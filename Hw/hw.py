class Memory:
    def __init__(self,
                 name :str, 
                 size : int):
        self.name  = name
        self.size = size  # size in bytes
        self.free = [0, size-1]
        
    def space(self): return self.size
    def __str__(self):
        return "Memory %s Size %d" % (self.name, self.size) 

class PE:
    def __init__(self,
                 name : str,
                 memory : Memory):
        self.name = name 
        self.internal_memory = memory
        self.graph = None 
        
    def space(self): return self.internal_memory.space()
    def compute(self):
        print(self)
        return self.graph.compute() if self.graph else None 
    def __str__(self):
        return "PE %s %s " % (self.name, str(self.internal_memory))

class AbstractHW:
    def __init__(self,
                 name : str,
                 pes : list = [
                     PE(str(i), Memory(str(i),1*(2**10)**2))
                     for i in range(2)
                 ], 
                 memory : Memory = Memory('main', 16*(2**10)**3)
    ):
        self.name = name 
        self.memory = memory
        self.pes = pes 

    def __str__(self):
        ps = [str(pe) for pe in self.pes ]
        s = ""
        for p in ps:
            s += p+"\n"
        m  = str(self.memory)
        return "HW %s " % self.name + "\n" + m +"\n" + s   

    def get_pe(self, i : int = 0) -> PE :
        if  i in [0,len(self.pe)]:
            return self.pe[i]
        return self.pe[0]
                 
    
    def compute(self):
        for pe in self.pes:
            print(pe.compute())
