import matplotlib.pyplot as plt
import graphviz
import time
import seaborn as sns
import re

Base = """
# type  alg  size gflops    max err
# pyt   reg   360 2.616e+01 0.00000e+00
# par     2   360 2.062e+01 6.66134e-14
# par     3   360 1.497e+01 6.97220e-14
# par   2x2   360 9.326e+00 8.61533e-14
# par   3x3   360 9.747e-01 1.67422e-13
# par   2x3   360 3.250e+00 1.24345e-13
# par   3x2   360 3.482e+00 1.11466e-13
# par 2x2x3   360 3.555e-01 3.07310e-13
# par 2x3x2   360 3.469e-01 2.54907e-13
# par 3x2x2   360 3.555e-01 3.53495e-13
# one     2   360 2.090e+01 6.75016e-14
# one     3   360 1.395e+01 6.43929e-14
# one   2x2   360 8.054e+00 9.05942e-14
# one   3x3   360 7.932e-01 1.98952e-13
# one   2x3   360 2.891e+00 1.16351e-13
# one   3x2   360 2.884e+00 1.01252e-13
# one 2x2x3   360 2.835e-01 2.87770e-13
# one 2x3x2   360 2.821e-01 2.46025e-13
# one 3x2x2   360 2.815e-01 2.33591e-13
"""



def initialize():

    KEYS = [ 
        '#pytreg',
        #'#par2',
        #'#par3',
        #'#par2x2',
        #'#par3x3',
        #'#par2x3',
        #'#par3x2',            
        #'#par2x2x3',
        #'#par2x3x2',
        #'#par3x2x2',
        '#one2',
        '#one3',
        '#one2x2',
        '#one3x3',
        '#one2x3',
        '#one3x2',
        '#one2x2x3',            
        '#one2x3x2',          
        '#one3x2x2'
    ]
    D = {}
    for k in KEYS:
        D[k] ={
            'size' : [],
            'gflops' : [],
            'max_err' : []
        } 
    return D

def parse_row(line : str, D):
    items = re.split("\s+",line)

    if len(items)>4:
        alg = items[0]+items[1]+items[2]
        size = int(items[3])
        gflops = float(items[4])
        max_err = float(items[5])
        if alg in D:
            D[alg]['size'].append(size)
            D[alg]['gflops'].append(gflops)
            D[alg]['max_err'].append(max_err)



def read_file(filename : str = "out.1"):

    file  = None
    with open(filename,'r') as F:
        lines = F.read()
        F.close()

        
        
    D = initialize()
    
    lines = lines.split("\n")
    for l in lines:
        print(l)
        if l.find("# type  alg")>=0: continue
        parse_row(l,D)

    return D

def plotting(D : dict,title : str = ""):
    Algs = D.keys()

    plt.title(title)
    plt.xlabel("matrix size")
    plt.ylabel("Standarized GFLOPS")
    
    for k in Algs:
        x = D[k]['size'] 
        y = D[k]['gflops']

        #x1 = []
        #for e in x: x1.insert(0,e)
        
        #y1 = [];
        #for e in y: y1.insert(0,e)
        
        if k == '#pytreg':
            plt.plot(x,y, label=k, linestyle = 'dotted', linewidth=3)
        else:
            plt.plot(x,y, label=k)

    plt.gca().invert_xaxis()
    plt.grid()
    plt.legend(loc='lower left')
    plt.show()
    


import sys
import argparse
import gc

Line = "# %s %5s %5d %1.3e %1.5e" 


if __name__ == "__main__":

        
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", help="filename", type =str, default ='out.1')
    args = parser.parse_args()


#    import pdb; pdb.set_trace()
    D = read_file(args.filename)
    plotting(D,args.filename)
