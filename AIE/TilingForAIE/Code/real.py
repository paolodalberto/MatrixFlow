import pandas as pd
import pdb
from code_gemm_cascade import GemmCascade
from code_gemm import Gemm
import math 
import matplotlib.pyplot as plt

F = (1)*10**9 
BC = 1 * 8*(2**30)
DM = 1 * 8*(2**30)
gemm = GemmCascade(COLS=4, m_align=8,
                   frequency=F ,
                   to_core_channel_bandwidth_gbits=  BC,
                   to_mem_channel_bandwidth_gbits =  DM
                   )

gemm_2 = Gemm(COLS=4, m_align=8,
              frequency= F ,
              to_core_channel_bandwidth_gbits=  BC,
              to_mem_channel_bandwidth_gbits =  DM
              )


wbits  = 4
inbits = 16
oubits = 16

def accurate(row):
    c= [ min(int(row['M']),16),64,128,inbits,wbits,oubits]
    P = [int(row['M']),int(row['N']), int(row['K']),inbits,wbits,oubits]
    
    #S = [int(row['M']),64, 128*gemm.ROWS,inbits,wbits,oubits]
    t,d = gemm.time(c,P)
    
    return row['OPS']/t/10**12   
def accurate_v(row):
    #print(row)
    c= [ min(int(row['M']),16),64,128,inbits,wbits,oubits]
    P = [int(row['M']),int(row['N']), int(row['K']),inbits,wbits,oubits]
    
    t,d = gemm.time(c,P,v = True)
    pdb.set_trace()
    return row['OPS']/t/10**12   
def accurate_one(row):
    #print(row)
    
    P = [int(row['M']),int(row['N']), int(row['K']),inbits,wbits,oubits]
    t = gemm.minimum_computation_time(P)

    
    return row['OPS']/t/10**12   
def accurate_two(row):
    #print(row)
    
    P = [int(row['M']),int(row['N']), int(row['K']),inbits,wbits,oubits]
    #Ref = gemm.minimum_computation_time(P)
    RR = gemm.gen_fm_par_fm_(P)
    Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q,  W,time  = RR
    
    return row['OPS']/time/10**12   

def accurate_three(row):
    #print(row)
    
    P = [int(row['M']),int(row['N']), int(row['K']),inbits,wbits,oubits]
    Ref = gemm_2.minimum_computation_time(P)
    RR = gemm_2.gen_fm_par_fm_(P)

    
    Tsub, coresubvolume,dba,dbb, dba_dbb, Ksplit,FatC, Q,  W,time  = RR
    #    if False and time <10:
    #        pdb.set_trace()
    #        RR = gemm_2.gen_fm_par_fm_(P)
    #    if  int(row['M'])==16:
    print(Ref, time, time/Ref)
    if Ref>time:
        pdb.set_trace()
        time_2 = gemm_2.time(W[0],P,Q)
        Ref = gemm_2.minimum_computation_time(P)
        pdb.set_trace()
    
    return row['OPS']//time/10**12   



def plotting(spamreader2):
    
    spamreader= spamreader2.sort_values('OPS')
    print(spamreader)
#    spamreader= spamreader2.reindex(columns=['OPS','TFLOPS','I_TFLOPS','II_TFLOPS','IV_TFLOPS'])

    
    pdb.set_trace()
    plt.plot(range(spamreader.shape[0]),#spamreader['OPS'],
             spamreader['TFLOPS'],     c ="red", marker="+", label='measured')
    plt.plot(range(spamreader.shape[0]),#spamreader['OPS'],
             spamreader['I_TFLOPS'],   c ="blue",marker="+",label='Estimated [8,64,128]')
    plt.plot(range(spamreader.shape[0]),#spamreader['OPS'],
             spamreader['II_TFLOPS'],  c ="green",label='Optimal Shreyas')
    plt.plot(range(spamreader.shape[0]),#spamreader['OPS'],
        spamreader['III_TFLOPS'], c ="orange",label='C optimal')
    plt.plot(range(spamreader.shape[0]),#spamreader['OPS'],
                spamreader['IV_TFLOPS'], c ="black",marker="+",label='UpperBound')
    plt.xlabel("#OPS")
    plt.ylabel("TFLOPS")
    plt.title("4x4 GEMM %d,%d,%d bits" %(inbits, wbits,oubits) )
    plt.legend()
    #plt.show()
    plt.savefig("gemm.png")
    plt.show()
    pdb.set_trace()


if __name__ == "__main__":

    #pdb.set_trace()
    with open('Code/golden_w3a16_stx.csv','r') as csvfile:
        spamreader = pd.read_csv(csvfile)

    del spamreader['A_Pad_time(ns)']
    del spamreader['C_Pad_time(ns)']
    del spamreader['C_depad_time(ns)']
    del spamreader['CPU_accum_time(ns)']

    

    print(spamreader)
    spamreader['OPS'] = 2*spamreader['M']*spamreader['N']*spamreader['K']
    spamreader['TFLOPS'] = spamreader['OPS']/(spamreader['run_aie_time(ns)']/10**9)/10**12 #2*spamreader['M']*spamreader['N']*spamreader['K']/spamreader['run_aie_time(ns)']/1000

    #pdb.set_trace()
    gemm.Mtrickery=[64,64,128]
    spamreader['I_TFLOPS'] = spamreader.apply(accurate,axis=1)  
    #pdb.set_trace()
    gemm.Mtrickery=None
    spamreader['II_TFLOPS'] = spamreader.apply(accurate_two,axis=1)  
    print(spamreader)
    #pdb.set_trace()
    #accurate_v(spamreader.iloc[9])
    #accurate_v(spamreader.iloc[8])

    spamreader['III_TFLOPS'] = spamreader.apply(accurate_three,axis=1)  
    spamreader['IV_TFLOPS'] = spamreader.apply(accurate_one,axis=1)  

    spamreader['ratio2'] =  spamreader['run_aie_time(ns)']/10**9 /spamreader['I_TFLOPS']


    
#    print(spamreader)
    plotting(spamreader)
    
