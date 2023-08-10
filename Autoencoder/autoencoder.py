

GPU   = False
dense = True
DEBUG= False
import os, sys, math

if "GPU" in os.environ:
  GPU = True
if "SPARSE" in os.environ:
  dense  = False
if "DEBUG" in os.environ:
  DEBUG = True

  
import rocmgpu as example
import procm

import scipy.sparse as sps
import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix
import time




if (dense) :
  def FC(A, x, b):
    if GPU:
      W = procm.fromsparse_dgemv(0, A,x,b)
      return W
    else:
      return  A.toarray()@x + b.toarray() 
  def ACT(x):
    y = np.round(np.maximum(x, 0))
    return y
else:
  def FC(A, x, b):
    if GPU:
      W = procm.dgemv_csr(0, A,x,b)
    else:
      W = A@x + b
    return W 
  def ACT(x):
    y = csr_matrix(np.round(np.maximum(x.toarray(), 0)))
    return y
  

def read_tensor(path):
  t_mtx = mmread(path)
  t = csr_matrix(t_mtx.transpose())#.transpose()
  return t



if __name__ == '__main__':

  BASE = "sparse-computations/AD08/training/model/ad08_0.9finSpar/"
  
  x        = read_tensor("sparse-computations/AD08/training/test_bench/sample_0/input_data.mtx")
  y_golden = read_tensor("sparse-computations/AD08/training/test_bench/sample_0/q_dense.mtx")
  G = []
  A = []
  b = [] 
  A_path = BASE + "q_dense_batchnorm_weights.mtx"
  b_path = BASE + "q_dense_batchnorm_bias.mtx"
  #import pdb; pdb.set_trace()
  A.append( read_tensor(A_path))
  b.append(read_tensor(b_path))
  for i in range(1,5) :
    A_path = BASE + "q_dense_batchnorm_" + str(i) + "_weights.mtx"
    b_path = BASE + "q_dense_batchnorm_" + str(i) + "_bias.mtx"
    #G_path = "sparse-computations/AD08/training/test_bench/sample_0/q_dense_batchnorm_" + str(i) + ".mtx"
    
    A.append( read_tensor(A_path))
    b.append(read_tensor(b_path))
    #G.append(read_tensor(G_path))

  A_path = BASE + "q_dense_weights.mtx"
  A.append(read_tensor(A_path))

  b_path = BASE + "q_dense_bias.mtx"
  b.append(read_tensor(b_path))
      
  start = time.time()
  for j in range(1):
    #import pdb; pdb.set_trace()
    y = ACT(FC(A[0], x.toarray() if dense else x, b[0]))
    
    for i in range(1,5):
      y = ACT(FC(A[i], y, b[i]))
    y = FC(A[5], y, b[5])
    
  end = time.time();

  print("time:", (end-start)/100)

  if DEBUG:
    print(y)
    print("Golden here:")
    print(y_golden)
    print("Difference")
    
  if dense:
    print(sum(sum(y-y_golden.toarray())))
  else:
    print((y-y_golden).sum().sum())
