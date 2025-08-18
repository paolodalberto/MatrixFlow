## WHAT YOU NEED 
# 
# You will need three files 
# * code_gemm.py (numerical implementations and other oddities)
# * load.py   (sand box)
# * quant.py  (AI quantization code)
#
# and you will need the a directory where the Phi-3.5 data is avalable
# ... likely I will give you a zip file with everything in it
# 
# ls phi/Phi-3.5-mini -lr
# total 518116
# -rw-r--r-- 1 paolod hd 4155947 Aug 12 18:04 gqo_4_9_v_dump.txt
# -rw-r--r-- 1 paolod hd 3980971 Aug 12 18:04 gqo_4_9_q_dump.txt
# -rw-r--r-- 1 paolod hd 4513908 Aug 12 18:04 gqo_4_9_output_dump.txt
# -rw-r--r-- 1 paolod hd 3939357 Aug 12 18:04 gqo_4_9_k_dump.txt
#
# The Phi-3.5 directory has 32 layers we
# and we have the DDR dumps of Q,K,V and the final result.
#
# There is also a rotation for each layer and thus I personally cannot
# reproduce the output tensor available here. We will explain how we
# compare the result in the sandbox, code and in this short tutorial
#

## SAND BOX how to call load.py (python3)
#
# 1) take a look at the data and how I think it is organized


python load.py -d true ## --distribution true

# Now you should have 32 png pictures. For example. 

feh i31.png

# How we produced this? We take the for example gqo_4_31_q_dump.txt
# and create a tensor shaped as Q = 128x3072, we split the tensor Q by
# 96 columns (inner dimensions) so that to produce 32 Q_i 128x96
# (partitions) and do the same for K and V. The picture i31.png is an
# average of Q_i, K_i, V_i showing the biased location of the maxima
# in the matrices .. we could and you could use L2 to make it more
# sensitive to outliers but for me I wanted to validate the zebra like
# stripes of the values.
#
# In principle, Q and K should use smooth quant or something similar
# to reduce peaks and valleys so that quantization and computation are
# a little smoother

# 2) take a look how quantization affect the representation of the
#    matrices. We compare layer 0 and layer 17 with different "block
#    quantizations" and we present once again the average but in this
#    case the average absolute error and its locations.
#
# You will play with 0 and 17, with quantization 16x16 and 128x1 and with and without SAGE normalization
# quant( 0,16,16), quant( 0,16,16,Norm=False),
# quant(0, 128,1), quant(0, 128,1,Norm=False),
# quant(17,16,16), quant(17,16,16,Norm=False),
# quant(17,128,1), quant(17,128,1,Norm=False)

python load.py -q true # --quantization true

# during the execution you will see images and summary of the maximum
# absolute error you can see or better I want you to notice that the
# "column" quantization provide the better approximation and this
# should provide also the most accurate computation.


# 3) Now for the numerical comparison. We take every layer and compare
#    the numerical properties for different algorithms. Once again we
#    provide summary and plot for the error. 


python load.py -c true -s false  # --compare true --sequential false 

# For example: summary 
# Averages L1 errors
# scipy 64     L1 9.745e-02 MAX 7.574e+00
# scipy        L1 1.405e-04 MAX 6.284e-02
# sepa         L1 1.491e-04 MAX 6.333e-02
# block        L1 1.491e-04 MAX 6.333e-02
# sage         L1 3.450e-03 MAX 1.228e+00

# Explanation of The comparison is a little tricky 
# 
# There is a rotation in every layer so we create a reference using
# floating point precision and numpy and scipy only implementation.
#
# scipy 64 is the difference between our computation and the R dump (with rotation).
# scipy is the comparison using float16 vs our reference float32
# sepa  is the non blocked computation (another reference)
# block is the flas attention like computation in float16
# sage  is the sage attention with quantization to 8 bits (you can change the number of bit if you like)


# for example picture 
feh block.png

#### NOTE NOTE NOTE
# Please delve into the code of sage or ask questions even with 10
# bits precision the situation improve especially for layer 0-16 but
# the improvement is lesser for layer 16-31 ...
#




