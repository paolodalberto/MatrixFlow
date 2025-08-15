



import numpy 

def quantize(x_float, num_bits=8, signed=False):
    """
    Quantizes a floating-point value to an 8-bit integer.

    Args:
        x_float (float): The floating-point value to quantize.
        min_val (float): The minimum value of the float range.
        max_val (float): The maximum value of the float range.
        num_bits (int): The number of bits for the integer representation (default: 8).
        signed (bool): Whether the integer representation is signed (default: True).

    Returns:
        int: The quantized 8-bit integer value.
    """
    # Calculate the range of the float and integer types
    
    min_val = numpy.min(x_float)
    max_val = numpy.max(x_float)

    if min_val == max_val:
        if min_val > 0:
            min_val = 0.0
        else:
            max_val= 0.0
    
    float_range = max_val - min_val
    
    if signed:
        int_min = -(2**(num_bits - 1))
        int_max = (2**(num_bits - 1)) - 1
    else:
        int_min = 0
        int_max = (2**num_bits) - 1

    int_range = int_max - int_min

    # Calculate the scaling factor (S) and zero-point (Z)
    scale = float_range / int_range
    if float_range!=0.0:
        zero_point = int_min - round(min_val / scale)
    
        # Quantize the float value and clip it to the integer range
        x_q = numpy.round(x_float / scale + zero_point)
        x_q = numpy.maximum(int_min, numpy.minimum(int_max, x_q)) # Clip to int8 range
    else:
        zero_point = 0
        x_q = numpy.round(x_float)
    return (x_q).astype(numpy.int64), scale, zero_point

def dequantize(X):
    x_q, scale, zero_point = X 
    return  scale * (x_q - zero_point)




if __name__ == "__main__":


    SIZE =256

    Q = numpy.random.rand(SIZE,SIZE) -1/2
    K = numpy.random.rand(SIZE,SIZE) -1/2
    
    R = numpy.matmul(Q,K)

    Qz,qs,qz = quantize(Q)
    print("MAX", numpy.max(Qz),numpy.min(Qz))
    print(f"Scale and zero: {qs} {qz}")
    Kz,ks,kz = quantize(K)
    print("MAX", numpy.max(Kz),numpy.min(Kz))
    print(f"Scale and zero: {ks} {kz}")

    QQ = dequantize([Qz,qs,qz])
    print("MAX", numpy.max(QQ))
    
    KK = dequantize([Kz,ks,kz])
    print("MAX", numpy.max(KK))
    
    
    print(numpy.max(numpy.fabs(Q-QQ)))
    print(numpy.max(numpy.fabs(K-KK)))
    
    #import pdb; pdb.set_trace()
    R1 = numpy.matmul(QQ,KK)

    print("R", numpy.max(numpy.fabs(R-R1)))
    
    RI = numpy.matmul(Qz-qz,Kz-kz)
    print("MAX", numpy.max(RI), numpy.min(RI),(2**7,2**15, 2**31))
    RQ = RI*qs*ks
    print(numpy.max(numpy.fabs(R-RQ)))
