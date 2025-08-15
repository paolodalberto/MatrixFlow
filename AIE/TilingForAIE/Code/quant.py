



import numpy 

def quantize_to_int8(x_float, min_val, max_val, num_bits=8, signed=False):
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
    zero_point = 0 # int_min - round(min_val / scale)
    #import pdb; pdb.set_trace()
    # Quantize the float value and clip it to the integer range
    x_q = numpy.round(x_float / scale + zero_point)
    x_q = numpy.maximum(int_min, numpy.minimum(int_max, x_q)) # Clip to int8 range

    return (x_q).astype(numpy.int16), scale, zero_point

def dequantize_from_int8(x_q, scale, zero_point):
    """
    Dequantizes an 8-bit integer value back to a floating-point representation.

    Args:

    Returns:
        float: The dequantized floating-point value.
    """
    # Calculate the range of the float and integer types

    # Dequantize the integer value
    x_float = scale * (x_q - zero_point)
    return x_float




if __name__ == "__main__":

    # Example usage
    float_value = 0.75
    min_range = -1.0
    max_range = 1.0
    
    # Quantize
    quantized_value,s,z = quantize_to_int8(float_value, min_range, max_range)
    print(f"Original float: {float_value}, Quantized int8: {quantized_value}")
    
    # Dequantize
    dequantized_value = dequantize_from_int8(quantized_value, s,z)
    print(f"Dequantized float: {dequantized_value}")
    
    # Example with a numpy array
    float_array = numpy.array([-0.8, 0.2, 0.9, -0.1], dtype=numpy.float32)
    
    quantized_array,scale, zero = quantize_to_int8(float_array, numpy.min(float_array), numpy.max(float_array))
    print(f"Original float array: {float_array}")
    print(f"Quantized int8 array: {quantized_array} {scale} {zero}")
    
    dequantized_array = dequantize_from_int8(quantized_array, scale,zero) 
    print(f"Dequantized float array: {dequantized_array}")

    SIZE =50

    Q = numpy.random.rand(SIZE,SIZE)
    K = numpy.random.rand(SIZE,SIZE)
    
    R = numpy.matmul(Q,K)

    Qz,qs,qz = quantize_to_int8(Q,numpy.min(Q),numpy.max(Q))
    print(f"Scale and zero: {qs} {qz}")
    Kz,ks,kz = quantize_to_int8(K,numpy.min(K),numpy.max(K))
    print(f"Scale and zero: {ks} {kz}")

    QQ = (Qz-qz)*qs
    KK = (Kz-kz)*ks
    print(numpy.max(numpy.fabs(Q-QQ)))
    print(numpy.max(numpy.fabs(K-KK)))
    
    import pdb; pdb.set_trace()
    RQ = numpy.matmul((Qz-qz)*qs,(Kz-kz)*ks)

    print(numpy.max(numpy.fabs(R-RQ)))
    
