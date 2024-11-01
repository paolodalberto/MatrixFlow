# Unified Norm and Tiling for AIE

We present a unification and abstraction of the norm computations. The
goal is to highlight the common read and write patterns for the
presentation of blocked computation.  

The abstraction distills the algorithms and brings forth their
structure, data memory patterns, and complexity. In fact, the
algorithms are uniquely represented by two operations that we call
_projection_ and _normalization_. We aim at creating a
tiling/traversal for a 4x4 AIE structure but the computation and
validation is pure Python. The tiling is optimal, it exploits double
buffering and the correctness of the tiling and algorithm can be
validated on the spot.


This repository is composed of two main parts: the python code and a
document that represents the mathematical description.

## Code

Here the code is quite simple.

``` bash
matrix.py
splitting.py
tnorm.py
enorm.py
```


