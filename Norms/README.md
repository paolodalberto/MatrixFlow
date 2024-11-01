# Unified Norm and Tiling for AIE

We present a unification and abstraction of the matrix norm
computations. The goal is to highlight the common read and write
patterns for the presentation of blocked computation.

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

### Matrices 

We work on Matrices and Vectors, _matrix.py_ contains the class
definitions and the introduction of Tiling. Tiling is a **partition**
of a Matrix using a traversal description, which is used for MLADF AIE
tiling. In short, Tiling is a composition of two ideas: Spatial and
Temporal.

#### Spatial and Temporal partition

We take a matrix and we split it into spatially separated spaces. For
example, into for 4. Each part is a _buffer_ and it has to be
transferred into a smaller memory in _tiles_ and these tiles are
stream into a channel (we assume there is an order to read and
write). The streaming infer a temporal transfer of tiles. In general,
the tiles are logically bound into rectangles, making our algorithm
being part of family of blocked algorithms. In practice, blocked
algorithms using row major layout are exploit "strided" patterns.

Tiling will represent the spatial and temporal move of data from a
Level A to a level A-1.


### Splitting matrices

There are many ways to split a matrix and split the correspondent
computation. We logically split a matrix by row, by columns and by a
sequence of both. If the buffer (level A) is given, we may wonder the
temporal partition in tiles that uses the best some specific space
constraints. For example, the tiles size has no more that 1024
elements.

### Norms

There are two basic norm class **Norm** and **LayerNorm**. Both will
do a row normalization, although the normalization function is
different. However, LayerNorm presents the case where a column
normalization by a set of parameters is possible and applicable.

We show that SoftMax is a simple extension of Norm. We show that
RMSNorm and Instance norm is an extension of LayerNorm.


### Testing

The testing follows a simple format. We have a numpy implementation,
which is the golden standard. We have a simplified two-pass
implementation using basic matrices. Then we have a multi staged
tiling computations.


cheers
Paolo



