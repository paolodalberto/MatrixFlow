Usage:

```sh python akBiniChecker.py <input_file> [<input_file_2>]```

This is a self-contained module for the validation of fast matrix multiplication 
algorithms in Bini's format

**Purpose:** Read and check correctness of a matrix multiplication algorithm
in Bini matrix form. Optionally combine two schemes.
 
Algorithms with purely positive coefficients are assumed to be modulo 2.

If two input files are specified, the checker will check both
and combine them into a "multiplied" Bini scheme. The resulting
scheme is written to a text file `s<p>x<q>x<n>_<k>.bini.txt`
It is allowed to combine a scheme with itself.

Axel.Kemper at gmail.com  18-Dec-2022


```sh
## play and create new algorithms and validate them 

python3 akBiniChecker.py MatMultSolutions20230104/s2x2x2_07.Strassen/s2x2x2_07.Strassen.Bini.txt MatMultSolutions20230104/s3x3x3_23.JinsooOh_20131111a/s3x3x3_23.JinsooOh_20131111a.Bini.txt 
python3 akBiniChecker.py MatMultSolutions20230104/s3x3x3_23.JinsooOh_20131111a/s3x3x3_23.JinsooOh_20131111a.Bini.txt MultSolutions20230104/s3x3x3_23.JinsooOh_20131111a/s3x3x3_23.JinsooOh_20131111a.Bini.txt 
python3 akBiniChecker.py MatMultSolutions20230104/s3x3x3_23.JinsooOh_20131111a/s3x3x3_23.JinsooOh_20131111a.Bini.txt MatMultSolutions20230104/s3x3x3_23.JinsooOh_20131111a/s3x3x3_23.JinsooOh_20131111a.Bini.txt 
python3 akBiniChecker.py MatMultSolutions20230104/s2x2x2_07.Strassen/s2x2x2_07.Strassen.Bini.txt MatMultSolutions20230104/s5x5x5_99.Sedoglavic/s5x5x5_99.Sedoglavic.Bini.txt 
python3 akBiniChecker.py MatMultSolutions20230104/s2x2x2_07.Strassen/s2x2x2_07.Strassen.Bini.txt s6x6x6_161.bini.txt
 ```
 
The factorization tools, the way of combining two different algorithms into a single set of three tensors, have important ramifications. 
If you consider a problem MxKxN, a single tensor solution creates a balanced computation: each product is well defined, the submatrices are well defined and the whole algorithm can be verified. The data dependecies between input submatrices and products are direct, and so the dependecy between products and output submatrices.   

For every tensor representation, there is an equivalent recursive algorithm. The recursion makes the data depedency tricker (we will no go into the details). But a recursive algorithm does not need the same depth from the root to the leaf computation, practical matrix products can be different, it is not clear if there is an equivalent tensor representation of the recursive algorithm, and there is no clear representation of how to apply matrix transposition to reduce the effect of error.     
