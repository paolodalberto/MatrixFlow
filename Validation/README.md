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
