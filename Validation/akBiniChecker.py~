import sys
from BiniScheme import BiniScheme
from util import *

welcomeText = '''
[]---------------------<  akBiniChecker  >-----------------------[]
| Read and check correctness of a matrix multiplication algorithm |
| in Bini matrix form. Optionally combine schemes.                |
|                                                                 |
| Algorithms with purely positive coefficients are assumed to be  |
| modulo 2.                                                       |
|                                                                 |
| If two input files are specified, the checker will check both   |
| and combine them into a "multiplied" Bini scheme. The resulting |
| scheme is written to a text file s<p>x<q>x<n>_<k>.bini.txt      |
| It is allowed to combine a scheme with itself.                  |
|                                                                 |
| Axel.Kemper@gmail.com  18-Dec-2022                              |
[]---------------------------------------------------------------[]
'''

usageText = '''
Usage:

python akBiniChecker.py <input_file> [<input_file_2>]

'''

# ======================================================================
#
#  Global variables
#

#  change this, if the Gamma matrix is transposed
transpose_matrix_c = False


# ======================================================================
#
#  Functions in alphabetical order
#


def get_arguments(argv):
    """Extract commandline argument"""
    if len(argv) not in [2, 3]:
        o("No file parameter(s) in commandline")
        usage()
    arg1 = argv[1]
    if len(argv) == 2:
        if "~/?-?--help/h-h".find(arg1) != -1:
            usage()

    check(exists(arg1), f'File {arg1} does not exist')
    o(f"Input file:  {arg1}")

    if len(argv) == 3:
        arg2 = argv[2]
        o(f"2nd input:   {arg2}")
    else:
        arg2 = ""
    return arg1, arg2


def read_and_validate(input_file_name):
    bs = BiniScheme(transpose_matrix_c)
    bs.read(input_file_name)
    res = bs.validate()
    check(res, f"Algorithm {bs.signature()} found to be invalid!")
    return bs


def usage():
    """"Show usage text and quit"""
    o(usageText)
    finish(1)


if __name__ == '__main__':
    o(welcomeText)
    input_file1_name, input_file2_name = get_arguments(sys.argv)
    bs1 = read_and_validate(input_file1_name)

    if input_file2_name != "":
        bs2 = read_and_validate(input_file2_name)

        bsc = BiniScheme(transpose_matrix_c)
        bsc.combine(bs1, bs2)
        res = bsc.validate()
        check(res, "Combined Bini scheme found to be invalid!")
        bsc.write()

    finish(0)

#  end of file akBiniChecker.py
