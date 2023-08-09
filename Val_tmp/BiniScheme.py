from functools import cached_property
from util import *

# =====================================================================
#
#  Globals
#
NaN = 9999999  # not a number
Debug = False


# =====================================================================
#
#  BiniScheme class
#
#  Axel.Kemper 'at' gmail.com  20-Dec-2022
#
class BiniScheme(object):
    """ This class implements operations around Bini forms to
        represent a matrix multiplication algorithm """

    def __init__(self, transpose_matrix_c: bool = False):
        """ To keep the number of member variables low,
            functools @cached_property member functions are used instead
            see: https://docs.python.org/3/library/functools.html """
        self.alpha = None
        """ 3D array  [row][col][product] """

        self.beta = None
        """ 3D array  [row][col][product] """

        self.gamma = None  # 3D array
        """ 3D array  [row][col][product] """

        self.transpose_matrix_c: bool = transpose_matrix_c
        """ Iff True, matrix Gamma is expected in transposed form """

        self.mod2_mode: bool = True
        """ Automatically set to False, if negative literals are found """

        self.output_file = None
        """  text output file cf. write() """

        self.output_line_count: int = 0
        """  counter for text lines output to file via write() """

    @staticmethod
    def array_cols(arr: list):
        return len(arr[0])

    @staticmethod
    def array_rows(arr: list):
        return len(arr)

    @cached_property
    def a_cols(self):
        return self.array_cols(self.alpha)

    @cached_property
    def a_rows(self):
        return self.array_rows(self.alpha)

    def auto_file_name(self):
        """ Return default file name for this Bini scheme"""
        return f"s{self.signature}.bini.txt"

    @cached_property
    def b_cols(self):
        return self.array_cols(self.beta)

    @cached_property
    def b_rows(self):
        return self.array_rows(self.beta)

    @cached_property
    def c_cols(self):
        return self.array_cols(self.gamma)

    @cached_property
    def c_rows(self):
        return self.array_rows(self.gamma)

    def combine(self, bsx: 'BiniScheme', bsy: 'BiniScheme'):
        """ make BiniScheme a combination of two other schemes """
        o()
        if bsx.signature == bsy.signature:
            o(f"Combining {bsx.signature} with itself to one Bini scheme")
        else:
            o(f"Combining {bsx.signature} and {bsy.signature} to one Bini scheme")
        check(bsx.transpose_matrix_c == bsy.transpose_matrix_c, "Incompatible transpose mode!")
        check(self.transpose_matrix_c == bsy.transpose_matrix_c, "Incompatible transpose mode!")
        check(bsx.mod2_mode == bsy.mod2_mode, "Incompatible mod 2 mode!")
        a_rows = bsx.a_rows * bsy.a_rows
        a_cols = bsx.a_cols * bsy.a_cols
        b_cols = bsx.b_cols * bsy.b_cols
        no_of_products = bsx.no_of_products * bsy.no_of_products
        self.create_matrices(a_rows, a_cols, b_cols, no_of_products)
        self.set_combined_matrix(self.alpha, bsx.alpha, bsx.r_products, bsy.alpha, bsy.r_products)
        self.set_combined_matrix(self.beta, bsx.beta, bsx.r_products, bsy.beta, bsy.r_products)
        self.set_combined_matrix(self.gamma, bsx.gamma, bsx.r_products, bsy.gamma, bsy.r_products)
        o(f"Resulting {self.signature} Bini scheme created")

    def create_matrices(self, a_rows, a_cols, b_cols, no_of_products):
        """ Create set of three Bini matrices as 3D arrays """
        check(not self.alpha, "Multiple Bini lines?")
        self.alpha = self.literal_array(a_rows, a_cols, no_of_products)
        self.beta = self.literal_array(a_cols, b_cols, no_of_products)
        self.gamma = self.literal_array(a_rows, b_cols, no_of_products)

    @staticmethod
    def determine_matrix_order(line: str, matrix_order: list):
        """Extract the order of the matrices from left to right"""
        check(not matrix_order, "Multiple 'product' lines?")

        #  trick to remove multiple blanks resulting in empty parts
        p = " ".join(line.split(" ")).split()
        check(len(p) == 4, f"Inconsistent line '{line}'")

        order = [find(["Alpha", "Beta", "Gamma"], sp) for sp in p[1:]]
        check(not (-1 in order), "Inconsistent Alpha, Beta, Gamma order")

        return order

    @staticmethod
    def extract_dimensions(line: str):
        """ Translate Bini line into matrix dimensions and number of products """
        p = line.split(" ")
        check(len(p) == 5, f"Inconsistent line '{line}'")
        a_rows = int(p[1])
        a_cols = int(p[2])
        b_cols = int(p[3])
        no_of_products = int(p[4])
        return a_rows, a_cols, b_cols, no_of_products

    def fill_matrices(self, line: str, matrix_order: list):
        """ Extract literals from line and insert them into matrices """
        p = line.split(";")
        check(len(p) == 4, f"Inconsistent product line. Fields: {len(p)}, expected: 4")
        product = int(p[0]) - 1
        check(0 <= product < self.no_of_products,
              f"Inconsistent product number {p[0]} outside [1 .. {self.no_of_products}]")
        matrices = [self.alpha, self.beta, self.gamma]

        #  loop through the line matrix-by-matrix
        for matrix_order_idx in range(len(matrices)):
            lit_idx = 0
            literals = p[1 + matrix_order_idx].strip().split()

            m_idx = matrix_order[matrix_order_idx]
            mat = matrices[m_idx]
            rows = self.r_array_rows(mat)
            cols = self.r_array_cols(mat)

            transpose = (m_idx == 2) and self.transpose_matrix_c

            for row in rows:
                for col in cols:
                    lit = int(literals[lit_idx])
                    if lit < 0:
                        # once we encounter a negative literal,
                        # it can no longer be a mod 2 algorithm
                        self.mod2_mode = False
                    cell = mat[col][row] if transpose else mat[row][col]
                    check(cell[product] == NaN, "Duplicate literal?")
                    cell[product] = lit
                    lit_idx += 1

    @staticmethod
    def literal_array(rows: int, cols: int, products: int):
        """ Create a 3D array rows x cols x products initiated to NaN """
        check((products > rows) or (products > cols), "Inconsistent products")
        # set all cells to NaN (not-a-number) to allow for error checking
        arr = [[[NaN for _ in range(products)] for _ in range(cols)] for _ in range(rows)]
        return arr

    @cached_property
    def no_of_products(self):
        """ Return number of products """
        return len(self.alpha[0][0])

    def r_array_cols(self, arr):
        return range(self.array_cols(arr))

    def r_array_rows(self, arr):
        return range(self.array_rows(arr))

    @cached_property
    def r_a_cols(self):
        return range(self.a_cols)

    @cached_property
    def r_a_rows(self):
        return range(self.a_rows)

    @cached_property
    def r_b_cols(self):
        return range(self.b_cols)

    @cached_property
    def r_b_rows(self):
        return range(self.b_rows)

    @cached_property
    def r_c_cols(self):
        return range(self.c_cols)

    @cached_property
    def r_c_rows(self):
        return range(self.c_rows)

    @cached_property
    def r_products(self):
        """ Return range of product numbers """
        return range(self.no_of_products)
    def read_ndarray(self,
                     alpha : numpy.ndarray, 
                     beta  : numpy.ndarray, 
                     gamma : numpy.ndarray, 
    ):

        
        
                     
        self.no_of_products = gamma.shape[2]
        self.create_ranges(alpha.shape[0], alpha.shape[1], beta.shape[1])
        self.create_matrices()

        matrix_rows = [self.r_a_rows, self.r_a_cols, self.r_a_rows]
        matrix_cols = [self.r_a_cols, self.r_b_cols, self.r_b_cols]
        matrices    = [self.alpha, self.beta, self.gamma]

        
        L = [alpha, beta,gamma] 
        
        #  loop through the line matrix-by-matrix
        for m_idx in range(len(matrices)):
                        
            mat = matrices[m_idx]
            rows = matrix_rows[m_idx]
            cols = matrix_cols[m_idx]
            transpose = (m_idx == 2) and self.transpose_matrix_c
                        
            for row in rows:
                for col in cols:
                    for p in range(L[m_idx].shape[2]):
                        
                        lit = L[m_idx][row][col][p]
                        if lit < 0:
                            # once we encounter a negative literal,
                            # it can no longer be a mod 2 algorithm
                            self.mod2_mode = False
                        cell = mat[col][row] if transpose else mat[row][col]
                        check(cell[p] == NaN, "Duplicate literal?")
                        cell[p] = lit

    def read(self, input_file_name: str):
        """ Read Bini form matrix multiplication algorithm into internal matrices """
        matrix_order = None  # list determines which matrix comes first in the Bini file
        no_of_lines = 0

        o(f"Reading input file '{input_file_name}'")

        s = "non-" if not self.transpose_matrix_c else ""
        o(f"Matrix Gamma is assumed to be in {s}transposed order")

        with open(input_file_name) as input_file:
            for line in input_file:
                line = line.replace("\n", "").strip()
                no_of_lines += 1
                if (len(line) == 0) or line.startswith("#"):
                    #  ignore empty lines and comments
                    pass
                elif line.startswith("Bini "):
                    a_rows, a_cols, b_cols, no_of_products = self.extract_dimensions(line)
                    self.create_matrices(a_rows, a_cols, b_cols, no_of_products)
                elif line.startswith("product"):
                    matrix_order = self.determine_matrix_order(line, matrix_order)
                else:
                    #  line must be a line with product number and matrix literals
                    self.fill_matrices(line, matrix_order)

        o(f"Lines read: {pretty_num(no_of_lines)}")

    def set_combined_matrix(self, m: list, x: list, x_products: range, y: list, y_products: range):
        """" Fill matrix m of literals by combining matrices x and y from smaller BiniSchemes """
        r_x_rows = self.r_array_rows(x)
        r_x_cols = self.r_array_cols(x)
        y_no_of_rows = self.array_rows(y)
        y_no_of_cols = self.array_cols(y)
        r_y_rows = self.r_array_rows(y)
        r_y_cols = self.r_array_cols(y)
        y_no_of_products = len(y_products)

        for x_row in r_x_rows:
            for x_col in r_x_cols:
                for x_product in x_products:
                    for y_row in r_y_rows:
                        for y_col in r_y_cols:
                            row = x_row * y_no_of_rows + y_row
                            col = x_col * y_no_of_cols + y_col
                            for y_product in y_products:
                                product = x_product * y_no_of_products + y_product
                                lit = x[x_row][x_col][x_product] * y[y_row][y_col][y_product]
                                if lit < 0:
                                    self.mod2_mode = False
                                m[row][col][product] = lit

    @property
    def signature(self):
        return f"{self.a_rows}x{self.a_cols}x{self.b_cols}_{self.no_of_products}"

    def validate(self):
        """ Check if Brent's equations are properly fulfilled. Return True iff OK, False otherwise """
        errors = 0
        equations = 0
        o()

        mode = " mod 2" if self.mod2_mode else ""
        o(f"Validating Brent's equations{mode} for {self.signature}")
        no_of_equations = (self.a_rows * self.a_cols * self.b_cols) ** 2
        #  show progress every 1% of the validation time
        delta_percent = 0.01 * no_of_equations
        next_show = delta_percent

        #  To speed up the following deeply nested loop,
        #  1D intermediate arrays are used to reduce costly 3D access
        for a_row in self.r_a_rows:
            for a_col in self.r_a_cols:
                arr_ak = [self.alpha[a_row][a_col][k] for k in self.r_products]
                for b_row in self.r_b_rows:
                    for b_col in self.r_b_cols:
                        arr_bk = [self.beta[b_row][b_col][k] for k in self.r_products]
                        for c_row in self.r_c_rows:
                            for c_col in self.r_c_cols:
                                eq_sum = 0
                                arr_c = self.gamma[c_row][c_col]
                                for product in self.r_products:
                                    # a = self.alpha[a_row][a_col][product]
                                    a = arr_ak[product]
                                    # b = self.beta[b_row][b_col][product]
                                    b = arr_bk[product]
                                    # c = self.gamma[c_row][c_col][product]
                                    c = arr_c[product]
                                    if Debug:
                                        check(a != NaN, "Inconsistent alpha cell")
                                        check(b != NaN, "Inconsistent beta cell")
                                        check(c != NaN, "Inconsistent gamma cell")
                                    eq_sum += a * b * c
                                #  odd is True iff the product of the three Kronecker deltas is True
                                odd = (a_row == c_row) and (b_row == a_col) and (b_col == c_col)
                                if self.mod2_mode:
                                    eq_sum = eq_sum % 2
                                if odd != (eq_sum == 1):
                                    errors += 1
                                    if errors == 1:
                                        o()
                                        o("Bummer! Error(s) found.")
                                        o()

                                equations += 1
                                if equations > next_show:
                                    print(".", end="")
                                    next_show += delta_percent
        o()
        o(f"Equations: {pretty_num(equations)}")
        check(equations == no_of_equations, "Inconsistent number of equations")
        if errors == 0:
            o(f"{self.signature} algorithm is OK! No errors found!")
        else:
            o(f"*** {self.signature} algorithm is not OK! *** Errors found: {pretty_num(errors)}")
        return errors == 0

    def w(self, s: str = ""):
        """ Write line of text to output file """
        self.output_file.write(s)
        self.output_file.write('\n')
        self.output_line_count += 1

    def write_header(self, output_file_name: str):
        """ Write header section to output file"""
        self.w("#")
        self.w(f"# File '{output_file_name}'")
        self.w("#")
        self.w(f"# Matrix multiplication algorithm {self.signature} in Bini format")
        self.w("#")
        self.w(f"# Created:  {date_stamp()}")
        self.w("#")
        self.w(f"Bini {self.a_rows} {self.a_cols} {self.b_cols} {self.no_of_products}")
        self.w()
        g = "Gamma".ljust(self.a_rows * self.b_cols * 3)
        a = "Alpha".ljust(self.a_rows * self.a_cols * 3)
        b = "Beta".ljust(self.a_cols * self.b_cols * 3)
        self.w(f"product {g}  {a}  {b}")

    def write_products(self):
        product_width = len(str(self.no_of_products))
        for product in self.r_products:
            s = " " + str(product + 1).rjust(product_width)
            for mat in [self.gamma, self.alpha, self.beta]:
                s += " ;"
                for row in self.r_array_rows(mat):
                    for col in self.r_array_cols(mat):
                        s += str(mat[row][col][product]).rjust(3)
            self.w(s)

    def write(self, output_file_name: str = ""):
        """ Store current Bini scheme as file. Use default file name, unless name is present """
        if not output_file_name:
            output_file_name = self.auto_file_name()
        o()
        o(f"Writing Bini scheme to file '{output_file_name}'")
        with open(output_file_name, 'w') as output_file:
            self.output_file = output_file
            self.output_line_count = 0
            self.write_header(output_file_name)
            self.write_products()
            self.w()
            self.w('#')
            self.w()
        o(f"File written. Lines: {self.output_line_count}")
        o()

#  end of file BiniScheme.py
