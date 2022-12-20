from Validation.util import *
import numpy 
# =====================================================================
#
#  Globals
#
NaN = 99999  # not a number




# =====================================================================
#
#  BiniScheme class (if you use deepmind "transpose_matrix_c = True
#
class BiniScheme(object):
    """ This class implements operations around Bini form to
        represent a matrix multiplication algorithm """

    def __init__(self, transpose_matrix_c : bool = False):
        self.alpha = []  # 2D matrix
        self.beta = []  # 2D matrix
        self.gamma = []  # 2D matrix
        self.a_rows = 0  # number of alpha rows
        self.a_cols = 0  # number of alpha columns
        self.b_cols = 0  # number of beta columns
        self.r_a_rows = []  # range of row numbers
        self.r_a_cols = []  # range of col numbers
        self.r_b_cols = []  # range of col numbers
        self.no_of_products = 0
        self.r_products = []  # range of product numbers (0 .. )
        self.transpose_matrix_c = transpose_matrix_c
        self.mod2_mode = True
        self.output_file = None
        self.output_line_count = 0

    def auto_file_name(self):
        """ Return default file name for this Bini scheme"""
        return f"s{self.signature()}.bini.txt"

    def combine(self, bsx: 'BiniScheme', bsy: 'BiniScheme'):
        """ make BiniScheme a combination of two other schemes """
        check(bsx.transpose_matrix_c == bsy.transpose_matrix_c, "Incompatible transpose mode!")
        check(self.transpose_matrix_c == bsy.transpose_matrix_c, "Incompatible transpose mode!")
        a_rows = len(bsx.r_a_rows) * len(bsy.r_a_rows)
        a_cols = len(bsx.r_a_cols) * len(bsy.r_a_cols)
        b_cols = len(bsx.r_b_cols) * len(bsy.r_b_cols)
        self.no_of_products = bsx.no_of_products * bsy.no_of_products
        self.create_ranges(a_rows, a_cols, b_cols)
        self.create_matrices()
        self.set_combined_matrix(self.alpha, bsx.alpha, bsx.r_products, bsy.alpha, bsy.r_products)
        self.set_combined_matrix(self.beta, bsx.beta, bsx.r_products, bsy.beta, bsy.r_products)
        self.set_combined_matrix(self.gamma, bsx.gamma, bsx.r_products, bsy.gamma, bsy.r_products)

    def create_matrices(self):
        """ Create set of three Bini matrices as 3D arrays """
        check(not self.alpha, "Multiple Bini lines?")
        self.alpha = [[[NaN for _ in self.r_products] for _ in self.r_a_cols] for _ in self.r_a_rows]
        self.beta = [[[NaN for _ in self.r_products] for _ in self.r_b_cols] for _ in self.r_a_cols]
        self.gamma = [[[NaN for _ in self.r_products] for _ in self.r_b_cols] for _ in self.r_a_rows]

    def create_ranges(self, a_rows: int, a_cols: int, b_cols: int):
        """ Ranges a practical for loops """
        self.a_rows = a_rows
        self.a_cols = a_cols
        self.b_cols = b_cols
        self.r_a_rows = range(a_rows)
        self.r_a_cols = range(a_cols)
        self.r_b_cols = range(b_cols)
        check(1 <= self.no_of_products <= a_rows * a_cols * b_cols,
              f"Inconsistent number of products '{self.no_of_products}'")
        self.r_products = range(self.no_of_products)

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

    def extract_dimensions(self, line: str):
        """ Translate Bini line into matrix dimensions and number of products """
        p = line.split(" ")
        check(len(p) == 5, f"Inconsistent line '{line}'")
        a_rows = int(p[1])
        a_cols = int(p[2])
        b_cols = int(p[3])
        self.no_of_products = int(p[4])
        self.create_ranges(a_rows, a_cols, b_cols)

    def fill_matrices(self, line: str, matrix_order: list):
        """Extract literals from line and insert them into matrices"""
        p = line.split(";")
        check(len(p) == 4, f"Inconsistent product line. Fields: {len(p)}, expected: 4")
        product = int(p[0]) - 1
        check(0 <= product < self.no_of_products,
              f"Inconsistent product number {p[0]} outside [1 .. {self.no_of_products}]")
        matrix_rows = [self.r_a_rows, self.r_a_cols, self.r_a_rows]
        matrix_cols = [self.r_a_cols, self.r_b_cols, self.r_b_cols]
        matrices = [self.alpha, self.beta, self.gamma]

        #import pdb; pdb.set_trace()
        
        #  loop through the line matrix-by-matrix
        for matrix_order_idx in range(len(matrices)):
            lit_idx = 0
            literals = p[1 + matrix_order_idx].strip().split()

            m_idx = matrix_order[matrix_order_idx]
            mat = matrices[m_idx]
            rows = matrix_rows[m_idx]
            cols = matrix_cols[m_idx]
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
        matrix_order = []  # determines which matrix comes first in the Bini file
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
                    self.extract_dimensions(line)
                    self.create_matrices()
                elif line.startswith("product"):
                    matrix_order = self.determine_matrix_order(line, matrix_order)
                else:
                    #  line must be a line with product number and matrix literals
                    self.fill_matrices(line, matrix_order)

        o(f"Lines read: {pretty_num(no_of_lines)}")

    def set_combined_matrix(self, m: list, x: list, x_products: list, y: list, y_products: list):
        """" Fill matrix m of literals by combining matrices x and y from smaller BiniSchemes """
        x_rows = range(len(x))
        x_cols = range(len(x[0]))
        y_no_of_rows = len(y)
        y_no_of_cols = len(y[0])
        y_rows = range(y_no_of_rows)
        y_cols = range(y_no_of_cols)
        y_no_of_products = len(y_products)

        for x_row in x_rows:
            for x_col in x_cols:
                for x_product in x_products:
                    for y_row in y_rows:
                        for y_col in y_cols:
                            row = x_row * y_no_of_rows + y_row
                            col = x_col * y_no_of_cols + y_col
                            for y_product in y_products:
                                product = x_product * y_no_of_products + y_product
                                lit = x[x_row][x_col][x_product] * y[y_row][y_col][y_product]
                                if lit < 0:
                                    self.mod2_mode = False
                                m[row][col][product] = lit

    def signature(self):
        return f"{len(self.r_a_rows)}x{len(self.r_a_cols)}x{len(self.r_b_cols)}_{self.no_of_products}"

    def validate(self):
        """ Check if Brent's equations are properly fulfilled. Return True iff OK, Fase otherwise """
        errors = 0
        equations = 0
        o()

        mode = " mod 2" if self.mod2_mode else ""
        o(f"Validating Brent's equations{mode} for {self.signature()}")
        no_of_eqations = (len(self.r_a_rows) * len(self.r_a_cols) * len(self.r_b_cols)) ** 2
        #  show progress every 1% of the validation time
        delta_percent = 0.01 * no_of_eqations
        next_show = delta_percent

        for ra in self.r_a_rows:
            for ca in self.r_a_cols:
                for rb in self.r_a_cols:
                    for cb in self.r_b_cols:
                        for rc in self.r_a_rows:
                            for cc in self.r_b_cols:
                                eq_sum = 0
                                for k in self.r_products:
                                    a = self.alpha[ra][ca][k]
                                    b = self.beta[rb][cb][k]
                                    c = self.gamma[rc][cc][k]
                                    check(a != NaN, "Inconsistent alpha cell")
                                    check(b != NaN, "Inconsistent beta cell")
                                    check(c != NaN, "Inconsistent gamma cell")
                                    eq_sum += a * b * c
                                odd = (ra == rc) and (rb == ca) and (cb == cc)
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
        check(equations == no_of_eqations, "Inconsistent number of equations")
        if errors == 0:
            o(f"{self.signature()} algorithm is OK! No errors found!")
        else:
            o(f"*** {self.signature()} algorithm is not OK! *** Errors found: {pretty_num(errors)}")
        return errors == 0


    def validate_ndarray(
            a :numpy.ndarray,
            b :numpy.ndarray,
            c :numpy.ndarray,

    ):
        
        
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
        self.w(f"# Matrix multiplication algorithm {self.signature()} in Bini format")
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
            s += " ;"
            for row in self.r_a_rows:
                for col in self.r_b_cols:
                    s += str(self.gamma[row][col][product]).rjust(3)
            s += " ;"
            for row in self.r_a_rows:
                for col in self.r_a_cols:
                    s += str(self.alpha[row][col][product]).rjust(3)
            s += " ;"
            for row in self.r_a_cols:
                for col in self.r_b_cols:
                    s += str(self.beta[row][col][product]).rjust(3)
            self.w(s)

    def write(self, output_file_name: str = ""):
        """ Store current Bini schema as file. Use default file name, unless name is present """
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
