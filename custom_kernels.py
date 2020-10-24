# Some necessary imports.
from matplotlib import pyplot as plt
import dcgpy
from time import time
import pyaudi
# Sympy is nice to have for basic symbolic manipulation.
from sympy import init_printing
from sympy.parsing.sympy_parser import *
init_printing()
# Fundamental for plotting.


def my_max(x):
    return max(x)


def my_max_print(x):
    s = ','
    return "max(" + s.join(x) + ")"


if __name__ == '__main__':
    a = my_max([0.2, -0.12, -0.0011])
    b = my_max_print(["x", "y", "z"])
    print(b + " is: " + str(a))

    my_max = dcgpy.kernel_double(my_max, my_max_print, "max")
    a = my_max([0.2, -0.12, -0.0011])
    b = my_max(["x", "y", "z"])
    print(b + " evaluates to: " + str(a))
    ks = dcgpy.kernel_set_double(["sum", "mul", "diff"])
    ks.push_back(my_max)
    print(ks)
