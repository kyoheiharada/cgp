# Some necessary imports.
import warnings
import dcgpy
import pygmo as pg
# Sympy is nice to have for basic symbolic manipulation.
from sympy import init_printing
from sympy.parsing.sympy_parser import *
init_printing()


warnings.filterwarnings("ignore")
# Sympy is nice to have for basic symbolic manipulation.
init_printing()
# Fundamental for plotting.


class MGG(pg.algorithm):
    def __init__(self, gen, crossover=None, cross_times=None, mut_ratio=None, max_eval=None):
        super(MGG, self).__init__()
        self.gen = gen
        self.crossover = crossover
        self.cross_times = cross_times
        self.mut_ratio = mut_ratio
        self.max_eval = max_eval

    def evolve(self, pop):
        n = len(pop.get_ID())
        print(n)
        return pop

    def set_verbosity(self, l):
        pass


def main():
    X, Y = dcgpy.generate_chwirut2()

    ss = dcgpy.kernel_set_double(["sum", "diff", "mul", "pdiv"])
    udp = dcgpy.symbolic_regression(points=X, labels=Y, kernels=ss())
    print(udp)
    uda = MGG(gen=2)
    prob = pg.problem(udp)
    algo = pg.algorithm(uda)
    # algo.set_verbosity(1000)

    pop = pg.population(prob, 4)
    pop = algo.evolve(pop)
    print("Best model loss:", pop.champion_f[0])

    x = pop.champion_x

    parse_expr(udp.prettier(pop.champion_x))


if __name__ == '__main__':
    main()
