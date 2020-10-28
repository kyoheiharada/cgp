# Some necessary imports.
import argparse
import numpy as np
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


def my_max(x):
    return max(x)


def my_max_print(x):
    s = ','
    return "max(" + s.join(x) + ")"


def my_min(x):
    return min(x)


def my_min_print(x):
    s = ','
    return "min(" + s.join(x) + ")"


class MGG:
    def __init__(self, gen, udp, nx, ny, rows, cols,
                 kernels, n_eph=1, cross_times=32,
                 mut_ratio=.01, max_eval=None):
        # super(MGG, self).__init__(gen=gen, m=mut_ratio)
        self.__gen = gen
        self.udp = udp
        self.nx = nx
        self.ny = ny
        self.rows = rows
        self.cols = cols
        self.kernels = kernels
        self.n_eph = n_eph
        self.__cross_times = cross_times
        self.mr = mut_ratio
        # self.selection = pg.algorithm._sga_selection_type.ROULETTE
        self.max_eval = max_eval
        self.verb = 1
        self.log = []
        self.feval = 0

    def cross(self, ind1, ind2):
        import random
        p = random.randint(1, len(ind1) - 1)

        indA = np.concatenate([ind1[:p], ind2[p:]])
        indB = np.concatenate([ind2[:p], ind1[p:]])

        # ind1[:p], ind2[p:] = ind2[:p], ind1[p:]
        return tuple((indA, indB, ))

    def mutate(self, pop):
        import random
        for i in range(len(pop)):
            if self.mr > random.random():
                new_x = dcgpy.expression_double(
                    self.nx, self.ny, self.rows, self.cols, self.cols + 1,
                    kernels=self.kernels, n_eph=1)
                new_x.set(list(map(lambda x: int(x), pop.get_x()[i][1:])))
                new_x.mutate_active()
                new_arr = new_x.get()
                new_arr.insert(0, pop.get_x()[i][:self.n_eph])
                pop.set_x(i, new_arr)

        return pop

    def selection(self, pop):
        # ルーレット選択
        import random
        fits = pop.get_f()
        xs = pop.get_x()
        sum_f = sum(list([1. / i for i in fits]))
        p = random.uniform(0, sum_f)
        now = 0
        for i in range(len(fits)):
            now += 1. / fits[i]
            if now >= p:
                break

        return xs[i], fits[i]

    def evolve(self, pop):
        print("Gen:\tBest:\tModel:")
        prob = pop.problem
        self.feval += len(pop.get_x())
        import random
        for g in range(1, self.__gen + 1):
            pop_cand = pg.population(prob)
            idx1 = random.randint(0, len(pop) - 1)
            idx2 = (idx1 + random.randint(0, len(pop) - 2)) % len(pop)
            ind1 = pop.get_x()[idx1]
            ind2 = pop.get_x()[idx2]
            pop_cand.push_back(ind1)
            pop_cand.push_back(ind2)
            for i in range(self.__cross_times):
                new_xs = self.cross(ind1, ind2)
                pop_cand.push_back(new_xs[0])
                pop_cand.push_back(new_xs[1])

            pop_cand = self.mutate(pop_cand)
            new_xs = pop_cand.get_x()
            for i, new_x in enumerate(new_xs):
                new_f = prob.fitness(new_x)
                pop_cand.set_xf(i, new_x, new_f)

            # Create Next-Gen Population
            # 1st: Elite
            pop.set_xf(idx1, pop_cand.champion_x, pop_cand.champion_f)
            # 2nd: Roulette Selection
            n_x, n_f = self.selection(pop_cand)
            pop.set_xf(idx2, n_x, n_f)

            if g % self.verb == 0:
                print("{}\t{}\t{}".format(
                    g, pop.champion_f, self.udp.prettier(pop.champion_x)))
            self.feval += len(pop.get_x())
            self.log.append((g, self.feval, float(pop.champion_f), pop.get_x()[
                            :self.n_eph], self.udp.prettier(pop.champion_x)))

        return pop

    def set_verbosity(self, l):
        self.verb = l

    def get_log(self):
        return self.log

    def __get_name(self):
        return "Minimal Generation Gap"


def main():
    parser = argparse.ArgumentParser(description='CGP by MGG')
    parser.add_argument('--mode', '-m', default="main",
                        help='Main or Aux')
    parser.add_argument('--gen', '-g', type=int, default=1000,
                        help='generation')
    parser.add_argument('--pop', '-p', type=int, default=100,
                        help='population')
    parser.add_argument('--rows', '-r', type=int, default=1,
                        help='rows')
    parser.add_argument('--cols', '-c', type=int, default=16,
                        help='cols')
    parser.add_argument('--fold', type=int, default=0, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()

    X, Y = dcgpy.generate_chwirut2()

    ss = dcgpy.kernel_set_double(["sum", "diff", "mul", "pdiv"])
    udp = dcgpy.symbolic_regression(
        points=X, labels=Y, rows=1, cols=16, kernels=ss())
    print(udp)

    uda = MGG(gen=args.gen, udp=udp, nx=X.shape[-1],
              ny=Y.shape[-1], rows=args.rows, cols=args.cols, kernels=ss())
    prob = pg.problem(udp)
    print(prob)
    algo = pg.algorithm(uda)
    # algo.set_verbosity(1000)

    pop = pg.population(prob, args.pop)
    pop = algo.evolve(pop)
    print("Best model loss:", pop.champion_f[0])

    x = pop.champion_x

    print(parse_expr(udp.prettier(pop.champion_x)))


if __name__ == '__main__':
    main()
