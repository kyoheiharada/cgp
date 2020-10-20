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


# class MGG(pg.sga):
class MGG:
    def __init__(self, gen, cross_times=32, mut_ratio=.01, max_eval=None):
        # super(MGG, self).__init__(gen=gen, m=mut_ratio)
        self.__gen = gen
        self.__cross_times = cross_times
        self.mr = mut_ratio
        # self.selection = pg.algorithm._sga_selection_type.ROULETTE
        self.max_eval = max_eval
        self.verb = 1

    def cross(self, ind1, ind2):
        import random
        p = random.randint(1, len(ind1) - 1)

        ind1[:p], ind2[p:] = ind2[:p], ind1[p:]
        return tuple((ind1, ind2, ))

    def mutate(self, ind):
        import random
        if self.mr > random.random():
            pass
        return ind

    def selection(self, pop):
        return pop.get_x(0), pop.get_f(0)

    def evolve(self, pop):
        print("Gen:\tBest:\tConstants:\tModels:")
        if len(pop) == 0:
            return pop
        prob = pop.problem
        pop_cand = pg.population(prob)
        import random
        for g in range(self.__gen):
            idx1 = random.randint(0, len(pop) - 1)
            idx2 = (idx1 + random.randint(0, len(pop) - 2)) % len(pop)
            ind1 = pop.get_x()[idx1]
            ind2 = pop.get_x()[idx2]
            pop_cand.push_back(ind1)
            pop_cand.push_back(ind2)
            for i in range(self.__cross_times):
                new_x = self.mutate(self.cross(ind1, ind2))
                pop_cand.push_back(new_x[0])
                pop_cand.push_back(new_x[1])

            # Create Next-Gen Population
            # 1st: Elite
            pop.set_xf(idx1, pop_cand.champion_x, pop_cand.champion_f)
            # 2nd: Roulette Selection
            pop.set_xf(idx2, self.selection(pop_cand))

            if g % self.verb == 0:
                print(g + "\t" + pop.champion_f + "\tConstants:\tModels:")

        return pop

    def set_verbosity(self, l):
        self.verb = l

    def __get_name(self):
        return "Minimal Generation Gap"


def main():
    X, Y = dcgpy.generate_chwirut2()

    ss = dcgpy.kernel_set_double(["sum", "diff", "mul", "pdiv"])
    udp = dcgpy.symbolic_regression(
        points=X, labels=Y, kernels=ss())
    print(udp)
    uda = MGG(gen=5000)
    prob = pg.problem(udp)
    print(prob)
    algo = pg.algorithm(uda)
    algo.set_verbosity(1000)
    print(algo.extract(object))
    print(algo.get_name())

    pop = pg.population(prob, 100)
    pop = algo.evolve(pop)
    print("Best model loss:", pop.champion_f[0])

    x = pop.champion_x

    print(parse_expr(udp.prettier(pop.champion_x)))


if __name__ == '__main__':
    main()
