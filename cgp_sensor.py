from sympy.parsing.sympy_parser import *
from sympy import init_printing
import pygmo as pg
import dcgpy
from matplotlib import pyplot as plt
import numpy as np
import argparse
import dataset_loader_2020 as dl
import json
import os
import pickle
import warnings
warnings.filterwarnings("ignore")
# Sympy is nice to have for basic symbolic manipulation.
init_printing()
# Fundamental for plotting.


def main():
    parser = argparse.ArgumentParser(description='CGP')
    parser.add_argument('--mode', '-m', default="main",
                        help='Main or Aux')
    parser.add_argument('--gen', '-g', type=int, default=10000,
                        help='generation')
    parser.add_argument('--max-mut', type=int, default=10000,
                        help='max mutation')
    parser.add_argument('--fold', type=int, default=0, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()

    print("Mode: {}".format(args.mode))

    with open("sensor_y{}.json".format(args.fold)) as f:
        sensors = json.load(f)
    y_sensor = sensors["Y"][:1]

    train_x = np.load(
        "E:/fp16/MainData/main{0:010d}.npy".format(1)).transpose(0, 2, 1)[:30]
    train_y = np.load(
        "E:/fp16/AuxData/aux{0:010d}.npy".format(1))[:, y_sensor, :].transpose(0, 2, 1)[:30]
    test_x = np.load(
        "E:/fp16/MainData/main{0:010d}.npy".format(2)).transpose(0, 2, 1)[:30]
    test_y = np.load(
        "E:/fp16/AuxData/aux{0:010d}.npy".format(2))[:, y_sensor, :].transpose(0, 2, 1)[:30]

    train_x = train_x.reshape(-1, train_x.shape[2])
    train_y = train_y.reshape(-1, train_y.shape[2])
    test_x = test_x.reshape(-1, test_x.shape[2])
    test_y = test_y.reshape(-1, test_y.shape[2])

    print(train_x.shape, train_y.shape)
    print(test_x.shape, test_y.shape)

    ss = dcgpy.kernel_set_double(["sum", "mul", "sin", "cos", "gaussian"])
    udp = dcgpy.symbolic_regression(
        points=train_x, labels=train_y, kernels=ss(),
        rows=1,
        cols=100,
        n_eph=1,
        levels_back=80)
    prob = pg.problem(udp)
    print(udp)

    uda = dcgpy.es4cgp(gen=args.gen, max_mut=args.max_mut)
    algo = pg.algorithm(uda)
    algo.set_verbosity(500)

    pop = pg.population(prob, 4)
    pop = algo.evolve(pop)
    print("Best model loss:", pop.champion_f[0])

    x = pop.champion_x
    print(type(x))
    a = parse_expr(udp.prettier(x))[0]
    # with open("champ.txt", "w") as f:
    #     pickle.dump(a, f)
    a = a.subs({"c1": x[0]})
    print(type(a))
    print(a)

    log = algo.extract(dcgpy.es4cgp).get_log()
    gen = [it[0] for it in log]
    loss = [it[2] for it in log]

    plt.semilogy(gen, loss)
    plt.title('last call to evolve')
    plt.xlabel('generation')
    plt.ylabel('loss')
    plt.show()
    plt.savefig("test.png")
    plt.close()


if __name__ == '__main__':
    main()
