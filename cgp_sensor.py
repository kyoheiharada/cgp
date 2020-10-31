import sympy
from sympy.parsing.sympy_parser import *
from sympy import init_printing
from sympy.printing import *
import pygmo as pg
import dcgpy
from matplotlib import pyplot as plt
import numpy as np
import datetime
import argparse
import json
import os
import pickle
from MGG.extend_algorithm import MGG
import warnings
warnings.filterwarnings("ignore")
# Sympy is nice to have for basic symbolic manipulation.
init_printing()
# Fundamental for plotting.


def main():
    sympy.init_printing()
    parser = argparse.ArgumentParser(description='CGP')
    parser.add_argument('--mode', '-m', default="main",
                        help='Main or Aux')
    parser.add_argument('--gen', '-g', type=int, default=50000,
                        help='generation')
    parser.add_argument('--mutation', type=float, default=.05,
                        help='mutation ratio')
    parser.add_argument('--pop', '-p', type=int, default=100,
                        help='population')
    parser.add_argument('--rows', '-r', type=int, default=1,
                        help='rows')
    parser.add_argument('--cols', '-c', type=int, default=15,
                        help='cols')
    parser.add_argument('--levels-back', '-l', type=int, default=10,
                        help='levels-back')
    parser.add_argument('--cross-times', '-t', type=int, default=10,
                        help='cross times')
    parser.add_argument('--fold', type=int, default=0, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()

    print("Mode: {}".format(args.mode))
    date = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    os.makedirs("result/" + date)
    with open("result/" + date + "/args.json", "w") as f:
        json.dump(args.__dict__, f, indent=4)

    with open("sensor_y{}.json".format(args.fold)) as f:
        sensors = json.load(f)
    y_sensor = sensors["Y"][:2]

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

    ss = dcgpy.kernel_set_double(
        ["sum", "diff", "mul", "pdiv", "sin", "cos", "exp", "log", "gaussian"])
    udp = dcgpy.symbolic_regression(
        points=train_x, labels=train_y, rows=args.rows, cols=args.cols,
        kernels=ss(), levels_back=args.levels_back, n_eph=1)
    prob = pg.problem(udp)
    print(udp)
    print(prob)

    # uda = dcgpy.es4cgp(gen=args.gen, max_mut=args.max_mut)
    uda = MGG(gen=args.gen, udp=udp, nx=train_x.shape[-1],
              ny=train_y.shape[-1], rows=args.rows, cols=args.cols,
              levels_back=args.levels_back, kernels=ss(), n_eph=1,
              cross_times=args.cross_times, mut_ratio=args.mutation)

    algo = pg.algorithm(uda)
    algo.set_verbosity(100)

    pop = pg.population(prob, args.pop)
    pop = algo.evolve(pop)
    print("Best model loss:", pop.champion_f[0])

    x = pop.champion_x

    equ = []
    for a in parse_expr(udp.prettier(x)):
        a = a.subs({"c1": x[0]})
        print(a)
        equ.append(sympy.latex(a))

    Y_pred = udp.predict(train_x, pop.champion_x)
    for i in range(Y_pred.shape[-1]):
        plt.plot(train_y[:, i], label="true")
        plt.plot(Y_pred[:, i], label='predict')
        plt.title('$f(x) = ' + equ[i] + '$')
        plt.savefig("result/" + date +
                    "/train_result_{}.png".format(sensors["Label"][i]))
        plt.close()

    Y_pred = udp.predict(test_x, pop.champion_x)
    for i in range(Y_pred.shape[-1]):
        plt.plot(test_y[:, i], label="true")
        plt.plot(Y_pred[:, i], label='predict')
        plt.title('$f(x) = ' + equ[i] + '$')
        plt.savefig("result/" + date +
                    "/test_result_{}.png".format(sensors["Label"][i]))
        plt.close()

    # save population
    with open("result/" + date + "/pop", 'wb') as f:
        pickle.dump(pop, f)

    log = algo.extract(MGG).get_log()
    gen = [it[0] for it in log]
    loss = [it[2] for it in log]
    res = {}
    res["loss"] = loss
    res["latex"] = equ
    json.dump(res, open("result/" + date + "/result.json", "w"))

    plt.semilogy(gen, loss)
    plt.title('last call to evolve')
    plt.xlabel('generation')
    plt.ylabel('loss')
    plt.savefig("result/" + date + "/result.png")
    plt.close()


if __name__ == '__main__':
    main()
