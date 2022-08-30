import numpy as np
import pandas as pd
import problems
from run import algorithms


def __main__():
    problems
    rosenbrock = np.array(pd.read_csv('problems/rosenbrock_problem.csv', header=None))
    rastrigin = np.array(pd.read_csv('problems/rastrigin_problem.csv', header=None))
    griewank = np.array(pd.read_csv('problems/griewank_problem.csv', header=None))
    shc = np.array(pd.read_csv('problems/shc_problem.csv', header=None))

    test = algorithms(1, 1000, ('rastrigin', rastrigin), 'GA')
    BSF, BCP, ACP, WCP, MBF, avg_fits = test.evolve(pc=1.0, n_c=10, pm=0.005, n_m=10)
    print(MBF)


__main__()