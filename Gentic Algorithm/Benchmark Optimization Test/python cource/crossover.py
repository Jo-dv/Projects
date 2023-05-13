import numpy as np
import copy

def SBX(individuals, lower, upper, pc, n):  # simulated binary crossover
    parents = copy.deepcopy(individuals)
    popsize = np.shape(parents)[0]
    dim = np.shape(parents[0])[0]
    crossover_pool = np.full((popsize, dim), np.nan)
    pow = 1 / (n + 1)

    c_probability = np.random.uniform(0, 1)
    if c_probability <= pc:
        for i, (p1, p2) in enumerate(zip(parents[0], parents[1])):
            u = np.random.uniform()
            if u <= 0.5:
                beta = np.power(2 * u, pow)
            else:
                beta = np.power(1 / (2 * (1 - u)), pow)

            crossover_pool[0][i] = min(max(0.5 * ((p1 + p2) + beta * (p1 - p2)), lower), upper)
            crossover_pool[1][i] = min(max(0.5 * ((p1 + p2) + beta * (p2 - p1)), lower), upper)
    else:
        crossover_pool[0] = parents[0]
        crossover_pool[1] = parents[1]

    return crossover_pool
