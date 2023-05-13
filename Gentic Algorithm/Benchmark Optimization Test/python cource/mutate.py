import numpy as np
import copy

def PM(individuals, lower, upper, pm, n): # polynomial mutation
    x = copy.deepcopy(individuals)
    popsize = np.shape(individuals)[0]
    dim = np.shape(individuals[0])[0]
    mutate_pool = np.full((popsize, dim), np.nan)
    pow = 1 / (n + 1)

    for i in range(popsize):
        for j in range(dim):
            m_probability = np.random.uniform(0, 1)
            if m_probability <= pm:
                u = np.random.uniform(0, 1)
                if u < 0.5:
                    xi = np.power(2 * u, pow) - 1
                    x[i][j] = min(max(x[i][j] + xi * (x[i][j] - lower), lower), upper)
                else:
                    xi = 1 - np.power(2 * (1 - u), pow)
                    x[i][j] = min(max(x[i][j] + xi * (upper - x[i][j]), lower), upper)

    mutate_pool[0] = x[0]
    mutate_pool[1] = x[1]

    return mutate_pool