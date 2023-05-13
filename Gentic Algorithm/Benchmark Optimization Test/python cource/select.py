import numpy as np

def sus_select(population, fit):
    N = 2
    dim = np.shape(population[0])[0]
    mating_pool = np.full((2, dim), np.nan)
    p = fit / np.sum(fit)

    cumfit = np.cumsum(p)
    cumfit[-1] = 1.0

    r = np.random.uniform(0, 1 / N)
    ptr = np.zeros(N)
    for i in range(N):
        ptr[i] = r + i * (1 / N)
    i = 0

    for marker, p in enumerate(ptr):
        while cumfit[i] < p:
            i += 1
        mating_pool[marker] = population[i]

    return mating_pool[0], mating_pool[1]


def bts_select(population, fit):
    k = 2
    popsize = np.shape(population)[0]
    dim = np.shape(population[0])[0]
    mating_pool = np.full((2, dim), np.nan)

    for i in range(2):
        indiv_idx = [np.random.choice(range(0, popsize)) for _ in range(k)]
        selected_indiv = population[indiv_idx]
        selected_fit = fit[indiv_idx]
        sorted_idx = np.argsort(-selected_fit)

        mating_pool[i] = selected_indiv[sorted_idx[0]]

    return mating_pool[0], mating_pool[1]