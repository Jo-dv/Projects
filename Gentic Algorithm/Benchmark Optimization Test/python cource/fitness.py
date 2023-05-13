import numpy as np

def fitness(population, func_op):
    dim = np.shape(population[0])[0]
    x = population
    if func_op == 0:  # rosenbrock
        x1 = x[:, :-1]
        x2 = x[:, 1:]
        fitness = np.sum(100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2, axis=1)

    if func_op == 1:  # rastrigin
        fitness = 10 * dim + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis=1)

    if func_op == 2:  # griewank
        i = np.arange(1, dim + 1)
        fitness = 1 + np.sum(x ** 2 / 4000, axis=1) - np.prod(np.cos(x / np.sqrt(i)), axis=1)

    if func_op == 3:  # SIX-HUMP CAMEL FUNCTION
        x1 = x[:, 0]
        x2 = x[:, 1]
        fitness = (4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) * x1 ** 2 + x1 * x2 + (-4 + 4 * x2 ** 2) * x2 ** 2

    return fitness