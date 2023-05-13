import copy
import math

import numpy as np

from evaluate import evaluate
from fitness import fitness


def find_similar(c, d):
    similarity = np.sum(np.power(c - d, 2))
    return np.sqrt(similarity)


def rts_select(population, mutant, w, op):  # restricted tournament selection
    new_population = copy.deepcopy(population)
    popsize = np.shape(population)[0]
    dim = np.shape(population[0])[0]

    for i in range(popsize):
        distance = math.inf
        position = 0
        mutant_eval = evaluate(fitness(mutant[i].reshape(-1, dim), op))
        idx = [np.random.choice(range(0, popsize)) for _ in range(w)]

        comparison_set = population[idx]
        for j in range(w):
            similarity = find_similar(mutant[i], comparison_set[j])
            if similarity < distance:
                distance = similarity
                competitor = comparison_set[j]
                position = idx[j]

        competitor_eval = evaluate(fitness(competitor.reshape(-1, dim), op))

        if mutant_eval > competitor_eval:
            new_population[position] = mutant[i]

    return new_population
