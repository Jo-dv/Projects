import copy
import math
import os
import time

import numpy as np

from rts import rts_select
from fitness import fitness
from evaluate import evaluate
import select
import crossover
import mutate


def cal_res(last_pi, last_fits):
    MBF = np.array(last_pi).mean()
    avg_fit = np.array(last_fits).mean(axis=0)

    return MBF, avg_fit


class algorithms:
    def __init__(self, n_run, gen, problem, selector):
        self.n_run = n_run
        self.gen = gen
        self.problem = problem
        self.BSF = []
        self.BCP = []
        self.ACP = []
        self.WCP = []
        self.last_ACP = []
        self.last_fits = []
        self.selector = selector

    def evolve(self, pc, n_c, pm, n_m, w=None):
        dim = np.shape(self.problem[1][0])[0] // 100
        if self.problem[0] == 'rosenbrock':
            lower = -30
            upper = 30
            func_op = 0
        if self.problem[0] == 'rastrigin':
            lower = -5.12
            upper = 5.12
            func_op = 1
        if self.problem[0] == 'griewank':
            lower = -600
            upper = 600
            func_op = 2
        if self.problem[0] == 'shc':
            lower = -5
            upper = 5
            func_op = 3

        for x in range(self.n_run):
            population = copy.deepcopy(self.problem[1][x]).reshape(100, dim)
            pool = np.full((np.shape(population)[0], np.shape(population[0])[0]), np.nan)
            popsize = np.shape(population)[0]

            i = 1
            found_BSF = math.inf

            np.set_printoptions(precision=6, suppress=True)
            while i <= self.gen:
                fit = fitness(population, func_op)

                if fit.min() < found_BSF:
                    found_BSF = fit.min()
                if i % (self.gen / 20) == 0 and x == 0:
                    self.BSF.append(found_BSF)
                    self.BCP.append(fit.min())
                    self.ACP.append(fit.mean())
                    self.WCP.append(fit.max())

                evaluation = evaluate(fit)
                time.sleep(0.005)
                os.system('cls')
                print("=" * 32 + f"run{x + 1}" + "=" * (38-len(str(x+1))))
                print("=" * 32 + f"gen{i}" + "=" * (38-len(str(i))))
                print(f"Fitness: {fit.mean(): .17f} \t\tEvaluation: {evaluation.mean(): .17f}")
                print("=" * 73)
                print(population.mean(axis=0))

                for j in range(0, popsize, 2):
                    if self.selector == 'GA': selection = select.bts_select(population, evaluation)
                    if self.selector == 'RTS': selection = population[[np.random.choice(range(0, popsize)) for i in range(2)]]
                    offspring = crossover.SBX(selection, lower, upper, pc, n_c)
                    mutation = mutate.PM(offspring, lower, upper, pm, n_m)
                    pool[j] = mutation[0]
                    pool[j + 1] = mutation[1]
                if self.selector == 'RTS':
                    pool = rts_select(population, pool, w, func_op)
                population = copy.deepcopy(pool)
                i += 1
            self.last_ACP.append(fit.mean())
            self.last_fits.append(fit)

        MBF, avg_fit = cal_res(self.last_ACP, self.last_fits)
        return self.BSF, self.BCP, self.ACP, self.WCP, MBF, avg_fit
