from pathlib import Path
import numpy as np
import pandas as pd


def generate():
    if not Path("problems/check.csv").is_file():
        rosenbrock_problem = np.full((20, 3000), np.nan)
        rastrigin_problem = np.full((20, 3000), np.nan)
        griewank_problem = np.full((20, 3000), np.nan)
        shc_problem = np.full((20, 200), np.nan)
        check = np.ones(1)

        for i in range(20):
            rosenbrock_problem[i] = np.random.uniform(low=-30, high=30, size=3000)
            rastrigin_problem[i] = np.random.uniform(low=-5.12, high=5.12, size=3000)
            griewank_problem[i] = np.random.uniform(low=-600, high=600, size=3000)
            shc_problem[i] = np.random.uniform(low=-5, high=5, size=200)

        pd.DataFrame(rosenbrock_problem).to_csv("problems/rosenbrock_problem.csv", header=None, index=None)
        pd.DataFrame(rastrigin_problem).to_csv("problems/rastrigin_problem.csv", header=None, index=None)
        pd.DataFrame(griewank_problem).to_csv("problems/griewank_problem.csv", header=None, index=None)
        pd.DataFrame(shc_problem).to_csv("problems/shc_problem.csv", header=None, index=None)
        pd.DataFrame(check).to_csv("problems/check.csv", header=None, index=None)

def load(problem):
    return np.array(pd.read_csv('problems/' + problem + '_problem.csv', header=None))
