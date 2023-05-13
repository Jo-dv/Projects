import problems
from run import algorithms


def main(algorithm, problem, run, gen, pc, n_c, pm, n_m, w=None):
    problems.generate()

    test = algorithms(run, gen, (problem, problems.load(problem)), algorithm)
    BSF, BCP, ACP, WCP, MBF, avg_fits = test.evolve(pc, n_c, pm, n_m, w)
    print(MBF)


if __name__ == '__main__':
    main('GA', 'shc', 10, 150, 1.0, 10, 0.005, 10, 2)
