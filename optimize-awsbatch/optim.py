import random
import numpy as np


class DifferentialEvolution:
    def __init__(self, func, bounds, F=0.8, rec=0.9, pop_size=None, population=None):
        self.n = len(bounds)
        self.func = func
        self.bounds = (np.array([bound[0] for bound in bounds]),
                       np.array([bound[1] for bound in bounds]))
        self.F = F
        self.rec = rec

        self.population = None
        self.scores = None
        self.old_population = None
        self.old_scores = None

        if pop_size is None:
            if population is None:
                pop_size = 10 * self.n
            else:
                pop_size = len(population)

        self.pop_size = pop_size

        if population is None:
            self.init_population()
        else:
            self.population = population

    def init_population(self):
        # create initial population
        self.population = np.array([
            np.random.uniform(self.bounds[0], self.bounds[1]) for _ in range(self.pop_size)
        ])
        self.scores = np.empty(self.pop_size)
        for i, vector in enumerate(self.population):
            self.scores[i] = self.func(vector)

    def new_generation(self):
        self.old_population = self.population.copy()
        for i in range(self.pop_size):
            # mutation
            v1, v2, v3 = self.old_population[random.sample(range(self.pop_size), 3)]
            mutated_vector = np.clip(v1 + self.F * (v2 - v3), self.bounds[0], self.bounds[1])
            # crossover
            genes_to_mutate = np.random.binomial(1, self.rec, size=self.n).astype(np.bool)
            self.population[i][genes_to_mutate] = mutated_vector[genes_to_mutate]

    def evaluate(self):
        self.old_scores = self.scores.copy()
        for i, vector in enumerate(self.population):
            self.scores[i] = self.func(vector)

    def selection(self):
        losers = self.scores > self.old_scores
        self.population[losers] = self.old_population[losers]
        self.scores[losers] = self.old_scores[losers]

    def step(self):
        self.new_generation()
        self.selection()
        self.evaluate()

    def evolution(self, max_iter=100, score=-np.inf):
        i = 0
        history = ([], [], [], [])
        while self.scores.min() > score and i < max_iter:
            history[0].append(self.scores.min())
            history[1].append(self.scores.mean())
            history[2].append(self.scores.max())
            history[3].append(self.convergence)
            self.step()
            i += 1
        return history

    @property
    def convergence(self):
        return self.scores.std() / self.scores.mean()

    @property
    def result(self):
        return self.population[self.scores == self.scores.min()], self.scores.min()
