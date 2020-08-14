import argparse
import numpy as np
import subprocess
import json
import os
import random


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


BOUNDS = [(0.2, 0.7), (-200., 0.), (2., 20.), (2., 20.), (-2., 0.), (-2., 0.), (0., 0.01)]

# search_space = dict(
#     mean_weight=convert(np.linspace(0.2, 0.7, 10)),
#     c_w=convert(np.linspace(-150., -50., 10)),
#     tau_pos=convert(np.linspace(2., 20., 10)),
#     tau_neg=convert(np.linspace(2., 20., 10)),
#     A_pos=convert(np.linspace(-2., 0, 10)),
#     A_neg=convert(np.linspace(-2., 0., 10)),
#     weight_decay=convert(np.linspace(0, 0.01, 10)),
# )


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', type=str, required=True)
    p.add_argument('--id', type=str, required=True)

    args = p.parse_args()

    population_old_path = f"optimize_awsbatch/parameters/{args.id}.npy"
    population_old = np.load(population_old_path)

    if args.mode == 'generate':
        df = DifferentialEvolution(func=None, bounds=BOUNDS, population=population_old)
        df.new_generation()

        new_population = df.population
        np.save(f"optimize_awsbatch/parameters/{args.id}-new.npy", new_population)
        print(f'New population generated. Path:\n{f"optimize_awsbatch/parameters/{args.id}-new.npy"}')

    elif args.mode == 'selection':
        df = DifferentialEvolution(func=None, bounds=BOUNDS, population=population_old)

        population_new_path = f"optimize_awsbatch/parameters/{args.id}-new.npy"
        subprocess.run(["aws", "s3", "cp",
                        f"s3://danielgafni-personal/bachelor/parameters/{args.id}.npy",
                        f"{population_new_path}", "--recursive"])
        population_new = np.load(population_new_path)

        scores_new_path = f"optimize_awsbatch/scores/{args.id}-new"
        subprocess.run(["aws", "s3", "cp",
                        f"s3://danielgafni-personal/bachelor/scores/{args.id}-new", f"{scores_new_path}", "--recursive"])

        scores_old_path = f"optimize_awsbatch/scores/{args.id}"
        subprocess.run(["aws", "s3", "cp",
                        f"s3://danielgafni-personal/bachelor/scores/{args.id}", f"{scores_old_path}",
                        "--recursive"])
        scores_old = np.empty(len(population_new))
        for i in range(len(population_new)):
            with open(scores_old_path + f"/{i}.json", "r") as file:
                score = json.load(file)
                scores_old[i] = score["patch_voting"]["accuracy"]
        np.save(f"{scores_old_path}.npy", scores_old)
        subprocess.run(["rm", "-r", scores_old_path])

        scores_new_path = f"optimize_awsbatch/scores/{args.id}-new"
        subprocess.run(["aws", "s3", "cp",
                        f"s3://danielgafni-personal/bachelor/scores/{args.id}-new", f"{scores_new_path}", "--recursive"])
        scores_new = np.empty(len(population_new))
        for i in range(len(population_new)):
            with open(scores_new_path+f"/{i}.json", "r") as file:
                score = json.load(file)
                scores_new[i] = score["patch_voting"]["accuracy"]
        np.save(f"{scores_new_path}.npy", scores_new)
        subprocess.run(["rm", "-r", scores_new_path])

        df.old_scores = scores_old
        df.scores = scores_new
        df.old_population = population_old
        df.population = population_new

        df.selection()

        population = df.population

        np.save(f"optimize_awsbatch/parameters/{int(args.id) + 1}.npy", population)
        print(f'New population generated. Path:\n{f"optimize_awsbatch/parameters/{int(args.id) + 1}.npy"}')

    else:
        raise NotImplementedError(f"Wrong mode: {args.mode}")
