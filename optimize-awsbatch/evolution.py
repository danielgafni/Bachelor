import argparse
from .optim import DifferentialEvolution
import numpy as np
import subprocess
import json


BOUNDS = None


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', type=str, required=True)
    p.add_argument('--id', type=str, required=True)

    args = p.parse_args()

    population_old_path = f"parameters/{args.id}.npy"
    population_old = np.load(population_old_path)

    if args.mode == 'generate':
        df = DifferentialEvolution(func=None, bounds=BOUNDS, population=population_old)
        df.new_generation()

        new_population = df.population()
        np.save(f"parameters/{args.id}-new.npy", new_population)
        print(f'New population generated. Path\n{f"parameters/{args.id}-new.npy"}')

    if args.mode == 'selection':
        df = DifferentialEvolution(func=None, bounds=BOUNDS, population=population_old)

        population_new_path = f"optimize-awsbatch/parameters/{id}-new.npy"
        subprocess.run(["aws", "s3", "cp",
                        f"s3://danielgafni-personal/bachelor/parameters/{id}.npy",
                        f"{population_new_path}", "--recursive"])
        population_new = np.load(population_new_path)

        scores_old_path = f"optimize-awsbatch/scores/{id}"
        scores_old = np.load(scores_old_path)

        scores_new_path = f"optimize-awsbatch/scores/{id}-new"
        subprocess.run(["aws", "s3", "cp",
                        f"s3://danielgafni-personal/bachelor/scores/{id}-new", f"{scores_new_path}", "--recursive"])
        scores_new = np.empty(len(population_new))
        for i in range(len(population_new)):
            with open(scores_new_path+f"/i.json", "r") as file:
                score = json.load(file)
                scores_new[i] = score["accuracy"]["patch_voting"]
        np.save(scores_new_path, scores_new)

        df.old_scores = scores_old
        df.old_population = population_old
        df.population = population_new

        df.selection()

        population = df.population

        np.save(f"parameters/{int(args.id) + 1}.npy", population)
        print(f'New population generated. Path\n{f"parameters/{int(args.id) + 1}.npy"}')
