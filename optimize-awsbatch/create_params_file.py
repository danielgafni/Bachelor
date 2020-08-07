import itertools
import json
import numpy as np

from spaceopt import SpaceOpt


def convert(x):
    return [float(item) for item in x]


search_space = dict(
    mean_weight=convert(np.linspace(0.2, 0.7, 10)),
    c_w=convert(np.linspace(-50., -150., 10)),
    n_filters=[25],
    time_max=[250],
    crop=[20],
    kernel_size=[16],
    stride=[4],
    intensity=[127.5],
    tau_pos=convert(np.linspace(2., 20., 10)),
    tau_neg=convert(np.linspace(2., 20., 10)),
    A_pos=convert(np.linspace(-2., 0, 10)),
    A_neg=convert(np.linspace(-2., 0., 10)),
    weight_decay=convert(np.linspace(0, 0.01, 10)),
)


space = SpaceOpt(search_space=search_space, target_name='accuracy', objective='min')

parameters_to_evaluate = space.get_random(num_spoints=3)
for i in range(len(parameters_to_evaluate)):
    parameters_to_evaluate[i]["c_w_min"] = None
    parameters_to_evaluate[i]["c_l"] = True
    parameters_to_evaluate[i]["immutable_name"] = None
    parameters_to_evaluate[i]["foldername"] = None
    parameters_to_evaluate[i]["loaded_from_disk"] = None
    parameters_to_evaluate[i]["n_iter"] = 0
    parameters_to_evaluate[i]["network_type"] = "LC_SNN"

with open('parameters_to_evaluate-test.json', 'w') as file:
    json.dump(parameters_to_evaluate, file)
