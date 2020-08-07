import itertools
import json
import numpy as np

from spaceopt import SpaceOpt


def convert(x):
    return [float(item) for item in x]


search_space = dict(
    mean_weight=convert(np.linspace(0.2, 0.7, 10)),
    c_w=convert(np.linspace(-50., -150., 10)),
    tau_pos=convert(np.linspace(2., 20., 10)),
    tau_neg=convert(np.linspace(2., 20., 10)),
    A_pos=convert(np.linspace(-2., 0, 10)),
    A_neg=convert(np.linspace(-2., 0., 10)),
    weight_decay=convert(np.linspace(0, 0.01, 10)),
)

space = SpaceOpt(search_space=search_space, target_name='accuracy', objective='min')

parameters_numpy = np.empty((0, 7))
parameters = space.get_random(num_spoints=3)
for par in parameters:
    parameters_net = np.array([
        par["mean_weight"], par["c_w"], par["tau_pos"], par["tau_neg"], par["A_pos"], par["A_neg"], par["weight_decay"]
    ])
    parameters_numpy = np.vstack((parameters_numpy, parameters_net))

np.save('optimize-awsbatch/parameters/test', parameters_numpy)
