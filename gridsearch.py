from thesis.nets import C_SNN
import numpy as np
import torch

accs = torch.zeros(np.arange(-200., 0., 10.).shape[0], np.arange(0.1, 1., 0.05).shape[0])
for i, c_w in enumerate(np.arange(-200., 0., 10.)):
    for j, norm in enumerate(np.arange(0.1, 1., 0.05)):
        net = C_SNN(c_w=c_w, norm=norm, n_iter=10000)
        net.train(10000)
        net.calibrate(10000)
        net.calculate_accuracy(1000)
        net.save()
        accs[i, j] = net.accuracy
torch.save(accs, 'accs')