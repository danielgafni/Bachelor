from LC_SNN import LC_SNN
import torch

accs = torch.zeros(7)
for i, n_train in enumerate([100, 1000, 2000, 5000, 10000, 30000, 60000]):
    net = LC_SNN(iter=n_train)
    net.train()
    net.calibrate_top_classes()
    accs[i] = net.accuracy()
torch.save(accs, r'gridsearch/acc(n_iter)')