from LC_SNN import LC_SNN
from IPython.display import clear_output
import os
import torch


def gridsearch(norms, comp_weights, n_iter, name, accuracy_iter=1000):
    accs = torch.zeros(len(norms), len(comp_weights))

    for i, norm in enumerate(norms):
        for j, c_w in enumerate(comp_weights):
            clear_output(wait=True)
            print(f'Current accuracies:\n{accs}')
            print(f'Best accuracy:\n{accs.max()}')
            print(f'Current parameters:\nnorm={norm}\tcompetitive_weight={c_w}')
            net = LC_SNN(norm=norm, competitive_weight=c_w, n_iter=n_iter)
            net.train(n_iter)
            top_classes, votes = net.calibrate_top_classes(n_iter)
            acc = net.accuracy(accuracy_iter)
            accs[i, j] = acc
            if not os.path.exists(f'gridsearch//{name}//{norm}-{c_w}-{n_iter}'):
                os.makedirs(f'gridsearch//{name}//{norm}-{c_w}-{n_iter}')
            net.network.save('gridsearch//' + name + f'//{norm}-{c_w}-{n_iter}//network')
            torch.save(top_classes,'gridsearch//' + name + f'//{norm}-{c_w}-{n_iter}//top_classes')
            torch.save(votes,'gridsearch//' + name + f'//{norm}-{c_w}-{n_iter}//votes')
            print()
    torch.save(accs, f'gridsearch//' + name + '//accuracies')
    return accs