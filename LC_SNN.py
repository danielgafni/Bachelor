import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

from time import time as t
import datetime
from tqdm import tqdm
from IPython import display


from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.network import Network, load
from bindsnet.learning import PostPre, WeightDependentPostPre
from bindsnet.network.monitors import Monitor, NetworkMonitor
from bindsnet.network.nodes import AdaptiveLIFNodes, Input
from bindsnet.network.topology import LocalConnection, Connection
from bindsnet.analysis.plotting import (
    plot_input,
    plot_conv2d_weights,
    plot_voltages,
    plot_spikes
    )

import streamlit as st


class LC_SNN:
    def __init__(self, norm=0.5, competitive_weight=-100., n_iter=1000, load=False):
        self.n_iter = n_iter
        if not load:
            self.create_network(norm=norm, competitive_weight=competitive_weight)
        else:
            self.norm = None
            self.competitive_weight = None
            self.n_iter = None
            self.time_max = 30


    def plot(self):
        plt.figure()
        test_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=1, shuffle=True)

        for whatever, batch in list(zip([0], test_dataloader)):
            #Processing
            inpts = {"X": batch["encoded_image"].transpose(0, 1)}
            label = batch["label"]
            self.network.run(inpts=inpts, time=self.time_max, input_time_dim=1)

            #Visualization
            # Optionally plot various simulation information.
            inpt_axes = None
            inpt_ims = None
            spike_ims = None
            spike_axes = None
            weights1_im = None
            voltage_ims = None
            voltage_axes = None
            image = batch["image"].view(28, 28)

            inpt = inpts["X"].view(self.time_max, 784).sum(0).view(28, 28)
            weights_XY = self.connection_XY.w
            weights_YY = self.connection_YY.w

            self._spikes = {
                "X": self.spikes["X"].get("s").view(self.time_max, -1),
                "Y": self.spikes["Y"].get("s").view(self.time_max, -1),
                }
            _voltages = {"Y": self.voltages["Y"].get("v").view(self.time_max, -1)}

            inpt_axes, inpt_ims = plot_input(
                image, inpt, label=label, axes=inpt_axes, ims=inpt_ims
                )
            spike_ims, spike_axes = plot_spikes(self._spikes, ims=spike_ims, axes=spike_axes)
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
            weights_XY = weights_XY.reshape(28, 28, -1)
            weights_to_display = torch.zeros(0, 28*25)
            i = 0
            while i < 625:
                for j in range(25):
                    weights_to_display_row = torch.zeros(28, 0)
                    for k in range(25):
                        weights_to_display_row = torch.cat((weights_to_display_row, weights_XY[:, :, i]), dim=1)
                        i += 1
                    weights_to_display = torch.cat((weights_to_display, weights_to_display_row), dim=0)
            im1 = ax1.imshow(weights_to_display.numpy())
            im2 = ax2.imshow(weights_YY.reshape(5*5*25, 5*5*25).numpy())
            f.colorbar(im1, ax=ax1)
            f.colorbar(im2, ax=ax2)
            ax1.set_title('XY weights')
            ax2.set_title('YY weights')
            f.show()
            voltage_ims, voltage_axes = plot_voltages(
                _voltages, ims=voltage_ims, axes=voltage_axes
                )
        self.network.reset_()  # Reset state variables
        return weights_to_display

    def create_network(self, norm=0.5, competitive_weight=-100.):
        self.norm = norm
        self.competitive_weight = competitive_weight
        self.time_max = 30
        dt = 1
        intensity = 127.5

        self.train_dataset = MNIST(
            PoissonEncoder(time=self.time_max, dt=dt),
            None,
            "MNIST",
            download=False,
            train=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
                )
            )

        # Hyperparameters
        n_filters = 25
        kernel_size = 12
        stride = 4
        padding = 0
        conv_size = int((28 - kernel_size + 2 * padding) / stride) + 1
        per_class = int((n_filters * conv_size * conv_size) / 10)
        tc_trace = 20.  # grid search check
        tc_decay = 20.
        thresh = -52
        refrac = 5

        wmin = 0
        wmax = 1

        # Network
        self.network = Network(learning=True)
        self.GlobalMonitor = NetworkMonitor(self.network, state_vars=('v', 's', 'w'))


        self.input_layer = Input(n=784, shape=(1, 28, 28), traces=True)

        self.output_layer = AdaptiveLIFNodes(
            n=n_filters * conv_size * conv_size,
            shape=(n_filters, conv_size, conv_size),
            traces=True,
            thres=thresh,
            trace_tc=tc_trace,
            tc_decay=tc_decay,
            theta_plus=0.05,
            tc_theta_decay=1e6)


        self.connection_XY = LocalConnection(
            self.input_layer,
            self.output_layer,
            n_filters=n_filters,
            kernel_size=kernel_size,
            stride=stride,
            update_rule=PostPre,
            norm=norm, #1/(kernel_size ** 2),#0.4 * kernel_size ** 2,  # norm constant - check
            nu=[1e-4, 1e-2],
            wmin=wmin,
            wmax=wmax)

        # competitive connections
        w = torch.zeros(n_filters, conv_size, conv_size, n_filters, conv_size, conv_size)
        for fltr1 in range(n_filters):
            for fltr2 in range(n_filters):
                if fltr1 != fltr2:
                    # change
                    for i in range(conv_size):
                        for j in range(conv_size):
                            w[fltr1, i, j, fltr2, i, j] = competitive_weight

        self.connection_YY = Connection(self.output_layer, self.output_layer, w=w)

        self.network.add_layer(self.input_layer, name='X')
        self.network.add_layer(self.output_layer, name='Y')

        self.network.add_connection(self.connection_XY, source='X', target='Y')
        self.network.add_connection(self.connection_YY, source='Y', target='Y')

        self.network.add_monitor(self.GlobalMonitor, name='Network')

        self.spikes = {}
        for layer in set(self.network.layers):
            self.spikes[layer] = Monitor(self.network.layers[layer], state_vars=["s"], time=self.time_max)
            self.network.add_monitor(self.spikes[layer], name="%s_spikes" % layer)
            #print('GlobalMonitor.state_vars:', self.GlobalMonitor.state_vars)

        self.voltages = {}
        for layer in set(self.network.layers) - {"X"}:
            self.voltages[layer] = Monitor(self.network.layers[layer], state_vars=["v"], time=self.time_max)
            self.network.add_monitor(self.voltages[layer], name="%s_voltages" % layer)



    def train(self, n_iter=None, plot=False, vis_interval=10):
        n_iter = self.n_iter
        self.network.train(True)
        print('Training network...')
        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=1, shuffle=True)
        progress_bar = st.progress(0)
        status = st.empty()
        t_start = t()
        cnt = 0

        if plot:
            f = plt.figure(figsize=(15, 15))
            plt.imshow(torch.randint(0, 2, (700, 700)).numpy(), cmap='YlOrBr')
            plt.colorbar()
            plot_weights = st.pyplot(f)

        for smth, batch in tqdm(list(zip(range(n_iter), train_dataloader))):
            progress_bar.progress(int((smth + 1)/n_iter*100))
            t_now = t()
            time_from_start = str(datetime.timedelta(seconds=(int(t_now - t_start))))
            speed = round((smth + 1) / (t_now - t_start), 2)
            time_left = str(datetime.timedelta(seconds=int((n_iter - smth) / speed)))
            status.text(f'{smth + 1}/{n_iter} [{time_from_start}] < [{time_left}], {speed}it/s')
            inpts = {"X": batch["encoded_image"].transpose(0, 1)}
            self.network.run(inpts=inpts, time=self.time_max, input_time_dim=1)

            if plot:
                if (t_now - t_start) / vis_interval > cnt:
                    weights_XY = self.connection_XY.w
                    weights_XY = weights_XY.reshape(28, 28, -1)
                    weights_to_display = torch.zeros(0, 28*25)
                    i = 0
                    while i < 625:
                        for j in range(25):
                            weights_to_display_row = torch.zeros(28, 0)
                            for k in range(25):
                                weights_to_display_row = torch.cat((weights_to_display_row, weights_XY[:, :, i]), dim=1)
                                i += 1
                            weights_to_display = torch.cat((weights_to_display, weights_to_display_row), dim=0)
                    self.weights_XY = weights_to_display.numpy()

                    self._spikes = {
                        "X": self.spikes["X"].get("s").view(self.time_max, -1),
                        "Y": self.spikes["Y"].get("s").view(self.time_max, -1),
                        }
                    fig_w = plt.figure(figsize=(15, 15))
                    plt.title('Weights_XY')
                    plt.imshow(self.weights_XY, cmap='YlOrBr')
                    plt.colorbar()
                    plot_weights.pyplot(fig_w)
                    plt.close(fig_w)
                    cnt += 1
                else:
                    pass
        self.network.reset_()  # Reset state variables
        self.network.train(False)

    def fit(self, X, y, n_iter=None):
        n_iter = self.n_iter
        self.network.train(True)
        print('Fitting network...')
        smth = 0

        for inp, label in tqdm(zip(X[:n_iter], y[:n_iter])):
            inpts = {"X": inp}
            self.network.run(inpts=inpts, time=self.time_max, input_time_dim=1)

            weights_XY = self.connection_XY.w
            weights_XY = weights_XY.reshape(28, 28, -1)
            weights_to_display = torch.zeros(0, 28*25)
            i = 0
            while i < 625:
                for j in range(25):
                    weights_to_display_row = torch.zeros(28, 0)
                    for k in range(25):
                        weights_to_display_row = torch.cat((weights_to_display_row, weights_XY[:, :, i]), dim=1)
                        i += 1
                    weights_to_display = torch.cat((weights_to_display, weights_to_display_row), dim=0)

            self.weights_XY = weights_to_display.numpy()

            spike_ims = None
            spike_axes = None

            self._spikes = {
                "X": self.spikes["X"].get("s").view(self.time_max, -1),
                "Y": self.spikes["Y"].get("s").view(self.time_max, -1),
                }


        self.network.reset_()  # Reset state variables

        self.network.train(False);


    def predict(self, n_iter=6000):
        y = []
        x = []
        self.network.train(False)
        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=1, shuffle=True)

        for i, batch in tqdm(list(zip(range(n_iter), train_dataloader))):
            inpts = {"X": batch["encoded_image"].transpose(0, 1)}

            self.network.run(inpts=inpts, time=self.time_max, input_time_dim=1)

            self._spikes = {
                "X": self.spikes["X"].get("s").view(self.time_max, -1),
                "Y": self.spikes["Y"].get("s").view(self.time_max, -1),
                }

            y.append(int(batch['label']))
            output = self._spikes['Y'].type(torch.int).sum(0)
            x.append(output)

        self.network.reset_()  # Reset state variables

        return tuple([x, y])

    def plot_weights_XY(self):
        weights_XY = self.network.connections[('X', 'Y')].w.reshape(28, 28, -1)
        weights_to_display = torch.zeros(0, 28*25)
        i = 0
        while i < 625:
            for j in range(25):
                weights_to_display_row = torch.zeros(28, 0)
                for k in range(25):
                    weights_to_display_row = torch.cat((weights_to_display_row, weights_XY[:, :, i]), dim=1)
                    i += 1
                weights_to_display = torch.cat((weights_to_display, weights_to_display_row), dim=0)
        plt.figure(figsize=(15, 15))
        plt.imshow(weights_to_display, cmap='YlOrBr')
        plt.colorbar()

    def load(self, file_path):
        self.network = load(file_path)

        self.spikes = {}
        for layer in set(self.network.layers):
            self.spikes[layer] = Monitor(self.network.layers[layer], state_vars=["s"], time=self.time_max)
            self.network.add_monitor(self.spikes[layer], name="%s_spikes" % layer)
            #print('GlobalMonitor.state_vars:', self.GlobalMonitor.state_vars)

        self.voltages = {}
        for layer in set(self.network.layers) - {"X"}:
            self.voltages[layer] = Monitor(self.network.layers[layer], state_vars=["v"], time=self.time_max)
            self.network.add_monitor(self.voltages[layer], name="%s_voltages" % layer)

        weights_XY = self.network.connections[('X', 'Y')].w

        weights_XY = weights_XY.reshape(28, 28, -1)
        weights_to_display = torch.zeros(0, 28*25)
        i = 0
        while i < 625:
            for j in range(25):
                weights_to_display_row = torch.zeros(28, 0)
                for k in range(25):
                    weights_to_display_row = torch.cat((weights_to_display_row, weights_XY[:, :, i]), dim=1)
                    i += 1
                weights_to_display = torch.cat((weights_to_display, weights_to_display_row), dim=0)

        self.weights_XY = weights_to_display.numpy()

        # weights_to_display = torch.zeros(0, 28*25)
        # i = 0
        # while i < 625:
        #     for j in range(25):
        #         weights_to_display_row = torch.zeros(12, 0)
        #         for k in range(25):
        #             if len((weights_XY[:, :, i]>0).nonzero()) > 0:
        #                 weights_to_display_row = torch.cat((weights_to_display_row, weights_XY[:, :, i]), dim=1)
        #                 i += 1
        #             weights_to_display = torch.cat((weights_to_display, weights_to_display_row), dim=0)
        #
        # self.weights_XY_formatted = weights_to_display.numpy()



    def show_neuron(self, n):
        weights_to_show = self.network.connections[('X', 'Y')].w.reshape(28, 28, -1)
        weights_to_show[:, :, n-1] = torch.ones(28, 28)
        weights_to_display = torch.zeros(0, 28*25)
        i = 0
        while i < 625:
            for j in range(25):
                weights_to_display_row = torch.zeros(28, 0)
                for k in range(25):
                    weights_to_display_row = torch.cat((weights_to_display_row, weights_to_show[:, :, i]), dim=1)
                    i += 1
                weights_to_display = torch.cat((weights_to_display, weights_to_display_row), dim=0)

        plt.figure(figsize=(15, 15))
        plt.title('Weights XY')

        plt.imshow(weights_to_display.numpy(), cmap='YlOrBr')
        plt.colorbar()

    def top_classes(self, n_iter=100):
        print('Calibrating top classes for each neuron...')
        (x, y) = self.predict(n_iter=n_iter)
        votes = torch.zeros(11, 625)
        votes[10, :] = votes[10, :].fill_(1/n_iter)
        for (label, layer) in zip(y, x):
            for i, spike_sum in enumerate(layer):
                votes[label, i] += spike_sum
        for i in range(10):
            votes[i, :] = votes[i, :] / len((np.array(y) == i).nonzero()[0])
        top_classes = votes.argsort(dim=0, descending=True).numpy()
        top_classes_formatted = np.where(top_classes!=10, top_classes, None)
        return top_classes_formatted, votes

    def accuracy(self, n_iter):
        self.network.train(False)
        top_classes, votes = self.top_classes(n_iter=n_iter)

        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=1, shuffle=True)

        print('Calculating accuracy...')

        x = []
        y = []

        for i, batch in tqdm(list(zip(range(n_iter), train_dataloader))):
            inpts = {"X": batch["encoded_image"].transpose(0, 1)}

            self.network.run(inpts=inpts, time=self.time_max, input_time_dim=1)

            self._spikes = {
                "X": self.spikes["X"].get("s").view(self.time_max, -1),
                "Y": self.spikes["Y"].get("s").view(self.time_max, -1),
                }

            output = self._spikes['Y'].type(torch.int).sum(0)
            top3 = output.argsort()[0:3]
            label = int(batch['label'])
            n_1, n_2, n_3 = top3[0], top3[1], top3[2]
            n_best = n_1
            if output[n_1] * votes[label][n_1] > output[n_2] * votes[label][n_2]:
                if output[n_2] * votes[label][n_2] > output[n_3] * votes[label][n_3]:
                    pass
                else:
                    if output[n_1] * votes[label][n_1] > output[n_3] * votes[label][n_3]:
                        pass
            else:
                if output[n_2] * votes[label][n_2] > output[n_3] * votes[label][n_3]:
                    n_best = n_2
                else:
                    if output[n_3] * votes[label][n_3] > output[n_1] * votes[label][n_1]:
                        n_best = n_3

            x.append(top_classes[0][n_best])
            y.append(label)

        corrects = []
        for i in range(len(x)):
            if x[i] == y[i]:
                corrects.append(1)
            else:
                corrects.append(0)
        corrects = np.array(corrects)

        self.network.reset_()
        print(f'Accuracy: {corrects.mean()}')
        return corrects.mean()

    def score(self, X, y_correct):
        self.network.train(False)
        top_classes, votes = self.top_classes(n_iter=len(y_correct))
        print('Calculating score...')

        x = []

        for inp, label in zip(X, y_correct):
            inpts = {'X': inp}

            self.network.run(inpts=inpts, time=self.time_max, input_time_dim=1)

            self._spikes = {
                "X": self.spikes["X"].get("s").view(self.time_max, -1),
                "Y": self.spikes["Y"].get("s").view(self.time_max, -1),
                }

            output = self._spikes['Y'].type(torch.int).sum(0)
            top3 = output.argsort()[0:3]
            n_1, n_2, n_3 = top3[0], top3[1], top3[2]
            # n_best = n_1
            # print(votes.shape)
            # if votes[label][n_1] > votes[label][n_2]:
            #     if votes[label][n_2] > votes[label][n_3]:
            #         pass
            #     else:
            #         if votes[label][n_1] > votes[label][n_3]:
            #             pass
            # else:
            #     if votes[label][n_2] > votes[label][n_3]:
            #         n_best = n_2
            #     else:
            #         if votes[label][n_3] > votes[label][n_1]:
            #             n_best = n_3

            # x.append(top_classes[0][n_best])
            x.append(top_classes[0][n_1])

        corrects = []
        for i in range(len(x)):
            if x[i] == y_correct[i]:
                corrects.append(1)
            else:
                corrects.append(0)
        corrects = np.array(corrects)

        self.network.reset_()
        print(f'Accuracy: {corrects.mean()}')
        return corrects.mean()

    def get_params(self, **args):
        return {'norm': self.norm,
                'competitive_weight': self.competitive_weight,
                'n_iter': self.n_iter
                }

    def set_params(self, norm, competitive_weight, n_iter):
        display.clear_output(wait=True)
        return LC_SNN(norm=norm, competitive_weight=competitive_weight, n_iter=n_iter)

    def __repr__(self):
        # return f'LC_SNN network with parameters:\nnorm = {self.norm}\ncompetitive_weights={self.competitive_weight}' \
        #        f'\nn_iter={self.n_iter}'
        return f'LC_SNN network with parameters:\n {self.get_params()}'