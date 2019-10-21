import datetime
import json
import os
from time import time as t

import numpy as np
import plotly.graph_objs as go
from sklearn.metrics import confusion_matrix
import streamlit as st
import torch
from IPython import display
from plotly.subplots import make_subplots
from torchvision import transforms
from tqdm import tqdm

from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.learning import PostPre
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor, NetworkMonitor
from bindsnet.network.nodes import AdaptiveLIFNodes, Input
from bindsnet.network.topology import Connection, LocalConnection
from bindsnet.utils import reshape_locally_connected_weights


class LC_SNN:
    def __init__(self, norm=0.2, c_w=-100., n_iter=1000, time_max=250, cropped_size=20,
                 kernel_size=12, n_filters=25, stride=4, intensity=127.5,
                 load=False):
        self.norm = norm
        self.c_w = c_w
        self.n_iter = n_iter
        self.calibrated = False
        self.conf_matrix = None
        self.acc = None
        self.time_max = time_max
        self.cropped_size = cropped_size
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.stride = stride
        self.intensity = intensity

        self.parameters = {
            'norm': self.norm,
            'c_w': self.c_w,
            'n_iter': self.n_iter,
            'time_max': self.time_max,
            'cropped_size': self.cropped_size,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'n_filters': self.n_filters,
            'intensity': self.intensity
            }

        if not load:
            self.create_network(norm=norm, c_w=c_w)
        else:
            pass

    def create_network(self, norm=0.5, c_w=-100.):
        self.norm = norm
        self.c_w = c_w
        dt = 1


        self.train_dataset = MNIST(
            PoissonEncoder(time=self.time_max, dt=dt),
            None,
            "MNIST",
            download=False,
            train=True,
            transform=transforms.Compose([
                transforms.CenterCrop(self.cropped_size),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * self.intensity)
                ])
            )

        # Hyperparameters

        padding = 0
        conv_size = int((self.cropped_size - self.kernel_size + 2 * padding) / self.stride) + 1
        per_class = int((self.n_filters * conv_size * conv_size) / 10)
        tc_trace = 20.  # grid search check
        tc_decay = 20.
        thresh = -52
        refrac = 2

        self.wmin = 0
        self.wmax = 1

        # Network
        self.network = Network(learning=True)
        self.GlobalMonitor = NetworkMonitor(self.network, state_vars=('v', 's', 'w'))

        self.n_input = self.cropped_size ** 2
        self.input_layer = Input(n=self.n_input, shape=(1, self.cropped_size, self.cropped_size), traces=True,
                                 refrac=refrac)

        self.n_output = self.n_filters * conv_size * conv_size
        self.output_shape = int(np.sqrt(self.n_output))
        self.output_layer = AdaptiveLIFNodes(
            n=self.n_output,
            shape=(self.n_filters, conv_size, conv_size),
            traces=True,
            thres=thresh,
            trace_tc=tc_trace,
            tc_decay=tc_decay,
            theta_plus=0.05,
            tc_theta_decay=1e6)

        self.connection_XY = LocalConnection(
            self.input_layer,
            self.output_layer,
            n_filters=self.n_filters,
            kernel_size=self.kernel_size,
            stride=self.stride,
            update_rule=PostPre,
            norm=self.norm,  # 1/(kernel_size ** 2),#0.4 * self.kernel_size ** 2,  # norm constant - check
            nu=[1e-4, 1e-2],
            wmin=self.wmin,
            wmax=self.wmax)

        # competitive connections
        w = torch.zeros(self.n_filters, conv_size, conv_size, self.n_filters, conv_size, conv_size)
        for fltr1 in range(self.n_filters):
            for fltr2 in range(self.n_filters):
                if fltr1 != fltr2:
                    # change
                    for i in range(conv_size):
                        for j in range(conv_size):
                            w[fltr1, i, j, fltr2, i, j] = c_w

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

        self._spikes = {
            "X": self.spikes["X"].get("s").view(self.time_max, -1),
            "Y": self.spikes["Y"].get("s").view(self.time_max, -1),
            }

        self.voltages = {}
        for layer in set(self.network.layers) - {"X"}:
            self.voltages[layer] = Monitor(self.network.layers[layer], state_vars=["v"], time=self.time_max)
            self.network.add_monitor(self.voltages[layer], name="%s_voltages" % layer)

        self.stride = self.stride
        self.conv_size = conv_size
        self.conv_prod = int(np.prod(conv_size))
        self.kernel_prod = int(np.prod(self.kernel_size))

        self.weights_XY = reshape_locally_connected_weights(self.network.connections[('X', 'Y')].w,
                                                            n_filters=self.n_filters,
                                                            kernel_size=self.kernel_size,
                                                            conv_size=self.conv_size,
                                                            locations=self.network.connections[('X', 'Y')].locations,
                                                            input_sqrt=self.n_input)

    def train(self, n_iter=None, plot=False, vis_interval=10, debug=False):
        if n_iter is None:
            n_iter = self.n_iter
        self.network.train(True)
        print('Training network...')
        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=1, shuffle=True)
        progress_bar = st.progress(0)
        status = st.empty()

        cnt = 0
        if plot:
            fig_weights = self.plot_weights_XY()
            fig_spikes = self.plot_spikes()

            weights_plot = st.plotly_chart(fig_weights)
            spikes_plot = st.plotly_chart(fig_spikes)


            if debug:
                fig_weights.show()
                fig_spikes.show()


        t_start = t()
        for smth, batch in tqdm(list(zip(range(n_iter), train_dataloader))):
            progress_bar.progress(int((smth + 1) / n_iter * 100))
            t_now = t()
            time_from_start = str(datetime.timedelta(seconds=(int(t_now - t_start))))
            speed = (smth + 1) / (t_now - t_start)
            time_left = str(datetime.timedelta(seconds=int((n_iter - smth) / speed)))
            status.text(f'{smth + 1}/{n_iter} [{time_from_start}] < [{time_left}], {round(speed, 2)}it/s')
            inpts = {"X": batch["encoded_image"].transpose(0, 1)}
            self.network.run(inpts=inpts, time=self.time_max, input_time_dim=1)

            self._spikes = {
                "X": self.spikes["X"].get("s").view(self.time_max, -1),
                "Y": self.spikes["Y"].get("s").view(self.time_max, -1),
                }

            if plot:
                if (t_now - t_start) / vis_interval > cnt:
                    fig_weights = self.plot_weights_XY()
                    fig_spikes = self.plot_spikes()

                    weights_plot.plotly_chart(fig_weights)
                    spikes_plot.plotly_chart(fig_spikes)

                    cnt += 1

                    if debug:
                        display.clear_output(wait=True)
                        fig_weights.show()
                        fig_spikes.show()


                else:
                    pass
        self.network.reset_()  # Reset state variables
        self.network.train(False)

    def class_from_spikes(self, top_n=None):
        if top_n is None:
            top_n = self.votes.shape[1]
        sum_output = self._spikes['Y'].sum(0)
        if sum_output.sum(0).item() == 0:
            return None
        args = self.votes.argsort(descending=True)[:, 0:top_n]
        top_n_votes = torch.zeros(self.votes.shape)
        for i, row in enumerate(args):
            for j, neuron_number in enumerate(row):
                top_n_votes[i, neuron_number] = self.votes[i, neuron_number]
        res = torch.matmul(top_n_votes.type(torch.FloatTensor), sum_output.type(torch.FloatTensor))
        return res.argmax().item()

    def debug(self, n_iter):
        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=1, shuffle=True)
        
        x = []
        y = []

        for i, batch in list(zip(range(n_iter), train_dataloader)):
            
            inpts = {"X": batch["encoded_image"].transpose(0, 1)}
            self.network.run(inpts=inpts, time=self.time_max, input_time_dim=1)
            self._spikes = {
                "X": self.spikes["X"].get("s").view(self.time_max, -1),
                "Y": self.spikes["Y"].get("s").view(self.time_max, -1),
                }
            prediction = self.class_from_spikes()
            correct = batch["label"][0]
            x.append(prediction)
            y.append(correct)
            print(f'Network prediction: {prediction}\nCorrect label: {correct}\n')
            #self.plot_spikes()
            
        return tuple([x, y])

    def predict_many(self, n_iter=6000):
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
            output = self._spikes['Y'].sum(0)
            x.append(output)
            y.append(int(batch['label']))

        self.network.reset_()  # Reset state variables

        return tuple([x, y])

    def calibrate_2(self, n_iter=100):
        print('Calibrating top classes for each neuron...')
        (x, y) = self.predict_many(n_iter=n_iter)
        votes = torch.zeros(11, self.n_output)
        votes[10, :] = votes[10, :].fill_(1 / (2 * n_iter))
        for (label, layer) in zip(y, x):
            for i, spike_sum in enumerate(layer):
                votes[label, i] += spike_sum
        for i in range(10):
            votes[i, :] = votes[i, :] / len((np.array(y) == i).nonzero()[0])
        top_classes = votes.argmax(dim=0)
        # top_classes_formatted = np.where(top_classes!=10, top_classes, None)
        self.top_classes = top_classes
        self.votes = votes
        self.calibrated = True
        return top_classes, votes

    def calibrate(self, n_iter=100):
        print('Calibrating top classes for each neuron...')
        (x, y) = self.predict_many(n_iter=n_iter)
        votes = torch.zeros(10, self.n_output)
        for (label, layer) in zip(y, x):
            for i, spike_sum in enumerate(layer):
                votes[label, i] += spike_sum
        for i in range(10):
            votes[i, :] = votes[i, :] / len((np.array(y) == i).nonzero()[0])
        top_classes = votes.argmax(dim=0)
        # top_classes_formatted = np.where(top_classes!=10, top_classes, None)
        self.top_classes = top_classes
        self.votes = votes
        self.calibrated = True
        return top_classes, votes

    def predict(self, batch, top_n):
        inpts = {"X": batch["encoded_image"].transpose(0, 1)}
        self.network.run(inpts=inpts, time=self.time_max, input_time_dim=1)
        self._spikes = {
            "X": self.spikes["X"].get("s").view(self.time_max, -1),
            "Y": self.spikes["Y"].get("s").view(self.time_max, -1),
            }
        prediction = self.class_from_spikes()
        label = batch['label']
        return tuple([prediction, label])

    def accuracy(self, n_iter=1000, top_n=None):
        self.network.train(False)
        if not self.calibrated:
            self.calibrate(n_iter=self.n_iter)

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

            prediction = self.class_from_spikes(top_n=top_n)
            label = batch['label']

            x.append(prediction)
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
        self.conf_matrix = confusion_matrix(y, x)
        self.acc = corrects.mean()

    def accuracy_nonzero(self, n_iter=1000, top_n=None):
        self.network.train(False)
        if not self.calibrated:
            self.calibrate(n_iter=self.n_iter)

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
            label = batch['label']
            prediction = self.class_from_spikes(top_n=top_n)
            if not prediction is None:
                x.append(prediction)
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
        self.conf_matrix = confusion_matrix(y, x)
        self.acc = corrects.mean()

    def plot_confusion_matrix(self):
        width = 800
        height = int(width * self.conf_matrix.shape[0] / self.conf_matrix.shape[1])

        fig_conf_matrix = go.Figure(data=go.Heatmap(z=self.conf_matrix, colorscale='YlOrBr'))
        fig_conf_matrix.update_layout(width=width, height=height,
                                 title=go.layout.Title(
                                     text="Confusion matrix",
                                     xref="paper",
                                     x=0
                                     )
                                 )

        return fig_conf_matrix

    def get_params(self, **args):
        return {'norm': self.norm,
                'c_w': self.c_w,
                'n_iter': self.n_iter
                }

    # TODO: add new params

    def set_params(self, norm, c_w, n_iter):
        display.clear_output(wait=True)
        return LC_SNN(norm=norm, c_w=c_w, n_iter=n_iter)

    # TODO: add new params

    def plot_weights_XY(self):
        self.weights_XY = reshape_locally_connected_weights(self.network.connections[('X', 'Y')].w,
                                                            n_filters=self.n_filters,
                                                            kernel_size=self.kernel_size,
                                                            conv_size=self.conv_size,
                                                            locations=self.network.connections[('X', 'Y')].locations,
                                                            input_sqrt=self.n_input)

        width = 800
        height = int(width * self.weights_XY.shape[0] / self.weights_XY.shape[1])

        fig_weights = go.Figure(data=go.Heatmap(z=self.weights_XY.numpy(), colorscale='YlOrBr'))
        fig_weights.update_layout(width=width, height=height,
                                  title=go.layout.Title(
                                      text="Weights XY",
                                      xref="paper",
                                      x=0
                                      )
                                  )
        return fig_weights

    def plot_spikes_Y(self):
        spikes = self._spikes['Y'].transpose(0, 1)
        width = 800
        height = int(width * spikes.shape[0] / spikes.shape[1])

        fig_spikes = go.Figure(data=go.Heatmap(z=spikes.numpy().astype(int), colorscale='YlOrBr'))
        fig_spikes.update_layout(width=width, height=height,
                                 title=go.layout.Title(
                                     text="Y spikes",
                                     xref="paper",
                                     x=0
                                     )
                                 )

        return fig_spikes

    def plot_spikes(self):
        spikes_X = self._spikes['X'].transpose(0, 1)
        spikes_Y = self._spikes['Y'].transpose(0, 1)
        width_X = spikes_X.shape[0] / (spikes_X.shape[0] + spikes_Y.shape[0])
        width_Y = 1 - width_X
        fig_spikes = make_subplots(rows=2, cols=1, subplot_titles=['X spikes', 'Y spikes'],
                                   vertical_spacing=0.04, row_width=[width_Y, width_X])

        trace_X = go.Heatmap(z=spikes_X.numpy().astype(int), colorscale='YlOrBr')
        trace_Y = go.Heatmap(z=spikes_Y.numpy().astype(int), colorscale='YlOrBr')
        fig_spikes.add_trace(trace_X, row=1, col=1)
        fig_spikes.add_trace(trace_Y, row=2, col=1)
        fig_spikes.update_layout(width=800, height=800,
                                 title=go.layout.Title(
                                     text="Network Spikes",
                                     xref="paper",
                                     )
                                 )

        return fig_spikes

    def save(self, path):
        #path = 'networks//' + path
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.network, path + '//network')
        if self.calibrated:
            torch.save(self.top_classes, path + '//top_classes')
            torch.save(self.votes, path + '//votes')
        with open(path + '//parameters.json', 'w') as file:
            json.dump(self.parameters, file)

    def feed_class(self, label, top_n=None, plot=False):
        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=1, shuffle=True)

        batch = next(iter(train_dataloader))
        while batch['label'] != label:
            batch = next(iter(train_dataloader))
        else:
            inpts = {"X": batch["encoded_image"].transpose(0, 1)}
            self.network.run(inpts=inpts, time=self.time_max, input_time_dim=1)
            self._spikes = {
                "X": self.spikes["X"].get("s").view(self.time_max, -1),
                "Y": self.spikes["Y"].get("s").view(self.time_max, -1),
                }

            prediction = self.class_from_spikes(top_n=top_n)
            print(f'Prediction: {prediction}')
            if plot:
                self.plot_spikes().show()
                plot_image(batch).show()

        return prediction

    def __repr__(self):
        # return f'LC_SNN network with parameters:\nnorm = {self.norm}\nc_ws={self.c_w}' \
        #        f'\nn_iter={self.n_iter}'
        return f'LC_SNN network with parameters:\n {self.parameters}'


def load_LC_SNN(path):
    if os.path.exists(path + '//parameters.json'):
        with open(path + '//parameters.json', 'r') as file:
            parameters = json.load(file)
            norm = parameters['norm']
            c_w = parameters['c_w']
            n_iter = parameters['n_iter']
            time_max = parameters['time_max']
            cropped_size = parameters['cropped_size']
            kernel_size = parameters['kernel_size']
            n_filters = parameters['n_filters']
            stride = parameters['stride']
            #intensity = parameters['intensity']

    top_classes = None

    votes = None

    net = LC_SNN(norm=norm, c_w=c_w, n_iter=n_iter, time_max=time_max, cropped_size=cropped_size,
                 kernel_size=kernel_size, n_filters=n_filters, stride=stride)#, intensity=intensity)

    if os.path.exists(path + '//top_classes'):
        top_classes = torch.load(path + '//top_classes')
        net.calibrated = True

    if os.path.exists(path + '//votes'):
        votes = torch.load(path + '//votes')

    network = torch.load(path + '//network')

    net.network = network
    net.top_classes = top_classes
    net.votes = votes

    net.spikes = {}
    for layer in set(net.network.layers):
        net.spikes[layer] = Monitor(net.network.layers[layer], state_vars=["s"], time=net.time_max)
        net.network.add_monitor(net.spikes[layer], name="%s_spikes" % layer)

    net._spikes = {
        "X": net.spikes["X"].get("s").view(net.time_max, -1),
        "Y": net.spikes["Y"].get("s").view(net.time_max, -1),
        }

    return net


def plot_image(batch):
    image = batch['image']

    width = 400
    height = int(width * image.shape[0] / image.shape[1])

    fig_img = go.Figure(data=go.Heatmap(z=np.flipud(image[0, 0, :, :].numpy()), colorscale='YlOrBr'))
    fig_img.update_layout(width=width, height=height,
                             title=go.layout.Title(
                                 text="Image",
                                 xref="paper",
                                 x=0
                                 )
                             )

    return fig_img