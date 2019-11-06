import datetime
import json
import os
from time import time as t
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.metrics import confusion_matrix
import streamlit as st
import torch
from IPython import display
from plotly.subplots import make_subplots
from torchvision import transforms
from tqdm import tqdm
import sqlite3
from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.learning import PostPre
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor, NetworkMonitor
from bindsnet.network.nodes import AdaptiveLIFNodes, Input
from bindsnet.network.topology import Connection, LocalConnection
from bindsnet.utils import reshape_locally_connected_weights
import shutil
import hashlib


class LC_SNN:
    def __init__(self, norm=0.48, c_w=-100., n_iter=1000, time_max=250, crop=20,
                 kernel_size=12, n_filters=25, stride=4, intensity=127.5):
        self.type = 'LC_SNN'
        self.norm = norm
        self.c_w = c_w
        self.n_iter = n_iter
        self.calibrated = False
        self.accuracy = None
        self.conf_matrix = None
        self.time_max = time_max
        self.crop = crop
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.stride = stride
        self.intensity = intensity

        self.parameters = {
            'norm': self.norm,
            'c_w': self.c_w,
            'n_iter': self.n_iter,
            'time_max': self.time_max,
            'crop': self.crop,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'n_filters': self.n_filters,
            'intensity': self.intensity
            }
        self.name = hashlib.sha224(str(self.parameters).encode('utf8')).hexdigest()
        self.create_network()

    def create_network(self):
        dt = 1
        self.train_dataset = MNIST(
            PoissonEncoder(time=self.time_max, dt=dt),
            None,
            ".//MNIST",
            download=False,
            train=True,
            transform=transforms.Compose([
                transforms.CenterCrop(self.crop),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * self.intensity)
                ])
            )

        # Hyperparameters
        padding = 0
        conv_size = int((self.crop - self.kernel_size + 2 * padding) / self.stride) + 1
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
        self.n_input = self.crop ** 2
        self.input_layer = Input(n=self.n_input, shape=(1, self.crop, self.crop), traces=True,
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
                            w[fltr1, i, j, fltr2, i, j] = self.c_w

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
        for smth, batch in tqdm(list(zip(range(n_iter), train_dataloader)), ncols=100):
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

        args = self.votes.argsort(descending=True)[:, 0:top_n]
        top_n_votes = torch.zeros(self.votes.shape)
        for i, row in enumerate(args):
            for j, neuron_number in enumerate(row):
                top_n_votes[i, neuron_number] = self.votes[i, neuron_number]
        res = torch.matmul(top_n_votes.type(torch.FloatTensor), sum_output.type(torch.FloatTensor))
        if res.sum(0).item() == 0:
            return -1
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
        return tuple([x, y])

    def predict_many(self, n_iter=6000):
        y = []
        x = []
        self.network.train(False)
        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=1, shuffle=True)

        for i, batch in tqdm(list(zip(range(n_iter), train_dataloader)), ncols=100):
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

    def calibrate(self, n_iter=None, top_n=None):
        print('Calibrating network...')
        if n_iter is None:
            n_iter = self.n_iter
        labels = []
        outputs = []
        self.network.train(False)
        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=1, shuffle=True)
        print('Collecting acrivity data...')
        for i, batch in tqdm(list(zip(range(n_iter), train_dataloader)), ncols=100):
            inpts = {"X": batch["encoded_image"].transpose(0, 1)}
            self.network.run(inpts=inpts, time=self.time_max, input_time_dim=1)
            self._spikes = {
                "X": self.spikes["X"].get("s").view(self.time_max, -1),
                "Y": self.spikes["Y"].get("s").view(self.time_max, -1),
                }
            outputs.append(self._spikes['Y'].sum(0))
            labels.append(batch['label'])
        votes = torch.zeros(10, self.n_output)
        print('Calculating votes...')
        for (label, layer) in tqdm(zip(labels, outputs), ncols=100):
            for i, spike_sum in enumerate(layer):
                votes[label, i] += spike_sum
        for i in range(10):
            votes[i, :] = votes[i, :] / len((np.array(labels) == i).nonzero()[0])
        self.votes = votes
        self.calibrated = True
        if top_n is None:
            top_n = self.votes.shape[1]
        scores = []
        labels_predicted = []
        print('Calculating accuracy...')
        for label, output in tqdm(zip(labels, outputs), ncols=100):
            args = self.votes.argsort(descending=True)[:, 0:top_n]
            top_n_votes = torch.zeros(self.votes.shape)
            for i, row in enumerate(args):
                for j, neuron_number in enumerate(row):
                    top_n_votes[i, neuron_number] = self.votes[i, neuron_number]
            label_predicted = torch.matmul(top_n_votes.type(torch.FloatTensor),
                                           output.type(torch.FloatTensor)).argmax().item()
            if output.sum(0).item() == 0:
                label_predicted = -1
                scores.append(0)
            else:
                if label == label_predicted:
                    scores.append(1)
                else:
                    scores.append(0)

            labels_predicted.append(label_predicted)

        self.accuracy = np.array(scores).mean()
        self.parameters['accuracy'] = self.accuracy
        self.conf_matrix = confusion_matrix(labels, labels_predicted)
        self.network.reset_()

    def calculate_accuracy(self, n_iter=1000, top_n=None, label=None):
        if top_n is None:
            top_n = 10
        if not self.calibrated:
            print('The network is not calibrated!')
            return None
        self.network.train(False)
        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=1, shuffle=True)
        print('Calculating accuracy...')
        x = []
        y = []
        for i, batch in tqdm(list(zip(range(n_iter), train_dataloader)), ncols=100):
            inpts = {"X": batch["encoded_image"].transpose(0, 1)}
            self.network.run(inpts=inpts, time=self.time_max, input_time_dim=1)
            self._spikes = {
                "X": self.spikes["X"].get("s").view(self.time_max, -1),
                "Y": self.spikes["Y"].get("s").view(self.time_max, -1),
                }
            label = batch['label']
            prediction = self.class_from_spikes(top_n=top_n)
            x.append(prediction)
            y.append(label)
        scores = []
        for i in range(len(x)):
            if x[i] == y[i]:
                scores.append(1)
            else:
                scores.append(0)
        scores = np.array(scores)
        self.network.reset_()
        print(f'Accuracy: {scores.mean()}')
        return confusion_matrix(y, x), scores.mean()

    def accuracy_from_top_n(self, n_iter=5000):
        accs = torch.zeros(10, 9, n_iter)
        for label in range(10):
            display.clear_output(wait=True)
            print(f'Calculating accuracy for label {label}...')
            for i in tqdm(range(n_iter), ncols=100):
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

                for top_n in range(1, 10):
                    prediction = self.class_from_spikes(top_n=top_n)
                    if prediction == label:
                        accs[label, top_n-1, i] = 1

        errors = accs.std(axis=-1)

        fig = go.Figure().update_layout(
            title=go.layout.Title(
                text='Accuracy dependence from top_n'
                )
            )

        for label in range(10):
            fig.add_scatter(x=list(range(1, 11)), y=accs.mean(axis=-1)[label, :].numpy(), name=f'label {label}',
                            error_y=dict(array=errors[label, :].numpy(), visible=True))
        fig.add_scatter(x=list(range(1, 11)), y=accs.mean(axis=-1).mean(axis=0).numpy(), name=f'Total',
                        error_y=dict(array=accs.mean(axis=0).std(axis=1).numpy(), visible=True))

        return accs, errors, fig




    def plot_weights_XY(self, width=800, height=800):
        self.weights_XY = reshape_locally_connected_weights(self.network.connections[('X', 'Y')].w,
                                                            n_filters=self.n_filters,
                                                            kernel_size=self.kernel_size,
                                                            conv_size=self.conv_size,
                                                            locations=self.network.connections[('X', 'Y')].locations,
                                                            input_sqrt=self.n_input)


        fig_weights = go.Figure(data=go.Heatmap(z=self.weights_XY.numpy(), colorscale='YlOrBr'))
        fig_weights.update_layout(width=width, height=800,
                                      title=go.layout.Title(
                                          text="Weights XY",
                                          xref="paper"),
                                      margin={'l': 20, 'r': 20, 'b': 20, 't': 40, 'pad': 4},
                                      xaxis=go.layout.XAxis(
                                          title_text='Neuron Index X',
                                          tickmode='array',
                                          tickvals=np.linspace(0, self.weights_XY.shape[0],
                                                               self.output_shape + 1) +
                                                   self.weights_XY.shape[0] / self.output_shape / 2,
                                          ticktext=np.linspace(0, self.output_shape, self.output_shape + 1),
                                          zeroline=False
                                          ),
                                      yaxis=go.layout.YAxis(
                                          title_text='Neuron Index Y',
                                          tickmode='array',
                                          tickvals=np.linspace(0, self.weights_XY.shape[1],
                                                               self.output_shape + 1) +
                                                   self.weights_XY.shape[1] / self.output_shape / 2,
                                          ticktext=np.linspace(0, self.output_shape, self.output_shape + 1),
                                          zeroline=False
                                          )
                                      )
        return fig_weights

    def plot_spikes_Y(self):
        spikes = self._spikes['Y'].transpose(0, 1)
        width = 800
        height = int(width * spikes.shape[0] / spikes.shape[1])
        fig_spikes = go.Figure(data=go.Heatmap(z=spikes.numpy().astype(int), colorscale='YlOrBr'))
        fig_spikes.update_layout(width=width, height=width,
                                 title=go.layout.Title(
                                     text="Y spikes",
                                     xref="paper",
                                     ),
                                 xaxis=go.layout.XAxis(
                                     title_text='Time'
                                     ),
                                 yaxis=go.layout.YAxis(
                                     title_text='Neuron Index'
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

    def accuracy_distribution(self):
        colnames = ['label', 'accuracy']
        accs = pd.DataFrame(columns=colnames)
        for i in range(self.conf_matrix.shape[0]):
            true = 0
            total = 0
            for j in range(1, self.conf_matrix.shape[1]):
                if i != j:
                    total += self.conf_matrix[i][j]
                else:
                    total += self.conf_matrix[i][j]
                    true += self.conf_matrix[i][j]
            accs = accs.append(pd.DataFrame([[i-1, true/total]], columns=colnames), ignore_index=True)

        accs = accs[accs.label != -1]

        accs_distibution_fig = go.Figure(go.Scatter(y=accs['accuracy'].values,
                                                    mode='markers', marker_size=20))
        # accs_distibution_fig.update_traces(marker=dict(size=16))

        accs_distibution_fig.update_layout(width=800, height=800,
                                            title=go.layout.Title(
                                                text="Accuracy Distribution",
                                                xref="paper"),
                                            margin={'l': 20, 'r': 20, 'b': 20, 't': 40, 'pad': 4},
                                            xaxis=go.layout.XAxis(
                                                title_text='Class',
                                                tickmode='array',
                                                tickvals=list(range(10)),
                                                ticktext=list(range(10)),
                                                # zeroline=False
                                                ),
                                            yaxis=go.layout.YAxis(
                                                title_text='Accuracy',
                                                zeroline=False,
                                                range=[0, 1]
                                                # tick0=1,

                                                )
                                            )

        return accs, accs_distibution_fig

    def average_confusion_matrix(self):
        row_sums = self.conf_matrix.sum(axis=1)
        average_confuion_matrix = self.conf_matrix[1:] / row_sums[:, np.newaxis][1:]

        return average_confuion_matrix

    def confusion(self):
        average_confusion_matrix = self.average_confusion_matrix()
        fig_confusion = go.Figure(data=go.Heatmap(z=average_confusion_matrix, colorscale='YlOrBr',
                                                  zmin=0,
                                                  zmax=1
                                                  )
                                  )
        fig_confusion.update_layout(width=800, height=800,
                                    title=go.layout.Title(
                                        text="Average Confusion Matrix",
                                        xref="paper"),
                                    margin={'l': 20, 'r': 20, 'b': 20, 't': 40, 'pad': 4},
                                    xaxis=go.layout.XAxis(
                                        title_text='Output',
                                        tickmode='array',
                                        tickvals=list(range(11)),
                                        ticktext=['No spikes'] + list(range(10)),
                                        zeroline=False
                                        ),
                                    yaxis=go.layout.YAxis(
                                        title_text='Input',
                                        tickmode='array',
                                        tickvals=list(range(11)),
                                        ticktext=list(range(10)),
                                        zeroline=False
                                        )
                                    )
        return fig_confusion
    
    def votes_distribution(self):
        votes_distibution_fig = go.Figure(go.Scatter(y=self.votes.sort(0, descending=True)[0].mean(axis=1).numpy(),
                                                     mode='markers', marker_size=20))
        votes_distibution_fig.update_layout(width=800, height=800,
                                            title=go.layout.Title(
                                                text="Votes Distribution",
                                                xref="paper"),
                                            margin={'l': 20, 'r': 20, 'b': 20, 't': 40, 'pad': 4},
                                            xaxis=go.layout.XAxis(
                                                title_text='Class',
                                                tickmode='array',
                                                tickvals=list(range(10)),
                                                ticktext=list(range(1, 11)),
                                                # zeroline=False
                                                ),
                                            yaxis=go.layout.YAxis(
                                                title_text='Mean Vote',
                                                # tick0=1,

                                                )
                                            )
        return votes_distibution_fig

    def save(self):
        path = f'networks//{self.name}'
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.network, path + '//network')
        if self.calibrated:
            torch.save(self.votes, path + '//votes')
            torch.save(self.accuracy, path + '//accuracy')
            torch.save(self.conf_matrix, path + '//confusion_matrix')

        with open(path + '//parameters.json', 'w') as file:
            json.dump(self.parameters, file)

        if not os.path.exists(r'networks/networks.db'):
            conn = sqlite3.connect(r'networks/networks.db')
            crs = conn.cursor()
            crs.execute('''CREATE TABLE networks(
             id BLOB,
             accuracy REAL,
             n_iter INT,
             type BLOB
             )''')
            conn.commit()
            conn.close()

        conn = sqlite3.connect(r'networks/networks.db')
        crs = conn.cursor()
        crs.execute('SELECT id FROM networks WHERE id = ?', (self.name, ))
        result = crs.fetchone()
        if result:
            pass
        else:
            crs.execute('INSERT INTO networks VALUES (?, ?, ?, ?)', (self.name, self.accuracy, self.n_iter, self.type))

        conn.commit()
        conn.close()

    def delete(self, sure=False):
        if not sure:
            print('Are you sure you want to delete the network? [Y/N]')
            if input() == 'Y':
                shutil.rmtree(f'networks//{self.name}')
                conn = sqlite3.connect(r'networks/networks.db')
                crs = conn.cursor()
                crs.execute(f'DELETE FROM networks WHERE id = ?', (self.name, ))
                conn.commit()
                conn.close()
                print('Network deleted!')
            else:
                print('Deletion canceled...')
        else:
            shutil.rmtree(f'networks//{self.name}')
            conn = sqlite3.connect(r'networks/networks.db')
            crs = conn.cursor()
            crs.execute(f'DELETE FROM networks WHERE id = ?', (self.name, ))
            conn.commit()
            conn.close()
            print('Network deleted!')

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







# TODO: accuraccy(class, top_n)
# TODO: conv connection