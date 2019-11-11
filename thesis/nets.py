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
from bindsnet.network.topology import Connection, LocalConnection, Conv2dConnection
from bindsnet.utils import reshape_locally_connected_weights
import shutil
import hashlib
from statsmodels.stats.proportion import proportion_confint
from PIL import Image


class AbstractSNN:
    def __init__(self, n_iter=1000, norm=0.48, c_w=-100., time_max=250, crop=20,
                 kernel_size=12, n_filters=25, stride=4, intensity=127.5, dt=1,
                 competitive_learn=False,
                 type_='Abstract SNN'):
        self.type = type_
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
        self.dt = dt

        self.parameters = {
            'type': self.type,
            'norm': self.norm,
            'n_iter': self.n_iter,
            'c_w': self.c_w,
            'time_max': self.time_max,
            'crop': self.crop,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'n_filters': self.n_filters,
            'intensity': self.intensity,
            'dt': self.dt
            }

        self.create_network()
        self.name = hashlib.sha224(str(self.parameters).encode('utf8')).hexdigest()

        self.train_dataset = MNIST(
            PoissonEncoder(time=self.time_max, dt=self.dt),
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

    def create_network(self):
        pass

    def train(self, n_iter=None, plot=False, vis_interval=10, app=False):
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
            fig_weights.show()
            fig_spikes.show()

        if app:
            weights_plot = st.plotly_chart(fig_weights)
            spikes_plot = st.plotly_chart(fig_spikes)

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
                    display.clear_output(wait=True)
                    fig_weights = self.plot_weights_XY()
                    fig_spikes = self.plot_spikes()
                    fig_weights.show()
                    fig_spikes.show()
                    cnt += 1

            if app:
                weights_plot.plotly_chart(fig_weights)
                spikes_plot.plotly_chart(fig_spikes)
                cnt += 1

        self.network.reset_()  # Reset state variables
        self.network.train(False)

    def class_from_spikes(self, top_n=None):
        if top_n is None:
            top_n = 10
        sum_output = self._spikes['Y'].sum(0)

        args = self.votes.argsort(axis=0, descending=True)[0:top_n, :]
        top_n_votes = torch.zeros(self.votes.shape)
        for i, top_i in enumerate(args):
            for j, label in enumerate(top_i):
                top_n_votes[label, j] = self.votes[label, j]
        res = torch.matmul(top_n_votes.type(torch.FloatTensor), sum_output.type(torch.FloatTensor))
        if res.sum(0).item() == 0:
            return torch.zeros(10).fill_(-1).type(torch.LongTensor), torch.zeros(10)
        return res.argsort(descending=True), (res / res.sum())[(res / res.sum()).argsort(descending=True)]

    def calibrate(self, n_iter=None, top_n=None):
        print('Calibrating network...')
        if n_iter is None:
            n_iter = self.n_iter
        labels = []
        outputs = []
        self.network.train(False)
        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=1, shuffle=True)
        print('Collecting activity data...')
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
            self.network.reset_()

        print('Calculating votes...')
        for (label, layer) in tqdm(zip(labels, outputs), total=len(labels), ncols=100):
            for i, spike_sum in enumerate(layer):
                votes[label, i] += spike_sum
        for i in range(10):
            votes[i, :] = votes[i, :] / len((np.array(labels) == i).nonzero()[0])
        self.votes = votes
        self.calibrated = True
        if top_n is None:
            top_n = self.votes.shape[0]
        labels_predicted = []
        print('Calculating accuracy...')
        for label, output in tqdm(zip(labels, outputs), total=len(labels), ncols=100):
            args = self.votes.argsort(axis=0, descending=True)[0:top_n, :]
            top_n_votes = torch.zeros(self.votes.shape)
            for i, top_i in enumerate(args):
                for j, label in enumerate(top_i):
                    top_n_votes[label, j] = self.votes[label, j]
            label_predicted = torch.matmul(top_n_votes.type(torch.FloatTensor),
                                           output.type(torch.FloatTensor)).argmax().item()
            if output.sum(0).item() == 0:
                label_predicted = torch.zeros(1).fill_(-1)
            labels_predicted.append(label_predicted)

        self.conf_matrix = confusion_matrix(labels, labels_predicted)
        self.accuracy = self.conf_matrix.trace() / self.conf_matrix.sum()
        self.parameters['accuracy'] = self.accuracy

        return labels, labels_predicted

    def votes_distribution(self):
        votes_distibution_fig = go.Figure(go.Scatter(y=self.votes.sort(0, descending=True)[0].mean(axis=1).numpy(),
                                                     mode='markers', marker_size=15))
        votes_distibution_fig.update_layout(width=800, height=400,
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

    def calculate_accuracy(self, n_iter=1000, top_n=None, k=1):
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
        predictions = []
        for i, batch in tqdm(list(zip(range(n_iter), train_dataloader)), ncols=100):
            inpts = {"X": batch["encoded_image"].transpose(0, 1)}
            self.network.run(inpts=inpts, time=self.time_max, input_time_dim=1)
            self._spikes = {
                "X": self.spikes["X"].get("s").view(self.time_max, -1),
                "Y": self.spikes["Y"].get("s").view(self.time_max, -1),
                }
            label = batch['label']
            prediction, confidence = self.class_from_spikes(top_n=top_n)
            x.append(prediction[0])
            predictions.append(prediction)
            y.append(label)
        scores = []
        for i in range(len(x)):
            if y[i] in predictions[i][0:k]:
                scores.append(1)
            else:
                scores.append(0)
        scores = np.array(scores)
        conf_interval = proportion_confint(scores.sum(), len(scores), 0.05)
        error = (conf_interval[1] - conf_interval[0]) / 2
        print(f'Accuracy: {scores.mean()} with 95% confidence error {round(error, 2)}')
        self.network.reset_()

        return confusion_matrix(y, x), scores.mean(), error

    def accuracy_distribution(self):
        self.network.train(False)
        colnames = ['label', 'accuracy', 'error']
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
            error = (proportion_confint(true, total, alpha=0.05)[1] -
                     proportion_confint(true, total, alpha=0.05)[0]) / 2
            accs = accs.append(pd.DataFrame([[i-1, true/total, error]], columns=colnames), ignore_index=True)

        accs = accs[accs.label != -1]
        accs_distibution_fig = go.Figure(go.Scatter(y=accs['accuracy'].values,
                                                    error_y=dict(array=accs['error'], visible=True),
                                                    mode='markers', marker_size=10))
        accs_distibution_fig.update_layout(width=800, height=400,
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

                                               ),
                                           )

        return accs, accs_distibution_fig

    def accuracy_on_top_n(self, n_iter=5000):
        self.network.train(False)
        scores = torch.zeros(10, 10, n_iter)
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
                    self.network.reset_()

                for top_n in range(1, 11):
                    prediction, confidence = self.class_from_spikes(top_n=top_n)
                    if prediction == label:
                        scores[label, top_n-1, i] = 1

        errors = (proportion_confint(scores.sum(axis=-1), scores.shape[-1], 0.05)[1] -
                  proportion_confint(scores.sum(axis=-1), scores.shape[-1], 0.05)[0])/2

        fig = go.Figure().update_layout(
            title=go.layout.Title(
                text='Accuracy dependence on top_n'
                ),
            xaxis=go.layout.XAxis(
                title_text='top_n',
                tickmode='array',
                tickvals=list(range(10)),
                ticktext=list(range(10)),
                ),
            yaxis=go.layout.YAxis(
                title_text='Accuracy',
                range=[0, 1]
                )
            )

        for label in range(10):
            fig.add_scatter(x=list(range(1, 11)), y=scores.mean(axis=-1)[label, :].numpy(), name=f'label {label}',
                            error_y=dict(array=errors[label, :], visible=True))
        fig.add_scatter(x=list(range(1, 11)), y=scores.mean(axis=-1).mean(axis=0).numpy(), name=f'Total',
                        error_y=dict(array=errors.mean(axis=0), visible=True))

        return scores, errors, fig

    def confusion(self):
        row_sums = self.conf_matrix.sum(axis=1)
        average_confusion_matrix = self.conf_matrix[1:] / row_sums[:, np.newaxis][1:]
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

    def get_weights_XY(self):
        pass

    def plot_weights_XY(self, width=800):
        self.weights_XY = self.get_weights_XY()
        fig_weights_XY = go.Figure(data=go.Heatmap(z=self.weights_XY.numpy(), colorscale='YlOrBr'))
        fig_weights_XY.update_layout(width=width, height=800,
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

        return fig_weights_XY

    def plot_spikes_Y(self):
        spikes = self._spikes['Y'].transpose(0, 1)
        width = 800
        height = int(width * spikes.shape[0] / spikes.shape[1])
        fig_spikes = go.Figure(data=go.Heatmap(z=spikes.numpy().astype(int), colorscale='YlOrBr'))
        fig_spikes.update_layout(width=width, height=height,
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

    def feed_class(self, label, top_n=None, k=1, to_print=True, plot=False):
        self.network.train(False)
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

            prediction, confidence = self.class_from_spikes(top_n=top_n)
            if to_print:
                print(f'Prediction: {prediction[0:k]}\nConfidence: {confidence[0:k]}')
            if plot:
                self.plot_spikes().show()
                plot_image(np.flipud(batch['image'][0, 0, :, :].numpy())).show()

        return prediction[0:k], confidence[0:k]

    def feed_image(self, path, top_n=None, k=1, to_print=True, plot=False):
        self.network.train(False)
        img = Image.open(fp=path).convert('1')
        transform=transforms.Compose([
            transforms.Resize(size=(self.crop, self.crop)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * self.intensity)
            ])
        image = self.intensity - transform(img)
        pe = PoissonEncoder(time=self.time_max, dt=1)
        encoded_image = pe.enc(torch.tensor(np.array(image)).type(torch.FloatTensor),
                               time=self.time_max, transform=True).unsqueeze(0)
        inpts = {'X': encoded_image.transpose(0, 1)}
        self.network.run(inpts=inpts, time=self.time_max, input_time_dim=1)
        self._spikes = {
            "X": self.spikes["X"].get("s").view(self.time_max, -1),
            "Y": self.spikes["Y"].get("s").view(self.time_max, -1),
            }

        prediction, confidence = self.class_from_spikes(top_n=top_n)
        if to_print:
            print(f'Prediction: {prediction[0:k]}\nConfidence: {confidence[0:k]}')
        if plot:
            self.plot_spikes().show()
            plot_image(np.flipud(image.squeeze().numpy())).show()

        return prediction[0:k], confidence[0:k]

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

    def __repr__(self):
        return f'Network with parameters:\n {self.parameters}'


########################################################################################################################


class LC_SNN(AbstractSNN):
    def __init__(self, n_iter=1000, norm=0.48, c_w=-100., time_max=250, crop=20,
                 kernel_size=12, n_filters=25, stride=4, intensity=127.5,
                 competitive_learn=False,):

        super().__init__(n_iter=n_iter, norm=norm, c_w=c_w, time_max=time_max, crop=crop,
                         kernel_size=kernel_size, n_filters=n_filters, stride=stride, intensity=intensity,
                         competitive_learn=competitive_learn,
                         type_='LC_SNN')

    def create_network(self):
        # Hyperparameters
        padding = 0
        conv_size = int((self.crop - self.kernel_size + 2 * padding) / self.stride) + 1
        per_class = int((self.n_filters * conv_size * conv_size) / 10)
        tc_trace = 20.
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

        self.weights_XY = self.get_weights_XY()

    def get_weights_XY(self):
        weights_XY = reshape_locally_connected_weights(self.network.connections[('X', 'Y')].w,
                                                       n_filters=self.n_filters,
                                                       kernel_size=self.kernel_size,
                                                       conv_size=self.conv_size,
                                                       locations=self.network.connections[('X', 'Y')].locations,
                                                       input_sqrt=self.n_input)
        return weights_XY


class CC_SNN(AbstractSNN):
    def __init__(self, norm=50, c_w=-100., n_iter=1000, time_max=250, crop=20,
                 kernel_size=12, n_filters=25, stride=4, intensity=127.5,
                 competitive_learn=False):

        super().__init__(n_iter=n_iter, norm=norm, c_w=c_w, time_max=time_max, crop=crop,
                         kernel_size=kernel_size, n_filters=n_filters, stride=stride, intensity=intensity,
                         competitive_learn=competitive_learn,
                         type_='CC_SNN')

    def create_network(self):
        # Hyperparameters
        padding = 0
        conv_size = int((self.crop - self.kernel_size + 2 * padding) / self.stride) + 1
        tc_trace = 20.
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
        total_kernel_size = int(np.prod(self.kernel_size))
        norm=0.5 * int(np.sqrt(total_kernel_size))

        self.connection_XY = Conv2dConnection(
            self.input_layer,
            self.output_layer,
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
        self.weights_XY_shape = int(np.sqrt(np.prod(self.network.connections[('X', 'Y')].w.shape)))

        self.weights_XY = self.network.connections[('X', 'Y')].w.reshape(self.weights_XY_shape,
                                                                         self.weights_XY_shape)

    def get_weights_XY(self):
        weights_XY = self.network.connections[('X', 'Y')].w.reshape(self.weights_XY_shape,
                                                                    self.weights_XY_shape)
        return weights_XY


def plot_image(image):
    width = 400
    height = int(width * image.shape[0] / image.shape[1])

    fig_img = go.Figure(data=go.Heatmap(z=image, colorscale='YlOrBr'))
    fig_img.update_layout(width=width, height=height,
                          title=go.layout.Title(
                              text="Image",
                              xref="paper",
                              x=0
                              )
                          )

    return fig_img