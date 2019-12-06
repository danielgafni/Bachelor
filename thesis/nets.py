import datetime
import hashlib
import json
import os
import shutil
import sqlite3
from time import time as t

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import torch
from IPython import display, get_ipython
from PIL import Image
from plotly.subplots import make_subplots
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from tqdm import tqdm, tqdm_notebook

from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.learning import PostPre
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor, NetworkMonitor
from bindsnet.network.nodes import AdaptiveLIFNodes, Input
from bindsnet.network.topology import Connection, Conv2dConnection, LocalConnection, SparseConnection
from bindsnet.utils import reshape_locally_connected_weights

tqdm_train = tqdm

def in_ipynb():
    try:
        cfg = get_ipython().config
        parent_appname = str(cfg['IPKernelApp']['parent_appname'])
        notebook_name = 'traitlets.config.loader.LazyConfigValue'
        if notebook_name in parent_appname:
            return True
        else:
            return False
    except NameError:
        return False
    except AttributeError:
        return False


if in_ipynb():
    tqdm = tqdm_notebook
    ncols = None
else:
    ncols = 100


class AbstractSNN:
    def __init__(self, mean_weight=0.26, c_w=-100., time_max=250, crop=20,
                 kernel_size=12, n_filters=25, stride=4, intensity=127.5, dt=1,
                 c_l=False, nu=None,
                 type_='Abstract SNN', immutable_name=False, foldername=None):
        self.n_iter_counter = 0
        if nu is None and c_l:
            nu = [-1, -0.1]
        self.nu = nu
        self.type = type_
        self.mean_weight = mean_weight
        self.c_w = c_w
        self.n_iter = 0
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
        self.c_l = c_l
        self.immutable_name = immutable_name
        self.foldername = foldername

        self.create_network()

        print(f'Created {self.type} network {self.name} with parameters\n{self.parameters}\n')

    @property
    def parameters(self):
        parameters = {
            'type': self.type,
            'mean_weight': self.mean_weight,
            'n_iter': self.n_iter,
            'c_w': self.c_w,
            'time_max': self.time_max,
            'crop': self.crop,
            'kernel_size': self.kernel_size,
            'kernel_prod': self.kernel_prod,
            'stride': self.stride,
            'n_filters': self.n_filters,
            'intensity': self.intensity,
            'dt': self.dt,
            'c_l': self.c_l,
            'nu': self.nu
            }
        return parameters

    @property
    def name(self):
        if self.immutable_name:
            return self.foldername
        else:
            return hashlib.sha224(str(self.parameters).encode('utf8')).hexdigest()

    def train(self, n_iter=None, plot=False, vis_interval=30, app=False):
        encoded_dataset = MNIST(
            PoissonEncoder(time=self.time_max, dt=self.dt),
            None,
            './/MNIST',
            download=False,
            train=True,
            transform=transforms.Compose([
                transforms.CenterCrop(self.crop),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * self.intensity)
                ])
            )
        train_dataset = encoded_dataset
        train_dataset.data = encoded_dataset.data[:50000, :, :]

        if n_iter is None:
            n_iter = 5000
        self.network.train(True)
        print('Training network...')
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=True)
        cnt = 0
        if plot:
            fig_weights_XY = self.plot_weights_XY()
            fig_weights_YY = self.plot_weights_YY()
            fig_spikes = self.plot_spikes()
            fig_weights_XY.show()
            fig_weights_YY.show()
            fig_spikes.show()

        t_start = t()
        for smth, batch in tqdm_train(list(zip(range(n_iter), train_dataloader))):
            t_now = t()
            time_from_start = str(datetime.timedelta(seconds=(int(t_now - t_start))))
            speed = (smth + 1) / (t_now - t_start)
            time_left = str(datetime.timedelta(seconds=int((n_iter - smth) / speed)))
            inpts = {'X': batch['encoded_image'].transpose(0, 1)}
            self.network.run(inpts=inpts, time=self.time_max, input_time_dim=1)
            self.n_iter += 1
            self.parameters['n_iter'] += 1

            self._spikes = {
                'X': self.spikes['X'].get('s').view(self.time_max, -1),
                'Y': self.spikes['Y'].get('s').view(self.time_max, -1),
                }

            if plot:
                if (t_now - t_start) / vis_interval > cnt:
                    display.clear_output(wait=True)
                    fig_weights_XY = self.plot_weights_XY()
                    fig_weights_YY = self.plot_weights_YY()
                    fig_spikes = self.plot_spikes()
                    fig_weights_XY.show()
                    fig_weights_YY.show()
                    fig_spikes.show()
                    cnt += 1

        self.network.reset_()
        self.network.train(False)

    def class_from_spikes(self, top_n=None):
        if top_n == 0:
            raise ValueError('top_n can\'t be zero')
        if top_n is None:
            top_n = 10
        args = self.votes.argsort(axis=0, descending=True)[0:top_n, :]
        top_n_votes = torch.zeros(self.votes.shape)
        for i, top_i in enumerate(args):
            for j, label in enumerate(top_i):
                top_n_votes[label, j] = self.votes[label, j]
        sum_outputs = self.spikes['Y'].get('s').squeeze(1).sum(0)
        res = top_n_votes.type(torch.FloatTensor) * sum_outputs.flatten().type(torch.FloatTensor)
        res = res.view(10, self.n_filters, self.conv_size, self.conv_size)
        res = res.max(1).values.sum((1, 2)).argsort(0, descending=True)

        if res.sum(0).item() == 0:
            return torch.zeros(10).fill_(-1).type(torch.LongTensor)
        return res

    def collect_activity(self, n_iter=None):
        self.network.train(False)
        if n_iter is None:
            n_iter = 5000
        encoded_dataset = MNIST(
            PoissonEncoder(time=self.time_max, dt=self.dt),
            None,
            './/MNIST',
            download=False,
            train=True,
            transform=transforms.Compose([
                transforms.CenterCrop(self.crop),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * self.intensity)
                ])
            )

        calibratation_dataset = encoded_dataset
        calibratation_dataset.data = encoded_dataset.data[50000:, :, :]

        calibration_dataloader = torch.utils.data.DataLoader(
            calibratation_dataset, batch_size=1, shuffle=True)

        print('Collecting activity data...')

        labels = []
        outputs = []

        for i, batch in tqdm(list(zip(range(n_iter), calibration_dataloader)), ncols=ncols):
            inpts = {'X': batch['encoded_image'].transpose(0, 1)}
            self.network.run(inpts=inpts, time=self.time_max, input_time_dim=1)
            self._spikes = {
                'X': self.spikes['X'].get('s').view(self.time_max, -1),
                'Y': self.spikes['Y'].get('s').view(self.time_max, -1),
                }
            outputs.append(self._spikes['Y'].sum(0).numpy())
            labels.append(batch['label'].item())
            self.network.reset_()

        data = {'outputs': outputs, 'labels': labels}
        self.save()
        torch.save(data, f'networks//{self.name}//activity_data')

    def calibrate(self, n_iter=None):
        encoded_dataset = MNIST(
            PoissonEncoder(time=self.time_max, dt=self.dt),
            None,
            './/MNIST',
            download=False,
            train=True,
            transform=transforms.Compose([
                transforms.CenterCrop(self.crop),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * self.intensity)
                ])
            )
        calibratation_dataset = encoded_dataset
        calibratation_dataset.data = encoded_dataset.data[50000:, :, :]
        calibratation_dataset.targets = encoded_dataset.targets[50000:]


        self.network.train(False)
        self.network.reset_()

        print('Calibrating network...')
        if n_iter is None:
            n_iter = 5000
        labels = []
        outputs = []

        calibration_dataloader = torch.utils.data.DataLoader(
            calibratation_dataset, batch_size=1, shuffle=True)

        print('Collecting activity data...')

        for i, batch in tqdm(list(zip(range(n_iter), calibration_dataloader)), ncols=ncols):
            inpts = {'X': batch['encoded_image'].transpose(0, 1)}
            self.network.run(inpts=inpts, time=self.time_max, input_time_dim=1)
            self._spikes = {
                'X': self.spikes['X'].get('s').view(self.time_max, -1),
                'Y': self.spikes['Y'].get('s').view(self.time_max, -1),
                }
            outputs.append(self._spikes['Y'].sum(0).numpy())
            labels.append(batch['label'].item())
            self.network.reset_()

        print('Calculating votes...')
        votes = torch.zeros(10, self.n_output)
        for (label, layer) in tqdm(zip(labels, outputs), total=len(labels), ncols=ncols):
            for i, spike_sum in enumerate(layer):
                votes[label, i] += spike_sum
        for i in range(10):
            votes[i, :] = votes[i, :] / len((np.array(labels) == i).nonzero()[0])
        self.votes = votes
        self.calibrated = True

    def calibrate_lc(self, n_iter=None):
        if n_iter is None:
            n_iter = 5000

        if not os.path.exists(f'networks//{self.name}//activity_data'):
            self.collect_activity(n_iter=n_iter)

        data = torch.load(f'networks//{self.name}//activity_data')
        outputs = data['outputs']
        labels = data['labels']

        print('Calibrating classifier...')

        self.classifier = SGDClassifier(n_jobs=-1)
        self.classifier.fit(outputs, labels)

    def calculate_accuracy_lc(self, n_iter=None):
        test_dataset = MNIST(
            PoissonEncoder(time=self.time_max, dt=self.dt),
            None,
            './/MNIST',
            download=False,
            train=False,
            transform=transforms.Compose([
                transforms.CenterCrop(self.crop),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * self.intensity)
                ])
            )
        print('Calculating accuracy...')
        self.network.reset_()

        # if not self.calibrated:
        #     print('The network is not calibrated!')
        #     return None
        self.network.train(False)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=True)
        x = []
        y = []
        print('Collecting activity data...')
        for i, batch in tqdm(list(zip(range(n_iter), test_dataloader)), ncols=ncols):
            inpts = {'X': batch['encoded_image'].transpose(0, 1)}
            self.network.run(inpts=inpts, time=self.time_max, input_time_dim=1)
            self._spikes = {
                'X': self.spikes['X'].get('s').view(self.time_max, -1),
                'Y': self.spikes['Y'].get('s').view(self.time_max, -1),
                }
            label = batch['label'].item()
            x.append(self._spikes['Y'].sum(0).numpy())
            y.append(label)

            score = self.classifier.score(x, y)
            y_predict = self.classifier.predict(x)

        self.conf_matrix = confusion_matrix(y, y_predict)
        self.accuracy = score

    def votes_distribution(self):
        votes_distibution_fig = go.Figure(go.Scatter(y=self.votes.sort(0, descending=True)[0].mean(axis=1).numpy(),
                                                     mode='markers', marker_size=15))
        votes_distibution_fig.update_layout(width=800, height=400,
                                            title=go.layout.Title(
                                                text='Votes Distribution',
                                                xref='paper'),
                                            margin={'l': 20, 'r': 20, 'b': 20, 't': 40, 'pad': 4},
                                            xaxis=go.layout.XAxis(
                                                title_text='Class',
                                                tickmode='array',
                                                tickvals=list(range(10)),
                                                ticktext=list(range(1, 11)),
                                                zeroline=False
                                                ),
                                            yaxis=go.layout.YAxis(
                                                title_text='Mean Vote',
                                                zeroline=False
                                                )
                                            )
        return votes_distibution_fig

    def calculate_accuracy(self, n_iter=1000, top_n=None):
        test_dataset = MNIST(
            PoissonEncoder(time=self.time_max, dt=self.dt),
            None,
            './/MNIST',
            download=False,
            train=False,
            transform=transforms.Compose([
                transforms.CenterCrop(self.crop),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * self.intensity)
                ])
            )
        self.network.reset_()
        if top_n is None:
            top_n = 10
        if not self.calibrated:
            print('The network is not calibrated!')
            return None
        self.network.train(False)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=True)
        x = []
        y = []
        for i, batch in tqdm(list(zip(range(n_iter), test_dataloader)), ncols=ncols):
            inpts = {'X': batch['encoded_image'].transpose(0, 1)}
            self.network.run(inpts=inpts, time=self.time_max, input_time_dim=1)
            self._spikes = {
                'X': self.spikes['X'].get('s').view(self.time_max, -1),
                'Y': self.spikes['Y'].get('s').view(self.time_max, -1),
                }
            label = batch['label'].item()
            prediction = self.class_from_spikes(top_n=top_n)
            x.append(prediction[0].item())
            y.append(label)

        scores = []
        for i in range(len(x)):
            if y[i] == x[i]:
                scores.append(1)
            else:
                scores.append(0)

        scores = np.array(scores)
        error = np.sqrt(scores.mean() * (1 - scores.mean()) / n_iter)
        print(f'Accuracy: {scores.mean()} with std {round(error, 2)}')

        self.conf_matrix = confusion_matrix(y, x)
        self.accuracy = scores.mean()
        self.error = error

    def accuracy_distribution(self):
        self.network.train(False)
        colnames = ['label', 'accuracy', 'error']
        accs = pd.DataFrame(columns=colnames)
        if self.conf_matrix.shape[0] == 10:
            for i in range(self.conf_matrix.shape[1]):
                true = self.conf_matrix[i, i]
                total = self.conf_matrix[i, :].sum()

                error = np.sqrt(true / total * (1 - true / total) / total)

                accs = accs.append(pd.DataFrame([[i, true / total, error]], columns=colnames), ignore_index=True)

        if self.conf_matrix.shape[1] == 11:
            for i in range(self.conf_matrix.shape[1]):
                true = self.conf_matrix[i, i]
                total = self.conf_matrix[i, :].sum()

                error = np.sqrt(true / total * (1 - true / total) / total)

                accs = accs.append(pd.DataFrame([[i - 1, true / total, error]], columns=colnames), ignore_index=True)

            accs = accs[accs['label'] != -1]

        accs_distibution_fig = go.Figure(go.Scatter(y=accs['accuracy'].values,
                                                    error_y=dict(array=accs['error'], visible=True,
                                                                 color='purple', width=5),
                                                    mode='markers', marker_size=5))
        accs_distibution_fig.update_layout(width=800, height=400,
                                           title=go.layout.Title(
                                               text='Accuracy Distribution',
                                               xref='paper'),
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

    def competition_distribution(self):
        w = self.network.connections[('Y', 'Y')].w
        w_comp = []
        for fltr1 in range(w.size(0)):
            for fltr2 in range(w.size(3)):
                if fltr1 != fltr2:
                    for i in range(w.size(1)):
                        for j in range(w.size(2)):
                            w_comp.append(w[fltr1, i, j, fltr2, i, j])
        w_comp = torch.tensor(w_comp)
        fig = go.Figure(go.Histogram(x=w_comp))
        fig.update_layout(width=800, height=500,
                          title=go.layout.Title(
                              text='Competition weights histogram',
                              xref='paper'),
                          margin={'l': 20, 'r': 20, 'b': 20, 't': 40, 'pad': 4},
                          xaxis=go.layout.XAxis(
                              title_text='Weight',
                              ),
                          yaxis=go.layout.YAxis(
                              title_text='Quantity',
                              zeroline=False,
                              ))

        return w_comp, fig

    def accuracy_on_top_n(self, n_iter=1000, labels=False):
        self.network.reset_()
        if not self.calibrated:
            print('The network is not calibrated!')
            return None
        self.network.train(False)

        if labels:
            scores = torch.zeros(10, 10, n_iter)
            for label in range(10):
                label_dataset = MNIST(
                    PoissonEncoder(time=self.time_max, dt=self.dt),
                    None,
                    './/MNIST',
                    download=False,
                    train=False,
                    transform=transforms.Compose([
                        transforms.CenterCrop(self.crop),
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: x * self.intensity)
                        ])
                    )
                label_indices = (label_dataset.targets == label).nonzero().flatten()
                label_dataset.data = torch.index_select(label_dataset.data, 0, label_indices)
                label_dataset.targets = label_dataset.targets[label_dataset.targets == label]

                test_dataloader = torch.utils.data.DataLoader(
                    label_dataset, batch_size=1, shuffle=True)

                display.clear_output(wait=True)
                print(f'Calculating accuracy for label {label}...')
                for i in tqdm(range(n_iter), ncols=ncols):
                    batch = next(iter(test_dataloader))

                    inpts = {'X': batch['encoded_image'].transpose(0, 1)}
                    self.network.run(inpts=inpts, time=self.time_max, input_time_dim=1)
                    self._spikes = {
                        'X': self.spikes['X'].get('s').view(self.time_max, -1),
                        'Y': self.spikes['Y'].get('s').view(self.time_max, -1),
                        }
                    self.network.reset_()

                    for top_n in range(1, 11):
                        prediction = self.class_from_spikes(top_n=top_n)[0]
                        if prediction == label:
                            scores[label, top_n - 1, i] = 1

            # errors = (proportion_confint(scores.sum(axis=-1), scores.shape[-1], 0.05)[1] -
            #           proportion_confint(scores.sum(axis=-1), scores.shape[-1], 0.05)[0]) / 2

            errors = ((scores.sum(axis=-1) / scores.shape[-1] * (1 - scores.sum(axis=-1)
                                                                 / scores.shape[-1]) / scores.shape[-1]) ** 0.5).numpy()

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
                                error_y=dict(array=errors[label, :], visible=True, width=5))
            fig.add_scatter(x=list(range(1, 11)), y=scores.mean(axis=-1).mean(axis=0).numpy(), name=f'Total',
                            error_y=dict(array=errors.mean(axis=0), visible=True, width=5))

            return scores, errors, fig

        else:
            test_dataset = MNIST(
                PoissonEncoder(time=self.time_max, dt=self.dt),
                None,
                './/MNIST',
                download=False,
                train=False,
                transform=transforms.Compose([
                    transforms.CenterCrop(self.crop),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x * self.intensity)
                    ])
                )


            test_dataloader = torch.utils.data.DataLoader(
                test_dataset, batch_size=1, shuffle=True)
            scores = torch.zeros(n_iter, 10)
            for i, batch in tqdm(list(zip(range(n_iter), test_dataloader)), ncols=ncols):
                inpts = {'X': batch['encoded_image'].transpose(0, 1)}
                self.network.run(inpts=inpts, time=self.time_max, input_time_dim=1)
                self._spikes = {
                    'X': self.spikes['X'].get('s').view(self.time_max, -1),
                    'Y': self.spikes['Y'].get('s').view(self.time_max, -1),
                    }
                label = batch['label'].item()
                for top_n in range(1, 11):
                    prediction = self.class_from_spikes(top_n=top_n)[0].item()
                    if prediction == label:
                        scores[i, top_n-1] = 1

            res = scores.mean(dim=0)
            errors = ((1 - res) * res / n_iter) ** 0.5
            fig = go.Figure(go.Scatter(x=list(range(1, 11)), y=res.numpy(),
                                       error_y=dict(array=errors, visible=True, width=5)))
            fig.update_layout(
                title=go.layout.Title(
                    text='Accuracy dependence on top_n'
                    ),
                xaxis=go.layout.XAxis(
                    title_text='top_n',
                    tickmode='array',
                    tickvals=list(range(1, 11)),
                    ticktext=list(range(1, 11)),
                    ),
                yaxis=go.layout.YAxis(
                    title_text='Accuracy',
                    range=[0, 1]
                    )
                )

            return scores, errors, fig

    def confusion(self):
        row_sums = self.conf_matrix.sum(axis=1)
        average_confusion_matrix = np.nan_to_num(self.conf_matrix / row_sums)
        fig_confusion = go.Figure(data=go.Heatmap(z=average_confusion_matrix, colorscale='YlOrBr',
                                                  zmin=0,
                                                  zmax=1
                                                  )
                                  )
        fig_confusion.update_layout(width=800, height=800,
                                    title=go.layout.Title(
                                        text='Confusion Matrix',
                                        xref='paper'),
                                    margin={'l': 20, 'r': 20, 'b': 20, 't': 40, 'pad': 4},
                                    xaxis=go.layout.XAxis(
                                        title_text='Output',
                                        tickmode='array',
                                        tickvals=list(range(10)),
                                        ticktext=list(range(10)),
                                        zeroline=False
                                        ),
                                    yaxis=go.layout.YAxis(
                                        title_text='Input',
                                        tickmode='array',
                                        tickvals=list(range(10)),
                                        ticktext=list(range(10)),
                                        zeroline=False
                                        )
                                    )
        return fig_confusion

    def get_weights_XY(self):
        pass

    def get_weights_YY(self):
        pass

    def plot_weights_XY(self, width=800):
        self.weights_XY = self.get_weights_XY()
        fig_weights_XY = go.Figure(data=go.Heatmap(z=self.weights_XY.numpy(), colorscale='YlOrBr'))
        fig_weights_XY.update_layout(width=width, height=800,
                                     title=go.layout.Title(
                                         text='Weights XY',
                                         xref='paper'),
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

    def plot_weights_YY(self, width=800):
        self.weights_YY = self.get_weights_YY()
        fig_weights_YY = go.Figure(data=go.Heatmap(z=self.weights_YY.numpy(), colorscale='YlOrBr'))
        fig_weights_YY.update_layout(width=width, height=800,
                                     title=go.layout.Title(
                                         text='Weights YY',
                                         xref='paper'),
                                     margin={'l': 20, 'r': 20, 'b': 20, 't': 40, 'pad': 4},
                                     xaxis=go.layout.XAxis(
                                         title_text='Neuron Index X',
                                         tickmode='array',
                                         tickvals=np.linspace(0, self.weights_YY.shape[0],
                                                              self.output_shape + 1) +
                                                  self.weights_YY.shape[0] / self.output_shape / 2,
                                         ticktext=np.linspace(0, self.output_shape, self.output_shape + 1),
                                         zeroline=False
                                         ),
                                     yaxis=go.layout.YAxis(
                                         title_text='Neuron Index Y',
                                         tickmode='array',
                                         tickvals=np.linspace(0, self.weights_YY.shape[1],
                                                              self.output_shape + 1) +
                                                  self.weights_YY.shape[1] / self.output_shape / 2,
                                         ticktext=np.linspace(0, self.output_shape, self.output_shape + 1),
                                         zeroline=False
                                         )
                                     )

        return fig_weights_YY

    def plot_spikes_Y(self):
        spikes = self._spikes['Y'].transpose(0, 1)
        width = 800
        height = int(width * spikes.shape[0] / spikes.shape[1])
        fig_spikes = go.Figure(data=go.Heatmap(z=spikes.numpy().astype(int), colorscale='YlOrBr'))
        fig_spikes.update_layout(width=width, height=height,
                                 title=go.layout.Title(
                                     text='Y spikes',
                                     xref='paper',
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
                                     text='Network Spikes',
                                     xref='paper',
                                     )
                                 )

        return fig_spikes

    def feed_class(self, label, top_n=None, k=1, to_print=True, plot=False):
        dataset = MNIST(
            PoissonEncoder(time=self.time_max, dt=self.dt),
            None,
            './/MNIST',
            download=False,
            train=True,
            transform=transforms.Compose([
                transforms.CenterCrop(self.crop),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * self.intensity)
                ])
            )
        self.network.reset_()
        self.network.train(False)
        label_mask = dataset.targets == label
        dataset.data = dataset.data[label_mask]
        dataset.targets = dataset.targets[label_mask]
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=True)

        batch = next(iter(dataloader))

        inpts = {'X': batch['encoded_image'].transpose(0, 1)}
        self.network.run(inpts=inpts, time=self.time_max, input_time_dim=1)
        self._spikes = {
            'X': self.spikes['X'].get('s').view(self.time_max, -1),
            'Y': self.spikes['Y'].get('s').view(self.time_max, -1),
            }

        prediction = self.class_from_spikes(top_n=top_n)
        if to_print:
            print(f'Prediction: {prediction[0:k]}')
        if plot:
            self.plot_spikes().show()
            plot_image(np.flipud(batch['image'][0, 0, :, :].numpy())).show()

        return prediction[0:k]

    def feed_class_lc(self, label, to_print=True, plot=False):
        train_dataset = MNIST(
            PoissonEncoder(time=self.time_max, dt=self.dt),
            None,
            './/MNIST',
            download=False,
            train=True,
            transform=transforms.Compose([
                transforms.CenterCrop(self.crop),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * self.intensity)
                ])
            )
        self.network.reset_()
        self.network.train(False)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=True)

        batch = next(iter(train_dataloader))
        while batch['label'] != label:
            batch = next(iter(train_dataloader))
        else:
            inpts = {'X': batch['encoded_image'].transpose(0, 1)}
            self.network.run(inpts=inpts, time=self.time_max, input_time_dim=1)
            self._spikes = {
                'X': self.spikes['X'].get('s').view(self.time_max, -1),
                'Y': self.spikes['Y'].get('s').view(self.time_max, -1),
                }

        prediction = self.classifier.predict([self._spikes['Y'].sum(0).numpy()])
        if to_print:
            print(f'Prediction: {prediction[0]}')
        if plot:
            self.plot_spikes().show()
            plot_image(np.flipud(batch['image'][0, 0, :, :].numpy())).show()

        return prediction[0]

    def top_voters(self):
        # sum_outputs = self.spikes['Y'].get('s').squeeze(1).sum(0)
        # winner_values = sum_outputs.max(0).values
        # indices_flatten = (sum_outputs == winner_values).flatten().nonzero().squeeze(1)
        # voters = torch.zeros(0, self.kernel_size * self.conv_size)
        # m = 0
        # for i in range(self.conv_size):
        #     row = torch.zeros(self.kernel_size, 0)
        #     for j in range(self.conv_size):
        #         if m < indices_flatten.shape[0]:
        #             row = torch.cat((row, self.network.connections[('X', 'Y')].w[:, indices_flatten[m]]), dim=1)
        #             m += 1
        #     voters = torch.cat((voters, row))

        # return voters
        pass

    def feed_image(self, path, top_n=None, k=1, to_print=True, plot=False):
        self.network.reset_()
        self.network.train(False)
        img = Image.open(fp=path).convert('1')
        transform = transforms.Compose([
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
            'X': self.spikes['X'].get('s').view(self.time_max, -1),
            'Y': self.spikes['Y'].get('s').view(self.time_max, -1),
            }

        prediction = self.class_from_spikes(top_n=top_n)
        if to_print:
            print(f'Prediction: {prediction[0:k]}')
        if plot:
            self.plot_spikes().show()
            plot_image(np.flipud(image.squeeze().numpy())).show()

        return prediction[0:k]

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
        crs.execute('SELECT id FROM networks WHERE id = ?', (self.name,))
        result = crs.fetchone()
        if result:
            print('Rewriting existing network...')
            crs.execute('INSERT INTO networks VALUES (?, ?, ?, ?)', (self.name, self.accuracy, self.n_iter, self.type))
        else:
            crs.execute('INSERT INTO networks VALUES (?, ?, ?, ?)', (self.name, self.accuracy, self.n_iter, self.type))

        conn.commit()
        conn.close()

    # def delete(self, sure=False):
    #     if not sure:
    #         print('Are you sure you want to delete the network? [Y/N]')
    #         if input() == 'Y':
    #             shutil.rmtree(f'networks//{self.name}')
    #             conn = sqlite3.connect(r'networks/networks.db')
    #             crs = conn.cursor()
    #             crs.execute(f'DELETE FROM networks WHERE id = ?', (self.name,))
    #             conn.commit()
    #             conn.close()
    #             print('Network deleted!')
    #         else:
    #             print('Deletion canceled...')
    #     else:
    #         shutil.rmtree(f'networks//{self.name}')
    #         conn = sqlite3.connect(r'networks/networks.db')
    #         crs = conn.cursor()
    #         crs.execute(f'DELETE FROM networks WHERE id = ?', (self.name,))
    #         conn.commit()
    #         conn.close()
    #         print('Network deleted!')

    def __str__(self):
        return f'Network with parameters:\n {self.parameters}'


########################################################################################################################


class LC_SNN(AbstractSNN):
    def __init__(self, mean_weight=0.4, c_w=-100., time_max=250, crop=20,
                 kernel_size=12, n_filters=25, stride=4, intensity=127.5,
                 c_l=False, nu=None, immutable_name=False, foldername=None):

        super().__init__(mean_weight=mean_weight, c_w=c_w, time_max=time_max, crop=crop,
                         kernel_size=kernel_size, n_filters=n_filters, stride=stride, intensity=intensity,
                         c_l=c_l, nu=nu, immutable_name=immutable_name, foldername=foldername,
                         type_='LC_SNN')

    def create_network(self):

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

        self.kernel_prod = self.kernel_size ** 2

        self.norm = self.mean_weight * self.kernel_prod * self.kernel_size ** 2

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
                    for i in range(conv_size):
                        for j in range(conv_size):
                            w[fltr1, i, j, fltr2, i, j] = self.c_w

        # size = self.n_filters * conv_size ** 2
        # sparse_w = torch.sparse.FloatTensor(w.view(size, size).nonzero().t(), w[w != 0].flatten(),
        #                                     (size, size))

        if not self.c_l:
            self.connection_YY = Connection(self.output_layer, self.output_layer, w=w)
        else:
            self.connection_YY = Connection(self.output_layer, self.output_layer, w=w,
                                            update_rule=PostPre,
                                            nu=self.nu,
                                            wmin=self.c_w * 1.2,
                                            wmax=0)

        self.network.add_layer(self.input_layer, name='X')
        self.network.add_layer(self.output_layer, name='Y')
        self.network.add_connection(self.connection_XY, source='X', target='Y')
        self.network.add_connection(self.connection_YY, source='Y', target='Y')
        self.network.add_monitor(self.GlobalMonitor, name='Network')

        self.spikes = {}
        for layer in set(self.network.layers):
            self.spikes[layer] = Monitor(self.network.layers[layer], state_vars=['s'], time=self.time_max)
            self.network.add_monitor(self.spikes[layer], name='%s_spikes' % layer)

        self._spikes = {
            'X': self.spikes['X'].get('s').view(self.time_max, -1),
            'Y': self.spikes['Y'].get('s').view(self.time_max, -1),
            }

        self.voltages = {}
        for layer in set(self.network.layers) - {'X'}:
            self.voltages[layer] = Monitor(self.network.layers[layer], state_vars=['v'], time=self.time_max)
            self.network.add_monitor(self.voltages[layer], name='%s_voltages' % layer)

        self.stride = self.stride
        self.conv_size = conv_size
        self.conv_prod = int(np.prod(conv_size))

        self.weights_XY = self.get_weights_XY()

    def get_weights_XY(self):
        weights_XY = reshape_locally_connected_weights(self.network.connections[('X', 'Y')].w,
                                                       n_filters=self.n_filters,
                                                       kernel_size=self.kernel_size,
                                                       conv_size=self.conv_size,
                                                       locations=self.network.connections[('X', 'Y')].locations,
                                                       input_sqrt=self.n_input)
        return weights_XY

    def get_weights_YY(self):
        shape_YY = self.network.connections[('Y', 'Y')].w.shape
        weights_YY = self.network.connections[('Y', 'Y')].w.view(int(np.sqrt(np.prod(shape_YY))),
                                                                 int(np.sqrt(np.prod(shape_YY))))
        return weights_YY


class C_SNN(AbstractSNN):
    def __init__(self, mean_weight=0.4, c_w=-100., time_max=250, crop=20,
                 kernel_size=12, n_filters=25, stride=4, intensity=127.5,
                 c_l=False, nu=None, immutable_name=False, foldername=None):

        super().__init__(mean_weight=mean_weight, c_w=c_w, time_max=time_max, crop=crop,
                         kernel_size=kernel_size, n_filters=n_filters, stride=stride, intensity=intensity,
                         c_l=c_l, nu=nu,immutable_name=immutable_name, foldername=foldername,
                         type_='C_SNN')

    def create_network(self):
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
        self.output_layer = AdaptiveLIFNodes(
            n=self.n_output,
            shape=(self.n_filters, conv_size, conv_size),
            traces=True,
            thres=thresh,
            trace_tc=tc_trace,
            tc_decay=tc_decay,
            theta_plus=0.05,
            tc_theta_decay=1e6)


        self.norm = self.mean_weight * self.kernel_size ** 2
        self.connection_XY = Conv2dConnection(
            self.input_layer,
            self.output_layer,
            kernel_size=self.kernel_size,
            stride=self.stride,
            update_rule=PostPre,
            norm=self.norm,
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
        size = self.n_filters * conv_size ** 2
        sparse_w = torch.sparse.FloatTensor(w.view(size, size).nonzero().t(), w[w != 0].flatten(),
                                            (size, size))

        if not self.c_l:
            self.connection_YY = Connection(self.output_layer, self.output_layer, w=w)
        else:
            self.connection_YY = Connection(self.output_layer, self.output_layer, w=w,
                                            update_rule=PostPre,
                                            nu=self.nu,
                                            wmin=self.c_w * 1.2,
                                            wmax=0)

        self.network.add_layer(self.input_layer, name='X')
        self.network.add_layer(self.output_layer, name='Y')
        self.network.add_connection(self.connection_XY, source='X', target='Y')
        self.network.add_connection(self.connection_YY, source='Y', target='Y')
        self.network.add_monitor(self.GlobalMonitor, name='Network')

        self.spikes = {}
        for layer in set(self.network.layers):
            self.spikes[layer] = Monitor(self.network.layers[layer], state_vars=['s'], time=self.time_max)
            self.network.add_monitor(self.spikes[layer], name='%s_spikes' % layer)

        self._spikes = {
            'X': self.spikes['X'].get('s').view(self.time_max, -1),
            'Y': self.spikes['Y'].get('s').view(self.time_max, -1),
            }

        self.voltages = {}
        for layer in set(self.network.layers) - {'X'}:
            self.voltages[layer] = Monitor(self.network.layers[layer], state_vars=['v'], time=self.time_max)
            self.network.add_monitor(self.voltages[layer], name='%s_voltages' % layer)

        self.stride = self.stride
        self.conv_size = conv_size
        self.conv_prod = int(np.prod(conv_size))
        self.kernel_prod = int(np.prod(self.kernel_size))
        self.output_shape = int(np.ceil(np.sqrt(self.network.connections[('X', 'Y')].w.size(0))))


        self.weights_XY = self.get_weights_XY()

    def get_weights_XY(self):
        weights = self.network.connections[('X', 'Y')].w
        height = int(weights.size(2))
        width = int(weights.size(3))
        reshaped = torch.zeros(0, width * self.output_shape)
        m = 0
        for i in range(self.output_shape):
            row = torch.zeros(height, 0)
            for j in range(self.output_shape):
                if m < weights.size(0):
                    row = torch.cat((row, weights[m, 0]), dim=1)
                    m += 1
            reshaped = torch.cat((reshaped, row))

        return reshaped

    def get_weights_YY(self):
        shape_YY = self.network.connections[('Y', 'Y')].w.shape
        weights_YY = self.network.connections[('Y', 'Y')].w.view(int(np.sqrt(np.prod(shape_YY))),
                                                                 int(np.sqrt(np.prod(shape_YY))))
        return weights_YY


def plot_image(image):
    width = 400
    height = int(width * image.shape[0] / image.shape[1])

    fig_img = go.Figure(data=go.Heatmap(z=image, colorscale='YlOrBr'))
    fig_img.update_layout(width=width, height=height,
                          title=go.layout.Title(
                              text='Image',
                              xref='paper',
                              x=0
                              )
                          )

    return fig_img




# TODO: gridsearch C_SNN (25 filters)
# TODO: calibration from article
# TODO: gist of competition weights