import math
import hashlib
import json
import os
from shutil import rmtree
import sqlite3
import random
import time

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import torch
from IPython import display, get_ipython
from PIL import Image
from plotly.subplots import make_subplots
from _plotly_utils.colors.qualitative import Plotly as colors
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from tqdm import tqdm, tqdm_notebook

from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.learning import PostPre
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import AdaptiveLIFNodes, Input
from bindsnet.network.topology import (
    Connection,
    Conv2dConnection,
    LocalConnection,
)
from bindsnet.utils import reshape_locally_connected_weights


def in_ipynb():
    try:
        cfg = get_ipython().config
        parent_appname = str(cfg["IPKernelApp"]["parent_appname"])
        notebook_name = "traitlets.config.loader.LazyConfigValue"
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


########################################################################################################################
######################################  ABSTRACT NETWORK  ##############################################################
########################################################################################################################


class AbstractSNN:
    def __init__(
        self,
        mean_weight=0.26,
        c_w=-100.0,
        time_max=250,
        crop=20,
        kernel_size=12,
        n_filters=25,
        stride=4,
        intensity=127.5,
        dt=1,
        c_l=False,
        A_pos=None,
        A_neg=None,
        tau_pos=8.0,
        tau_neg=20.0,
        weight_decay=None,
        type_="Abstract SNN",
        immutable_name=False,
        foldername=None,
        loaded_from_disk=False,
        c_w_min=None,
        c_w_max=None,
        n_iter=0,
    ):

        """
        Constructor for abstract network.
        :param mean_weight: a value for normalization of XY weights
        :param c_w: competitive YY weight value
        :param time_max: simulation time
        :param crop: the size in pixels MNIST images are cropped to
        :param kernel_size: kernel size for Local Connection
        :param n_filters: number of filter for each patch
        :param stride: stride for Local Connection
        :param intensity: intensity to use in Poisson Distribution to emit X spikes
        :param dt: time step in milliseconds
        :param c_l: To train or not to train YY connections
        :param A_pos: A- parameter of STDP for Y neurons.
        :param A_neg: A+ parameter of STDP for Y neurons.
        :param tau_pos: tau- parameter of STDP for Y neurons.
        :param tau_neg: tau+ parameter of STDP for Y neurons.
        :param type_: Network type
        :param immutable_name: if True, then the network name will be `foldername`. If False, then the name will be
        generated from the network parameters.
        :param foldername: Name to use with `immutable_name` = True
        :param c_w_min: minimum value of competitive YY weights
        :param c_w_max: maximum value of competitive YY weights
        :param n_iter: number of complete training iterations.
        """
        self.n_iter_counter = 0
        self.n_iter = n_iter
        self.network_type = type_
        self.mean_weight = mean_weight
        self.c_w = c_w
        self.c_w_min = c_w_min
        self.c_w_max = c_w_max
        if c_w_min is None:
            self.c_w_min = -np.inf
        if c_w_max is None:
            self.c_w_max = 0
        self.calibrated = False
        self.score = {
            "patch_voting": {"accuracy": None, "error": None, "n_iter": None},
            "all_voting": {"accuracy": None, "error": None, "n_iter": None},
            "spikes_first": {"accuracy": None, "error": None, "n_iter": None},
            "lc": {"accuracy": None, "error": None, "n_iter": None},
        }
        self.classifier = None
        self.conf_matrix = None
        self.time_max = time_max
        self.crop = crop
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.stride = stride
        self.intensity = intensity
        self.dt = dt
        self.c_l = c_l
        self.train_method = None

        self.A_pos = A_pos
        self.A_neg = A_neg
        self.weight_decay = weight_decay

        if not self.c_l:
            self.A_pos = None
            self.A_neg = None
            self.weight_decay = None
        else:
            if self.weight_decay is None:
                self.weight_decay = 0
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.immutable_name = immutable_name
        self.foldername = foldername
        self.loaded_from_disk = loaded_from_disk
        self.mask_YY = None
        self.create_network()
        # for c in self.network.connections:
        #     self.network.connections[c].learning = True

        self.voltages = None  # Can be created with self.record_voltages(True)

        print(
            f"Created {self.network_type} network {self.name} with parameters\n{self.parameters}\n"
        )

    @property
    def parameters(self):
        """
        Combines network parameters in one dict
        :return: network parameters dict
        """
        parameters = {
            "network_type": self.network_type,
            "mean_weight": self.mean_weight,
            "n_iter": self.n_iter,
            "c_w": self.c_w,
            "c_w_min": self.c_w_min,
            "c_w_max": self.c_w_max,
            "time_max": self.time_max,
            "crop": self.crop,
            "kernel_size": self.kernel_size,
            "kernel_prod": self.kernel_prod,
            "stride": self.stride,
            "n_filters": self.n_filters,
            "intensity": self.intensity,
            "dt": self.dt,
            "c_l": self.c_l,
            "A_pos": self.A_pos,
            "A_neg": self.A_neg,
            "tau_pos": self.tau_pos,
            "tau_neg": self.tau_neg,
            "weight_decay": self.weight_decay,
            "train_method": self.train_method,
            "immutable_name": self.immutable_name,
        }
        return parameters

    @property
    def name(self):
        """
        Network name.
        :return: returns network name.
        """
        if self.loaded_from_disk:
            return self.foldername
        else:
            if not self.immutable_name:
                return hashlib.sha224(str(self.parameters).encode("utf8")).hexdigest()
            else:
                return self.foldername

    @property
    def network_state(self):
        """
        Hash of network weights.
        :return: returns hash of network weights.
        """
        hash = hashlib.md5()
        hash.update(self.name.encode('utf8'))
        hash.update(self.get_weights_XY().numpy())
        hash.update(self.get_weights_YY().numpy())
        return hash.hexdigest()

    @property
    def best_voters(self):
        """
        Spiking activity of best neurons for each patch.
        :return: torch.return_types.max - max spiking neurons for each patch
        """
        if self.spikes is not None:
            best_patches_max = (
                self.spikes["Y"]
                .get("s")
                .sum(0)
                .squeeze(0)
                .view(self.n_filters, self.conv_size ** 2)
                .max(0)
            )
            return best_patches_max
        else:
            return None

    @property
    def best_voters_locations(self):
        """
        Get locations of best voters
        :return: locations of best Y neurons
        """
        if self.spikes is not None:
            best_patches_max = self.spikes["Y"].get("s").sum(0).squeeze(0).max(0)
            return best_patches_max.indices
        else:
            return None

    @property
    def accuracy(self):
        best_accuracy = -1
        for method in self.score.keys():
            if method == "lc":
                continue
            if self.score[method]["accuracy"] is not None:
                if self.score[method]["accuracy"] > best_accuracy:
                    best_accuracy = self.score[method]["accuracy"]
        return best_accuracy

    @property
    def error(self):
        best_accuracy = -1
        best_method = None
        for method in self.score.keys():
            if method == "lc":
                continue
            if self.score[method]["accuracy"] is not None:
                if self.score[method]["accuracy"] > best_accuracy:
                    best_accuracy = self.score[method]["accuracy"]
                    best_method = method
        return self.score[best_method]["error"]

    @property
    def n_parameters(self):
        """
        :return: overall weights quantity
        """
        n = 0
        for c in self.network.connections:
            n += np.prod(self.network.connections[c].w.size())
        return n

    def learning(self, learning_XY, learning_YY=None):
        """
        Controls if XY and YY connections are mutable or not.
        :param learning_XY: True - XY is mutable, False - XY is immutable
        :param learning_YY: True - YY is mutable, False - YY is immutable
        """
        if learning_YY is None:
            learning_YY = learning_YY
        self.network.connections[("X", "Y")].learning = learning_XY
        self.network.connections[("Y", "Y")].learning = learning_YY

    def train(
        self, n_iter=None, plot=False, vis_interval=30, shuffle=True, interval=None, download=False
    ):
        """
        The main training function. Simultaneously trains XY and YY connection weights.
        If plot=True will visualize information about the network. Use in Jupyter.
        :param n_iter: number of training iterations
        :param plot: True - to plot info, False - no plotting.
        :param vis_interval: how often to update plots in seconds.
        :param shuffle: if to shuffle training data
        :param interval: interval [n_start, n_stop] of training examples to pass
        """
        if n_iter is None:
            if interval is None:
                raise TypeError("One of a ac and interval must be provided")
        if n_iter is not None:
            if interval is not None:
                raise TypeError("Only one of n_iter and interval must be provided")
        encoded_dataset = MNIST(
            PoissonEncoder(time=self.time_max, dt=self.dt),
            None,
            ".//MNIST",
            download=download,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.CenterCrop(self.crop),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x * self.intensity),
                ]
            ),
        )
        train_dataset = encoded_dataset
        train_dataset.data = encoded_dataset.data[:50000, :, :]
        if n_iter is not None:
            random_choice = torch.randint(0, train_dataset.data.size(0), (n_iter,))
            train_dataset.data = train_dataset.data[random_choice]
        if interval is not None:
            train_dataset.data = train_dataset.data[interval[0] : interval[1]]

        if n_iter is not None:
            count = n_iter
        else:
            count = interval[1] - interval[0]
        self.network.train(True)
        print("Training network...")
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=shuffle
        )
        cnt = 0
        if plot:
            fig_weights_XY = self.plot_weights_XY()
            fig_spikes = self.plot_best_spikes_Y()
            fig_image = plot_image(np.zeros((self.crop, self.crop)))

            display.display(fig_weights_XY)
            display.display(fig_spikes)
            display.display(fig_image)

            fig1, fig2 = self.plot_best_voters()
            display.display(fig1)
            if fig2 is not None:
                display.display(fig2)
            #  Plot random neuron voltage to compare with best neurons
            random_index = random.randint(0, self.n_output - 1)
            while random_index in self.best_voters.indices:
                random_index = random.randint(0, self.n_output - 1)
            random_figure = self.plot_neuron_voltage(random_index)
            random_figure.layout.title.text = f"Random Y neuron voltage"
            if self.voltages:
                display.display(random_figure)

            if self.c_l:
                #  Plot competitive weights distribution
                _, fig_competition_distribtion = self.competition_distribution()
                display.display(fig_competition_distribtion)

        t_start = time.time()
        for speed_counter, batch in tqdm(
            enumerate(train_dataloader), total=count, ncols=ncols
        ):
            t_now = time.time()

            inpts = {"X": batch["encoded_image"].transpose(0, 1)}
            self.network.run(inpts=inpts, time=self.time_max, input_time_dim=1)
            self.n_iter += 1
            if self.mask_YY is not None and self.c_l:
                self.network.connections[("Y", "Y")].w *= self.mask_YY

            if plot:
                if (t_now - t_start) / vis_interval > cnt:
                    #  Plot XY weights
                    self.plot_weights_XY(fig=fig_weights_XY)

                    #  Plot best Y spikes
                    self.plot_best_spikes_Y(fig_spikes)

                    #  Plot input image
                    plot_image(np.flipud(batch["image"][0, 0, :, :].numpy()), fig_image)

                    #  Plot best Y neurons weights and voltages
                    self.plot_best_voters(fig1=fig1, fig2=fig2)

                    #  Plot random neuron voltage to compare with best neurons
                    random_index = random.randint(0, self.n_output - 1)
                    while random_index in self.best_voters.indices:
                        random_index = random.randint(0, self.n_output - 1)
                    self.plot_neuron_voltage(random_index, fig=random_figure)

                    if self.c_l:
                        #  Plot competitive weights distribution
                        self.competition_distribution(fig=fig_competition_distribtion)
                    cnt += 1
            self.network.reset_()

        self.network.reset_()
        self.network.train(False)
        self.train_method = "basic"

    def train_two_steps(self, n_iter=None, plot=False, vis_interval=30):
        """
        Alternative training function. Sequentially trains XY and YY connection weights.
        First trains XY, then locks them, drops YY to zero and trains them.
        If plot=True will visualize information about the network. Use in Jupyter Notebook.
        :param n_iter: number of training iterations
        :param plot: True - to plot info, False - no plotting.
        :param vis_interval: how often to update plots in seconds.
        """
        if n_iter is None:
            n_iter = 5000
        encoded_dataset = MNIST(
            PoissonEncoder(time=self.time_max, dt=self.dt),
            None,
            ".//MNIST",
            download=False,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.CenterCrop(self.crop),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x * self.intensity),
                ]
            ),
        )
        train_dataset = encoded_dataset
        train_dataset.data = encoded_dataset.data[:50000, :, :]
        random_choice = torch.randint(0, train_dataset.data.size(0), (n_iter,))
        train_dataset.data = train_dataset.data[random_choice]

        self.network.train(True)
        self.network.connections[("Y", "Y")].w.fill_(self.c_w)
        print("Training network...")
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=True
        )
        cnt = 0
        if plot:
            fig_image = plot_image(np.zeros((self.crop, self.crop)))
            display.display(fig_image)
            fig_weights_XY = self.plot_weights_XY()
            fig_spikes = self.plot_best_spikes_Y()
            display.display(fig_weights_XY)

            if self.c_l:
                _, fig_competition_distribtion = self.competition_distribution()
                display.display(fig_competition_distribtion)

            display.display(fig_spikes)

            fig1, fig2 = self.plot_best_voters()
            display.display(fig1)
            if fig2 is not None:
                display.display(fig2)
            #  Plot random neuron voltage to compare with best neurons
            random_index = random.randint(0, self.n_output - 1)
            while random_index in self.best_voters.indices:
                random_index = random.randint(0, self.n_output - 1)
            random_figure = self.plot_neuron_voltage(random_index)
            random_figure.layout.title.text = f"Random Y neuron voltage"

        self.network.connections[("Y", "Y")].learning = False
        self.network.connections[("X", "Y")].learning = True
        print("Training XY connection...")
        t_start = time.time()
        for speed_counter, batch in tqdm(
            enumerate(train_dataloader), total=n_iter, ncols=ncols
        ):

            t_now = time.time()

            inpts = {"X": batch["encoded_image"].transpose(0, 1)}
            self.network.run(inpts=inpts, time=self.time_max, input_time_dim=1)

            if plot:
                if (t_now - t_start) / vis_interval > cnt:
                    #  Plot input image
                    plot_image(np.flipud(batch["image"][0, 0, :, :].numpy()), fig_image)
                    #  Plot XY weights
                    self.plot_weights_XY(fig=fig_weights_XY)

                    #  Plot best Y spikes
                    self.plot_best_spikes_Y(fig_spikes)

                    #  Plot best Y neurons weights and voltages
                    self.plot_best_voters(fig1=fig1, fig2=fig2)

                    #  Plot random neuron voltage to compare with best neurons
                    random_index = random.randint(0, self.n_output - 1)
                    while random_index in self.best_voters.indices:
                        random_index = random.randint(0, self.n_output - 1)
                    self.plot_neuron_voltage(random_index, fig=random_figure)

                    cnt += 1

            self.network.reset_()
        self.network.connections[("X", "Y")].learning = False
        if self.c_l:
            self.network.connections[("Y", "Y")].w.fill_(0)
            self.network.connections[("Y", "Y")].learning = True
            print("Training YY connection...")
            t_start = time.time()
            cnt = 0
            for speed_counter, batch in tqdm(
                enumerate(train_dataloader), total=n_iter, ncols=ncols
            ):
                t_now = time.time()

                inpts = {"X": batch["encoded_image"].transpose(0, 1)}
                self.network.run(inpts=inpts, time=self.time_max, input_time_dim=1)
                if self.mask_YY is not None:
                    self.network.connections[("Y", "Y")].w *= self.mask_YY
                self.n_iter += 1
                self.parameters["n_iter"] += 1

                if plot:
                    if (t_now - t_start) / vis_interval > cnt:
                        #  Plot input image
                        plot_image(
                            np.flipud(batch["image"][0, 0, :, :].numpy()), fig_image
                        )
                        #  Plot XY weights
                        self.plot_weights_XY(fig=fig_weights_XY)

                        if self.c_l:
                            #  Plot competitive weights distribution
                            self.competition_distribution(
                                fig=fig_competition_distribtion
                            )

                        #  Plot best Y spikes
                        self.plot_best_spikes_Y(fig_spikes)

                        #  Plot best Y neurons weights and voltages
                        self.plot_best_voters(fig1=fig1, fig2=fig2)

                        #  Plot random neuron voltage to compare with best neurons
                        random_index = random.randint(0, self.n_output - 1)
                        while random_index in self.best_voters.indices:
                            random_index = random.randint(0, self.n_output - 1)
                        self.plot_neuron_voltage(random_index, fig=random_figure)

                        cnt += 1
                self.network.reset_()
            if plot:
                if (t_now - t_start) / vis_interval > cnt:
                    #  Plot input image
                    plot_image(np.flipud(batch["image"][0, 0, :, :].numpy()), fig_image)
                    #  Plot XY weights
                    self.plot_weights_XY(fig=fig_weights_XY)

                    #  Plot best Y spikes
                    self.plot_best_spikes_Y(fig_spikes)

                    #  Plot best Y neurons weights and voltages
                    self.plot_best_voters(fig1=fig1, fig2=fig2)

                    #  Plot random neuron voltage to compare with best neurons
                    random_index = random.randint(0, self.n_output - 1)
                    while random_index in self.best_voters.indices:
                        random_index = random.randint(0, self.n_output - 1)
                    self.plot_neuron_voltage(random_index, fig=random_figure)

                    if self.c_l:
                        #  Plot competitive weights distribution
                        self.competition_distribution(fig=fig_competition_distribtion)
            self.network.connections[("Y", "Y")].learning = False
            # shape = int(np.sqrt(np.prod(self.network.connections[('Y', 'Y')].w.shape)))
            # for i in range(shape):
            #     self.network.connections[('Y', 'Y')].w.view(shape, shape)[i, i] = 0

        self.network.train(False)
        self.train_method = "two_steps"

    def class_from_spikes(self, method="patch_voting", top_n=None, spikes=None):
        """
        Abstract method for getting predicted label from current spikes.
        """
        pass

    def collect_activity_calibration(self, n_iter=None, shuffle=True, download=False):
        """
        Collect network spiking activity on calibartion dataset and save it to disk. Sum of spikes for each neuron are being recorded.
        :param n_iter: number of iterations
        """
        self.network.train(False)
        if n_iter is None:
            n_iter = 10000
        encoded_dataset = MNIST(
            PoissonEncoder(time=self.time_max, dt=self.dt),
            None,
            ".//MNIST",
            download=download,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.CenterCrop(self.crop),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x * self.intensity),
                ]
            ),
        )

        calibratation_dataset = encoded_dataset
        calibratation_dataset.data = encoded_dataset.data[50000:, :, :]
        calibratation_dataset.targets = encoded_dataset.targets[50000:]
        random_choice = torch.randint(0, calibratation_dataset.data.size(0), (n_iter,))
        calibratation_dataset.data = calibratation_dataset.data[random_choice]
        calibratation_dataset.targets = calibratation_dataset.targets[random_choice]

        calibration_dataloader = torch.utils.data.DataLoader(
            calibratation_dataset, batch_size=1, shuffle=shuffle
        )
        labels = []
        outputs = []

        self.network.reset_()
        print("Collecting calibration activity data...")
        for batch in tqdm(calibration_dataloader, ncols=ncols):
            inpts = {"X": batch["encoded_image"].transpose(0, 1)}
            self.network.run(inpts=inpts, time=self.time_max, input_time_dim=1)

            outputs.append(self.spikes["Y"].get("s").sum(0).squeeze(0))
            labels.append(batch["label"].item())

            self.network.reset_()

        data = {"outputs": outputs, "labels": labels}
        if not os.path.exists(f"activity//{self.name}//activity"):
            os.makedirs(f"activity//{self.name}//activity")
        torch.save(
            data, f"activity//{self.name}//activity//{self.network_state}-{n_iter}"
        )

    def collect_activity_test(self, n_iter=None, shuffle=True, download=False):
        """
        Collect network spiking activity on test dataset and save it to disk. Sum of spikes for each neuron are being recorded.
        :param n_iter: number of iterations
        :param shuffle: if to shuffle data
        """
        self.network.train(False)
        if n_iter is None:
            n_iter = 10000

        test_dataset = MNIST(
            PoissonEncoder(time=self.time_max, dt=self.dt),
            None,
            ".//MNIST",
            download=download,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.CenterCrop(self.crop),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x * self.intensity),
                ]
            ),
        )
        random_choice = torch.randint(0, test_dataset.data.size(0), (n_iter,))
        test_dataset.data = test_dataset.data[random_choice]
        test_dataset.targets = test_dataset.targets[random_choice]

        if not self.calibrated:
            raise Exception("The network is not calibrated.")
        self.network.train(False)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=shuffle
        )

        outputs = []
        labels = []
        self.network.reset_()
        print('Collecting test activity data...')
        for batch in tqdm(test_dataloader):
            inpts = {"X": batch["encoded_image"].transpose(0, 1)}
            self.network.run(inpts=inpts, time=self.time_max, input_time_dim=1)
            label = batch["label"].item()
            output = self.spikes["Y"].get("s").sum(0).squeeze(0).float()
            labels.append(label)
            outputs.append(output)
            self.network.reset_()

        data = {"outputs": outputs, "labels": labels}

        if not os.path.exists(f"activity//{self.name}//activity_test"):
            os.makedirs(f"activity//{self.name}//activity_test")
        torch.save(
            data, f"activity//{self.name}//activity_test//{self.network_state}-{n_iter}"
        )

    def calibrate(self, n_iter=None, lc=True, shuffle=True, download=False):
        """
        Calculate network `self.votes` based on spiking activity.
        Each neuron has a vote for each label. The votes are equal to mean amount of spikes.
        If activity was previously recorded and the weights didn't change since then, saved activity will be used.
        :param n_iter: number of iterations
        :param lc: if to calibrate linear classifier too
        """
        print("Calibrating network...")
        if n_iter is None:
            n_iter = 5000

        found_activity = False
        try:
            for name in os.listdir(f"activity//{self.name}//activity/"):
                if self.network_state in name:
                    n_iter_saved = int(name.split("-")[-1])
                    if n_iter <= n_iter_saved:
                        data = torch.load(f"activity//{self.name}//activity//{name}")
                        data_outputs = data["outputs"]
                        data_labels = data["labels"]
                        data_outputs = data_outputs[:n_iter]
                        data_labels = data_labels[:n_iter]
                        data = {"outputs": data_outputs, "labels": data_labels}
                        found_activity = True
                        break
        except FileNotFoundError:
            pass

        if not found_activity:
            self.collect_activity_calibration(n_iter=n_iter, shuffle=shuffle, download=download)
            data = torch.load(
                f"activity//{self.name}//activity//{self.network_state}-{n_iter}"
            )

        print("Calculating votes...")

        outputs = data["outputs"]
        labels = data["labels"]

        votes = torch.zeros([10] + list(outputs[0].shape))
        for (label, output) in zip(labels, outputs):
            votes[label] += output
        for i in range(10):
            votes[i] /= len((np.array(labels) == i).nonzero()[0])

        self.votes = votes
        self.calibrated = True

        if lc:
            self.calibrate_lc(n_iter=n_iter)

    def calibrate_lc(self, n_iter=None):
        """
        Train a linear classifier on network outputs.
        :param n_iter: number of training iterations
        """
        if n_iter is None:
            n_iter = 5000

        found_activity = False
        if not os.path.exists(f"activity//{self.name}//activity/"):
            self.collect_activity_calibration(n_iter)

        for name in os.listdir(f"activity//{self.name}//activity/"):
            if self.network_state in name:
                n_iter_saved = int(name.split("-")[-1])
                if n_iter <= n_iter_saved:
                    data = torch.load(f"activity//{self.name}//activity//{name}")
                    data_outputs = data["outputs"]
                    data_labels = data["labels"]
                    data_outputs = data_outputs[:n_iter]
                    data_labels = data_labels[:n_iter]
                    data = {"outputs": data_outputs, "labels": data_labels}
                    found_activity = True
                    break

        if not found_activity:
            self.collect_activity_calibration(n_iter=n_iter)
            data = torch.load(
                f"activity//{self.name}//activity//{self.network_state}-{n_iter}"
            )

        outputs = [output.flatten().numpy() for output in data["outputs"]]
        labels = data["labels"]

        print("Calibrating classifier...")

        self.classifier = SGDClassifier(n_jobs=-1)
        self.classifier.fit(outputs, labels)

    def votes_distribution(self, fig=None):
        """
        Plots mean votes for top classes.
        :return: plotly.graph_objs.Figure - distribution of votes
        """
        values = self.votes.sort(0, descending=True).values
        y = values.mean(axis=list(range(1, len(self.votes.shape))))
        error_y = values.max(1).values.max(1).values.max(1).values - y
        errorminus_y = y - values.min(1).values.min(1).values.min(1).values
        if fig is None:
            fig = go.Figure(
                go.Scatter(
                    x=list(range(1, 11)),
                    y=y,
                    error_y=dict(
                        type="data",
                        array=error_y,
                        arrayminus=errorminus_y,
                        width=5,
                        visible=True,
                    ),
                    mode="markers",
                    marker_size=10,
                )
            )
            fig.update_layout(
                width=800,
                height=400,
                title_text="Votes Distribution",
                margin={"l": 20, "r": 20, "b": 20, "t": 40, "pad": 4},
                xaxis=go.layout.XAxis(
                    title_text="Top class",
                    tickmode="array",
                    tickvals=list(range(1, 11)),
                    ticktext=list(range(1, 11)),
                    # zeroline=False,
                ),
                yaxis=go.layout.YAxis(title_text="Mean Vote", zeroline=False),
            )
            fig.update_layout(
                title_font_size=17,
                xaxis_title_font_size=16,
                xaxis_tickfont_size=15,
                yaxis_title_font_size=16,
                yaxis_tickfont_size=15,
            )
            fig = go.FigureWidget(fig)
            return fig
        else:
            fig.data[0].y = y
            fig.data[0].error_y = dict(
                type="data",
                array=error_y,
                arrayminus=errorminus_y,
                width=5,
                color="purple",
                visible=True,
            )

    def calculate_accuracy(
        self,
        n_iter=1000,
        top_n=None,
        method="patch_voting",
        to_print=True,
        all=False,
        shuffle=True,
        download=False
    ):
        """
        Calculate network accuracy.
        Everything is stored in `self.score`.
        All network responses are stored in `self.conf_matrix`.
        :param n_iter: number of iterations
        :param top_n: how many labels can each neuron vote for
        :param method: algorithm to determine class from spikes
        :param to_print: if to print the result
        :param all: if to calculate accuracy with all existing methods
        """
        if all:
            for method in self.accuracy_methods:
                self.calculate_accuracy(
                    n_iter=n_iter,
                    method=method,
                    to_print=to_print,
                    all=False,
                    shuffle=True,
                )
            return None

        if not os.path.exists(f"activity//{self.name}//activity"):
            self.collect_activity_calibration(n_iter, download=download)

        found_activity = False
        try:
            for name in os.listdir(f"activity//{self.name}//activity_test/"):
                if self.network_state in name:
                    n_iter_saved = int(name.split("-")[-1])
                    if n_iter <= n_iter_saved:
                        data = torch.load(
                            f"activity//{self.name}//activity_test//{name}"
                        )
                        data_outputs = data["outputs"]
                        data_labels = data["labels"]
                        data_outputs = data_outputs[:n_iter]
                        data_labels = data_labels[:n_iter]
                        data = {"outputs": data_outputs, "labels": data_labels}
                        found_activity = True
                        break
        except FileNotFoundError:
            pass

        if not found_activity:
            self.collect_activity_test(n_iter=n_iter, shuffle=shuffle, download=download)
            data = torch.load(
                f"activity//{self.name}//activity_test//{self.network_state}-{n_iter}"
            )

        data = torch.load(
            f"activity//{self.name}//activity_test//{self.network_state}-{n_iter}"
        )
        predictions = []
        for output in data["outputs"]:
            prediction = self.class_from_spikes(
                top_n=top_n, method=method, spikes=output
            )[0].item()
            predictions.append(prediction)

        labels = np.array(data["labels"])
        predictions = np.array(predictions)

        accuracy = (labels == predictions).mean()
        error = np.sqrt(accuracy * (1 - accuracy) / n_iter)
        if to_print:
            print(f"Accuracy with {method}: {accuracy} with std {round(error, 3)}")

        self.conf_matrix = confusion_matrix(labels, predictions)
        self.score[method]["accuracy"] = accuracy
        self.score[method]["error"] = error
        self.score[method]["n_iter"] = n_iter

    def accuracy_distribution(self, fig=None):
        """
        Get network accuracy distribution across labels.
        :return: pandas.DataFrame with accuracies for each label; plotly.graph_objs.Figure with a plot.
        """
        self.network.train(False)
        colnames = ["label", "accuracy", "error"]
        accs = pd.DataFrame(columns=colnames)
        if self.conf_matrix.shape[0] == 10:
            for i in range(self.conf_matrix.shape[1]):
                true = self.conf_matrix[i, i]
                total = self.conf_matrix[i, :].sum()

                error = np.sqrt(true / total * (1 - true / total) / total)

                accs = accs.append(
                    pd.DataFrame([[i, true / total, error]], columns=colnames),
                    ignore_index=True,
                )

        if self.conf_matrix.shape[1] == 11:
            for i in range(self.conf_matrix.shape[1]):
                true = self.conf_matrix[i, i]
                total = self.conf_matrix[i, :].sum()

                error = np.sqrt(true / total * (1 - true / total) / total)

                accs = accs.append(
                    pd.DataFrame([[i - 1, true / total, error]], columns=colnames),
                    ignore_index=True,
                )

            accs = accs[accs["label"] != -1]

        if fig is None:
            fig = go.Figure(
                go.Scatter(
                    y=accs["accuracy"].values,
                    error_y=dict(array=accs["error"], visible=True, width=5),
                    mode="markers",
                    marker_size=5,
                )
            )
            fig.update_layout(
                width=800,
                height=400,
                title=go.layout.Title(text="Accuracy Distribution", xref="paper"),
                margin={"l": 20, "r": 20, "b": 20, "t": 40, "pad": 4},
                xaxis=go.layout.XAxis(
                    title_text="Class",
                    tickmode="array",
                    tickvals=list(range(10)),
                    ticktext=list(range(10)),
                    # zeroline=False
                ),
                yaxis=go.layout.YAxis(
                    title_text="Accuracy",
                    zeroline=False,
                    range=[0, 1]
                    # tick0=1,
                ),
            )
            fig = go.FigureWidget(fig)
            return accs, fig
        else:
            fig.data[0].y = accs["accuracy"].values
            fig.data[0].error_y = dict(array=accs["error"], visible=True, width=5)

    def competition_distribution(self, fig=None):
        """
        Get network competition weights distribution.
        :return: padnas.DataFrame with competition weights; plotly.graph_objs.Figure with a histogram.
        """
        w = self.network.connections[("Y", "Y")].w
        w_comp = []
        for fltr1 in range(w.size(0)):
            for fltr2 in range(w.size(3)):
                if fltr1 != fltr2:
                    for i in range(w.size(1)):
                        for j in range(w.size(2)):
                            w_comp.append(w[fltr1, i, j, fltr2, i, j])
        w_comp = torch.tensor(w_comp)
        if fig is None:
            fig = go.Figure(go.Histogram(x=w_comp))
            fig.update_layout(
                width=800,
                height=500,
                title=go.layout.Title(
                    text="Competition YY weights histogram", xref="paper"
                ),
                margin={"l": 20, "r": 20, "b": 20, "t": 40, "pad": 4},
                xaxis=go.layout.XAxis(title_text="Weight",),
                yaxis=go.layout.YAxis(title_text="Quantity", zeroline=False,),
            )
            fig = go.FigureWidget(fig)

            return w_comp, fig
        else:
            fig.data[0].x = w_comp

    def accuracy_on_top_n(self, n_iter=1000, labels=False, method="patch_voting"):
        """
        This method might not work at the moment, maybe I am going to remove it.
        Calculate accuracy dependence on top_n.
        :param n_iter: number of iterations
        :param labels: If labels=True the accuracy will be calculated separately for each label.
        :return: accuracies, errors, plotly.graph_objs.Figure - plot.
        """
        self.network.reset_()
        if not self.calibrated:
            print("The network is not calibrated!")
            return None
        self.network.train(False)

        if labels:
            scores = torch.zeros(10, 10, n_iter)
            for label in range(10):
                label_dataset = MNIST(
                    PoissonEncoder(time=self.time_max, dt=self.dt),
                    None,
                    ".//MNIST",
                    download=False,
                    train=False,
                    transform=transforms.Compose(
                        [
                            transforms.CenterCrop(self.crop),
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: x * self.intensity),
                        ]
                    ),
                )
                label_indices = (label_dataset.targets == label).nonzero().flatten()
                label_dataset.data = torch.index_select(
                    label_dataset.data, 0, label_indices
                )
                label_dataset.targets = label_dataset.targets[
                    label_dataset.targets == label
                ]
                random_choice = torch.randint(0, label_dataset.data.size(0), (n_iter,))
                label_dataset.data = label_dataset.data[random_choice]
                label_dataset.targets = label_dataset.targets[random_choice]

                test_dataloader = torch.utils.data.DataLoader(
                    label_dataset, batch_size=1, shuffle=True
                )

                display.clear_output(wait=True)
                print(f"Calculating accuracy for label {label}...")
                test_iter = iter(test_dataloader)
                for i in tqdm(range(n_iter), ncols=ncols):
                    batch = next(test_iter)

                    inpts = {"X": batch["encoded_image"].transpose(0, 1)}
                    self.network.run(inpts=inpts, time=self.time_max, input_time_dim=1)

                    self.network.reset_()

                    for top_n in range(1, 11):
                        prediction = self.class_from_spikes(top_n=top_n, method=method)[
                            0
                        ]
                        if prediction == label:
                            scores[label, top_n - 1, i] = 1

            # errors = (proportion_confint(scores.sum(axis=-1), scores.shape[-1], 0.05)[1] -
            #           proportion_confint(scores.sum(axis=-1), scores.shape[-1], 0.05)[0]) / 2

            errors = (
                (
                    scores.sum(axis=-1)
                    / scores.shape[-1]
                    * (1 - scores.sum(axis=-1) / scores.shape[-1])
                    / scores.shape[-1]
                )
                ** 0.5
            ).numpy()

            fig = go.Figure().update_layout(
                title=go.layout.Title(text="Accuracy dependence on top_n"),
                xaxis=go.layout.XAxis(
                    title_text="top_n",
                    tickmode="array",
                    tickvals=list(range(10)),
                    ticktext=list(range(10)),
                ),
                yaxis=go.layout.YAxis(title_text="Accuracy", range=[0, 1]),
            )

            for label in range(10):
                fig.add_scatter(
                    x=list(range(1, 11)),
                    y=scores.mean(axis=-1)[label, :].numpy(),
                    name=f"label {label}",
                    error_y=dict(array=errors[label, :], visible=True, width=5),
                )
            fig.add_scatter(
                x=list(range(1, 11)),
                y=scores.mean(axis=-1).mean(axis=0).numpy(),
                name=f"Total",
                error_y=dict(array=errors.mean(axis=0), visible=True, width=5),
            )

            return scores, errors, fig

        else:
            test_dataset = MNIST(
                PoissonEncoder(time=self.time_max, dt=self.dt),
                None,
                ".//MNIST",
                download=False,
                train=False,
                transform=transforms.Compose(
                    [
                        transforms.CenterCrop(self.crop),
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: x * self.intensity),
                    ]
                ),
            )
            random_choice = torch.randint(0, test_dataset.data.size(0), (n_iter,))
            test_dataset.data = test_dataset.data[random_choice]
            test_dataset.targets = test_dataset.targets[random_choice]
            test_dataloader = torch.utils.data.DataLoader(
                test_dataset, batch_size=1, shuffle=True
            )
            scores = torch.zeros(n_iter, 10)
            for i, batch in tqdm(enumerate(test_dataloader), ncols=ncols):
                inpts = {"X": batch["encoded_image"].transpose(0, 1)}
                self.network.run(inpts=inpts, time=self.time_max, input_time_dim=1)

                label = batch["label"].item()
                for top_n in range(1, 11):
                    prediction = self.class_from_spikes(top_n=top_n, method=method)[
                        0
                    ].item()
                    if prediction == label:
                        scores[i, top_n - 1] = 1

            res = scores.mean(dim=0)
            errors = ((1 - res) * res / n_iter) ** 0.5
            fig = go.Figure(
                go.Scatter(
                    x=list(range(1, 11)),
                    y=res.numpy(),
                    error_y=dict(array=errors, visible=True, width=5),
                )
            )
            fig.update_layout(
                title=go.layout.Title(text="Accuracy dependence on top_n"),
                xaxis=go.layout.XAxis(
                    title_text="top_n",
                    tickmode="array",
                    tickvals=list(range(1, 11)),
                    ticktext=list(range(1, 11)),
                ),
                yaxis=go.layout.YAxis(title_text="Accuracy", range=[0, 1]),
            )

            return scores, errors, fig

    def confusion(self, fig_confusion=None):
        """
        Plot confusion matrix after accuracy calculation.
        :return: plotly.graph_objs.Figure - confusion matrix heatmap.
        """
        row_sums = self.conf_matrix.sum(axis=1)
        average_confusion_matrix = np.nan_to_num(self.conf_matrix / row_sums)
        if fig_confusion is None:
            fig_confusion = go.Figure(
                data=go.Heatmap(
                    z=average_confusion_matrix, colorscale="YlOrBr", zmin=0, zmax=1
                )
            )
            fig_confusion.update_layout(
                width=800,
                height=800,
                title=go.layout.Title(text="Confusion Matrix", xref="paper"),
                margin={"l": 20, "r": 20, "b": 20, "t": 40, "pad": 4},
                xaxis=go.layout.XAxis(
                    title_text="Output",
                    tickmode="array",
                    tickvals=list(range(10)),
                    ticktext=list(range(10)),
                    zeroline=False,
                ),
                yaxis=go.layout.YAxis(
                    title_text="Input",
                    tickmode="array",
                    tickvals=list(range(10)),
                    ticktext=list(range(10)),
                    zeroline=False,
                ),
            )
            fig_confusion = go.FigureWidget(fig_confusion)
            return fig_confusion
        else:
            fig_confusion.data[0].z = average_confusion_matrix

    def get_weights_XY(self):
        """
        Abstract method to get XY weights in a proper shape to plot them later.
        """
        pass

    def get_weights_YY(self):
        """
        Abstract method to get YY weights in a proper shape to plot them later.
        """
        pass

    def plot_weights_XY(self, fig=None):
        """
        Plots XY weights with a plotly FigureWidget.
        :param fig: widget to update
        :return: plotly.graph_objs.FigureWidget - XY weights heatmap.
        """
        self.weights_XY = self.get_weights_XY()
        if fig is None:
            fig = go.FigureWidget()
            fig.add_heatmap(
                z=self.weights_XY, colorscale="YlOrBr", colorbar_title="Weight"
            )
            fig.update_layout(
                width=800,
                height=750,
                title=go.layout.Title(text="Weights XY", xref="paper"),
                margin={"l": 20, "r": 20, "b": 20, "t": 40, "pad": 4},
                xaxis=go.layout.XAxis(
                    title_text="Neuron Index",
                    tickmode="array",
                    tickvals=np.linspace(
                        0, self.weights_XY.shape[0], self.output_shape + 1
                    )
                    + self.weights_XY.shape[0] / self.output_shape / 2,
                    ticktext=[str(i) for i in range(self.output_shape + 1)],
                    zeroline=False,
                    scaleanchor="x",
                    scaleratio=1,
                ),
                yaxis=go.layout.YAxis(
                    title_text="Neuron Index",
                    tickmode="array",
                    tickvals=np.linspace(
                        0, self.weights_XY.shape[0], self.output_shape + 1
                    )
                    + self.weights_XY.shape[0] / self.output_shape / 2,
                    ticktext=[str(i) for i in range(self.output_shape + 1)],
                    zeroline=False,
                    scaleanchor="x",
                    scaleratio=1,
                ),
            )
            return fig
        else:
            fig.data[0].z = self.weights_XY

    def plot_weights_YY(self, fig_weights_YY=None):
        """
        Plots YY weights.
        :param width: figure width
        :return: plotly.graph_objs.Figure - YY weights heatmap.
        """
        self.weights_YY = self.get_weights_YY()
        if fig_weights_YY is None:
            fig_weights_YY = go.FigureWidget()
            fig_weights_YY.add_heatmap(
                z=self.weights_YY, colorscale="YlOrBr", colorbar_title="Weight"
            )

            fig_weights_YY.update_layout(
                width=800,
                height=750,
                title=go.layout.Title(text="Weights YY", xref="paper"),
                margin={"l": 20, "r": 20, "b": 20, "t": 40, "pad": 4},
                xaxis=go.layout.XAxis(
                    title_text="Neuron Index",
                    tickmode="array",
                    tickvals=np.linspace(
                        0, self.weights_YY.shape[0], self.n_filters + 1
                    )
                    + self.weights_YY.shape[0] / self.n_filters / 2,
                    ticktext=[str(i) for i in range(self.n_filters + 1)],
                    zeroline=False,
                    scaleanchor="x",
                    scaleratio=1,
                    constrain="domain",
                ),
                yaxis=go.layout.YAxis(
                    title_text="Neuron Index",
                    tickmode="array",
                    tickvals=np.linspace(
                        0, self.weights_YY.shape[1], self.n_filters + 1
                    )
                    + self.weights_YY.shape[1] / self.n_filters / 2,
                    ticktext=[str(i) for i in range(self.n_filters + 1)],
                    zeroline=False,
                    scaleanchor="x",
                    scaleratio=1,
                    constrain="domain",
                ),
            )
            return fig_weights_YY
        else:
            fig_weights_YY.data[0].z = self.weights_YY

    def plot_votes(self, fig=None):
        if fig is None:
            fig = go.Figure(
                go.Heatmap(
                    z=self.votes.view(10, -1),
                    colorscale="YlOrBr",
                    colorbar_title="Vote",
                ),
                layout_title_text="Votes",
                layout_xaxis_title_text="Y neuron index",
                layout_yaxis_title_text="Label",
                layout_yaxis_tickvals=list(range(10)),
                layout_yaxis_ticktext=list(range(10)),
            )
            fig = go.FigureWidget(fig)
            return go.FigureWidget(fig)
        else:
            fig.data[0].z = self.votes.view(10, -1)
        return fig

    def plot_spikes_Y(self, fig_spikes=None):
        """
        Plots all Y spikes.
        :return: plotly.graph_objs.Figure - Y spikes heatmap.
        """
        width = 1000
        height = 800
        spikes = (
            self.spikes["Y"]
            .get("s")
            .squeeze(1)
            .view(self.time_max, -1)
            .type(torch.LongTensor)
            .t()
        )
        active_spikes_indices = spikes.sum(1).nonzero().squeeze(1)
        if fig_spikes is None:
            fig_spikes = go.FigureWidget()
            fig_spikes.add_heatmap(
                z=spikes[active_spikes_indices, :], colorscale="YlOrBr"
            )
            tickvals = list(range(spikes.size(0)))
            fig_spikes.update_layout(
                width=width,
                height=height,
                title=go.layout.Title(text="Y neurons spikes", xref="paper",),
                xaxis=go.layout.XAxis(title_text="Time"),
                yaxis=go.layout.YAxis(
                    title_text="Neuron location",
                    tickmode="array",
                    tickvals=list(range(len(active_spikes_indices))),
                    ticktext=active_spikes_indices,
                    zeroline=False,
                ),
                showlegend=False,
            )
            return fig_spikes
        else:
            fig_spikes.data[0].z = spikes[active_spikes_indices, :]

    def plot_best_spikes_Y(self, fig_spikes=None):
        """
        Plots Y spikes only for best neurons in each patch.
        :return: plotly.graph_objs.Figure - best Y spikes heatmap.
        """
        width = 1000
        height = 800
        spikes = self.spikes["Y"].get("s").squeeze(1)
        best_spikes = torch.zeros(self.time_max, self.conv_size ** 2)
        best_indices = []
        best_voters = self.spikes["Y"].get("s").sum(0).squeeze(0).max(0).indices

        for i, row in enumerate(best_voters):
            for j, index in enumerate(row):
                best_spikes[:, i * self.conv_size + j] = spikes[
                    :, best_voters[i][j], i, j
                ]
                best_indices.append(
                    f"Filter {best_voters[i][j].item()},<br>patch ({i}, {j})"
                )
        best_spikes = best_spikes.type(torch.LongTensor).t()
        if fig_spikes is None:
            fig_spikes = go.FigureWidget()
            fig_spikes.add_heatmap(z=best_spikes, colorscale="YlOrBr")
            tickvals = list(range(best_spikes.size(0)))
            fig_spikes.update_layout(
                width=width,
                height=height,
                title=go.layout.Title(text="Best Y neurons spikes", xref="paper",),
                xaxis=go.layout.XAxis(title_text="Time"),
                yaxis=go.layout.YAxis(
                    title_text="Neuron location",
                    tickmode="array",
                    tickvals=tickvals,
                    ticktext=best_indices,
                    zeroline=False,
                ),
                showlegend=False,
            )
            return fig_spikes
        else:
            fig_spikes.data[0].z = best_spikes
            fig_spikes.layout.xaxis.ticktext = best_indices
            fig_spikes.layout.yaxis.ticktext = best_indices

    def plot_best_voters(self, fig1=None, fig2=None):
        """
        Plots information about best Y neurons from current spikes.
        :return: plotly.graph_objs.Figure - weights of best neurons;
        plotly.graph_objs.Figure - voltages of best neurons;
        """
        to_return = False
        if fig1 is None:
            to_return = True
        w = self.network.connections[("X", "Y")].w
        k1, k2 = self.kernel_size, self.kernel_size
        c1, c2 = self.conv_size, self.conv_size
        c1sqrt, c2sqrt = int(math.ceil(math.sqrt(c1))), int(math.ceil(math.sqrt(c2)))
        locations = self.network.connections[("X", "Y")].locations

        best_patches_indices = self.best_voters.indices
        best_patches_values = self.best_voters.values
        subplot_titles = []

        for i, patch_index in enumerate(best_patches_indices):
            total_spikes = best_patches_values.flatten()[i]
            text = f"Total spikes: {total_spikes.item()}"
            subplot_titles.append(
                f"Filter {patch_index}, patch ({i // self.conv_size}, {i % self.conv_size})<br>{text}"
            )
        if fig1 is None:
            fig1 = make_subplots(
                subplot_titles=subplot_titles,
                rows=self.conv_size,
                cols=self.conv_size,
                horizontal_spacing=0.07,
                vertical_spacing=0.1,
            )
            for patch_number, filter_number in enumerate(
                best_patches_indices.flatten()
            ):
                filter_ = w[
                    locations[:, patch_number],
                    filter_number * self.conv_size ** 2
                    + (patch_number // c2sqrt) * c2sqrt
                    + (patch_number % c2sqrt),
                ].view(k1, k2)

                fig1.add_trace(
                    go.Heatmap(
                        z=filter_.flip(0),
                        zmin=0,
                        zmax=1,
                        colorbar_title="Weight",
                        coloraxis="coloraxis",
                    ),
                    row=patch_number // self.conv_size + 1,
                    col=patch_number % self.conv_size + 1,
                )

            fig1.update_layout(
                coloraxis=dict(colorscale="YlOrBr"),
                coloraxis_colorbar_title="Weight",
                height=800,
                width=800,
                title=go.layout.Title(text="Best Y neurons weights", xref="paper", x=0),
            )
            fig1 = go.FigureWidget(fig1)
        else:
            for patch_number, filter_number in enumerate(
                best_patches_indices.flatten()
            ):
                filter_ = w[
                    locations[:, patch_number],
                    filter_number * self.conv_size ** 2
                    + (patch_number // c2sqrt) * c2sqrt
                    + (patch_number % c2sqrt),
                ].view(k1, k2)

                fig1.data[patch_number].z = filter_.flip(0)
                fig1.layout.annotations[patch_number].text = subplot_titles[
                    patch_number
                ]

        if fig2 is None:
            fig2 = None
            if self.voltages is not None:
                fig2 = make_subplots(
                    subplot_titles=subplot_titles,
                    rows=self.conv_size,
                    cols=self.conv_size,
                    horizontal_spacing=0.1,
                    vertical_spacing=0.12,
                )
                for patch_number, filter_number in enumerate(best_patches_indices):
                    voltage = (
                        self.voltages["Y"]
                        .get("v")
                        .squeeze(1)
                        .view(self.time_max, self.n_filters, self.conv_size ** 2)[
                            :, filter_number, patch_number
                        ]
                    )

                    spike_timings = (
                        self.spikes["Y"]
                        .get("s")
                        .squeeze(1)
                        .view(self.time_max, self.n_filters, self.conv_size ** 2)[
                            :, best_patches_indices[patch_number], patch_number
                        ]
                        .nonzero()
                        .squeeze(1)
                    )

                    subplot_voltage = go.Scatter(
                        x=list(range(self.time_max)),
                        y=voltage,
                        line=dict(color=colors[0]),
                        opacity=1,
                        legendgroup="Voltage",
                    )
                    subplot_spikes = go.Scatter(
                        x=spike_timings,
                        y=voltage[spike_timings],
                        mode="markers",
                        marker=dict(color=colors[1]),
                        opacity=1,
                        legendgroup="Spikes",
                    )

                    fig2.add_trace(
                        subplot_voltage,
                        row=patch_number // self.conv_size + 1,
                        col=patch_number % self.conv_size + 1,
                    )
                    fig2.add_trace(
                        subplot_spikes,
                        row=patch_number // self.conv_size + 1,
                        col=patch_number % self.conv_size + 1,
                    )

                for row in range(self.conv_size):
                    for col in range(self.conv_size):
                        fig2.update_xaxes(title_text="Time", row=row + 1, col=col + 1)
                        fig2.update_yaxes(
                            title_text="Voltage", row=row + 1, col=col + 1
                        )
                fig2.update_layout(
                    showlegend=False,
                    title_text="Best Y neurons voltages",
                    height=1000,
                    width=1000,
                )
                fig2 = go.FigureWidget(fig2)
        else:
            if self.voltages is not None:
                for patch_number, filter_number in enumerate(best_patches_indices):
                    voltage = (
                        self.voltages["Y"]
                        .get("v")
                        .squeeze(1)
                        .view(self.time_max, self.n_filters, self.conv_size ** 2)[
                            :, filter_number, patch_number
                        ]
                    )

                    spike_timings = (
                        self.spikes["Y"]
                        .get("s")
                        .squeeze(1)
                        .view(self.time_max, self.n_filters, self.conv_size ** 2)[
                            :, best_patches_indices[patch_number], patch_number
                        ]
                        .nonzero()
                        .squeeze(1)
                    )
                    fig2.data[patch_number * 2].x = list(range(self.time_max))
                    fig2.data[patch_number * 2].y = voltage

                    fig2.data[patch_number * 2 + 1].x = spike_timings
                    fig2.data[patch_number * 2 + 1].y = voltage[spike_timings]
                    fig2.layout.annotations[patch_number].text = subplot_titles[
                        patch_number
                    ]
        if to_return:
            return fig1, fig2

    def plot_neuron_voltage(self, index, location1=None, location2=None, fig=None):
        """
        Plots neuron voltage. If given one index, uses flatten indexing.
        :param index: flat neuron index or filter number
        :param location1: location 1
        :param location2: location 2
        :return: plotly.graph_objs.Figure with voltage
        """
        if fig is None:
            fig = None
            if self.voltages is not None:
                if location1 is None and location2 is None:
                    v = (
                        self.voltages["Y"]
                        .get("v")
                        .squeeze(1)
                        .view(self.time_max, -1)[:, index]
                    )
                    total_spikes = (
                        self.spikes["Y"].get("s").sum(0).squeeze(0).flatten()[index]
                    )
                    text = f"Total spikes: {total_spikes.item()}"
                    spike_timings = (
                        self.spikes["Y"]
                        .get("s")
                        .squeeze(1)
                        .view(self.time_max, -1)[:, index]
                        .nonzero()
                        .squeeze(1)
                    )
                    title_text = f"Neuron {index} voltage<br>{text}"
                else:
                    v = (
                        self.voltages["Y"]
                        .get("v")
                        .squeeze(1)[:, index, location1, location2]
                    )
                    total_spikes = (
                        self.spikes["Y"]
                        .get("s")
                        .sum(0)
                        .squeeze(0)[index, location1, location2]
                    )
                    text = f"Total spikes: {total_spikes.item()}"
                    spike_timings = (
                        self.spikes["Y"]
                        .get("s")
                        .squeeze(1)[:, index, location1, location2]
                        .nonzero()
                        .squeeze(1)
                    )

                    title_text = (
                        f"Filter {index}, ({location1}, {location2}) voltage<br>{text}"
                    )

                subplot_voltage = go.Scatter(
                    x=list(range(self.time_max)),
                    y=v,
                    line=dict(color=colors[0]),
                    mode="lines",
                )
                subplot_spikes = go.Scatter(
                    x=spike_timings,
                    y=[-65 for _ in spike_timings],
                    mode="markers",
                    marker=dict(color=colors[1]),
                )

                fig = go.Figure()
                fig.add_trace(subplot_voltage)
                fig.add_trace(subplot_spikes)

                fig.update_layout(
                    title_text=title_text,
                    showlegend=False,
                    height=400,
                    width=800,
                    xaxis_title_text="Time",
                    yaxis_title_text="Voltage",
                )
            fig = go.FigureWidget(fig)
            return fig
        else:
            if self.voltages is not None:
                if location1 is None and location2 is None:
                    v = (
                        self.voltages["Y"]
                        .get("v")
                        .squeeze(1)
                        .view(self.time_max, -1)[:, index]
                    )
                    total_spikes = (
                        self.spikes["Y"].get("s").sum(0).squeeze(0).flatten()[index]
                    )
                    text = f"Total spikes: {total_spikes.item()}"
                    spike_timings = (
                        self.spikes["Y"]
                        .get("s")
                        .squeeze(1)
                        .view(self.time_max, -1)[:, index]
                        .nonzero()
                        .squeeze(1)
                    )
                    title_text = f"Neuron {index} voltage<br>{text}"
                else:
                    v = (
                        self.voltages["Y"]
                        .get("v")
                        .squeeze(1)[:, index, location1, location2]
                    )
                    total_spikes = (
                        self.spikes["Y"]
                        .get("s")
                        .sum(0)
                        .squeeze(0)[index, location1, location2]
                    )
                    text = f"Total spikes: {total_spikes.item()}"
                    spike_timings = (
                        self.spikes["Y"]
                        .get("s")
                        .squeeze(1)[:, index, location1, location2]
                        .nonzero()
                        .squeeze(1)
                    )

                    title_text = (
                        f"Filter {index}, ({location1}, {location2}) voltage<br>{text}"
                    )

                fig.data[0].x = list(range(self.time_max))
                fig.data[0].y = v

                fig.data[1].x = spike_timings
                fig.data[1].y = v[spike_timings]

                fig.layout.title.text = title_text

    def feed_label(
        self,
        label,
        top_n=None,
        k=1,
        to_print=True,
        plot=False,
        method="patch_voting",
        shuffle=True,
    ):
        """
        Inputs given label into the network, calculates network prediction.
        If plot=True will visualize information about the network. Use in Jupyter Notebook.
        :param label: input label (0 - 9)
        :param top_n: how many labels can each neuron vote for
        :param k: how many possible labels are returned in the prediction
        :param to_print: True - to print the prediction, False - not to print.
        :param plot: True - to plot info, False - no plotting.
        :return: torch.tensor with predictions in descending confidence order.
        """
        dataset = MNIST(
            PoissonEncoder(time=self.time_max, dt=self.dt),
            None,
            ".//MNIST",
            download=False,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.CenterCrop(self.crop),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x * self.intensity),
                ]
            ),
        )
        self.network.reset_()
        self.network.train(False)
        label_mask = dataset.targets == label
        dataset.data = dataset.data[label_mask]
        dataset.targets = dataset.targets[label_mask]
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=shuffle)

        batch = next(iter(dataloader))

        inpts = {"X": batch["encoded_image"].transpose(0, 1)}
        self.network.run(inpts=inpts, time=self.time_max, input_time_dim=1)

        prediction = self.class_from_spikes(top_n=top_n, method=method)
        if to_print:
            print(f"Prediction: {prediction[0:k][0]}")
        if plot:
            plot_image(np.flipud(batch["image"][0, 0, :, :].numpy())).show()
            self.plot_best_spikes_Y().show()

        return prediction[0:k]

    def feed_label_lc(self, label, to_print=True, plot=False):
        train_dataset = MNIST(
            PoissonEncoder(time=self.time_max, dt=self.dt),
            None,
            ".//MNIST",
            download=False,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.CenterCrop(self.crop),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x * self.intensity),
                ]
            ),
        )
        self.network.reset_()
        self.network.train(False)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=True
        )

        batch = next(iter(train_dataloader))
        while batch["label"] != label:
            batch = next(iter(train_dataloader))
        else:
            inpts = {"X": batch["encoded_image"].transpose(0, 1)}
            self.network.run(inpts=inpts, time=self.time_max, input_time_dim=1)

        prediction = self.classifier.predict(
            [
                self.spikes["Y"]
                .get("s")
                .squeeze(1)
                .view(self.time_max, -1)
                .sum(0)
                .numpy()
            ]
        )
        if to_print:
            print(f"Prediction: {prediction[0]}")
        if plot:
            self.plot_best_spikes_Y().show()
            plot_image(np.flipud(batch["image"][0, 0, :, :].numpy())).show()

        return prediction[0]

    def feed_image(self, path, top_n=None, k=1, to_print=True, plot=False):
        """
        Inputs given image into the network, calculates network prediction.
        If plot=True will visualize information about the network. Use in Jupyter Notebook.
        :param path: filepath to input image
        :param top_n: how many labels can each neuron vote for
        :param k: how many possible labels are returned in the prediction
        :param to_print: True - to print the prediction, False - not to print.
        :param plot: True - to plot info, False - no plotting.
        :return: torch.tensor with predictions in descending confidence order.
        """
        self.network.reset_()
        self.network.train(False)
        img = Image.open(fp=path).convert("1")
        transform = transforms.Compose(
            [
                transforms.Resize(size=(self.crop, self.crop)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * self.intensity),
            ]
        )
        image = self.intensity - transform(img)
        pe = PoissonEncoder(time=self.time_max, dt=1)
        encoded_image = pe.enc(
            torch.tensor(np.array(image)).type(torch.FloatTensor),
            time=self.time_max,
            transform=True,
        ).unsqueeze(0)
        inpts = {"X": encoded_image.transpose(0, 1)}
        self.network.run(inpts=inpts, time=self.time_max, input_time_dim=1)

        prediction = self.class_from_spikes(top_n=top_n)
        if to_print:
            print(f"Prediction: {prediction[0:k]}")
        if plot:
            fig, fig2 = self.plot_best_voters()
            i1 = random.randint(0, self.n_filters - 1)
            i2 = random.randint(0, self.conv_size - 1)
            i3 = random.randint(0, self.conv_size - 1)
            while i1 == self.best_voters_locations[i2, i3]:
                i1 = random.randint(0, self.n_filters - 1)
                i2 = random.randint(0, self.conv_size - 1)
                i3 = random.randint(0, self.conv_size - 1)

            fig3 = self.plot_neuron_voltage(i1, i2, i3)
            fig.show()
            if fig2 is not None:
                fig2.show()
            if fig3 is not None:
                fig3.show()
            plot_image(np.flipud(image.squeeze().numpy())).show()

        return prediction[0:k]

    def reset(self, sure=False):
        """
        Reset network weights.
        This will make XY weights random again and competition weights in YY will become `self.c_w`.
        :param sure:
        """
        if not sure:
            print("Are you sure you want to reset the network weights? [Y/N]")
            i = input().lower()
            if i != "y":
                print("Weights clearing canceled.")
                return None

        self.network.connections[("X", "Y")].w.data = torch.rand(
            self.network.connections[("X", "Y")].w.data.shape
        )
        self.network.connections[("Y", "Y")].w.data.masked_fill_(
            self.mask_YY.type(torch.BoolTensor), self.c_w
        )

    def record_voltages(self, to_record=False):
        if to_record:
            self.voltages = {}
            self.voltages["Y"] = Monitor(
                self.network.layers["Y"], state_vars=["v"], time=self.time_max
            )
            self.network.add_monitor(self.voltages["Y"], name="Y_voltages")

        else:
            self.voltages = None

    def report(self, label=None, pdf=True, top_n=10, method="patch_voting"):
        if label is None:
            label = random.randint(0, 9)
        set_true = False
        if self.voltages:
            set_true = True
        self.record_voltages(True)
        dataset = MNIST(
            PoissonEncoder(time=self.time_max, dt=self.dt),
            None,
            ".//MNIST",
            download=False,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.CenterCrop(self.crop),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x * self.intensity),
                ]
            ),
        )
        self.network.reset_()
        self.network.train(False)
        label_mask = dataset.targets == label
        dataset.data = dataset.data[label_mask]
        dataset.targets = dataset.targets[label_mask]
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

        batch = next(iter(dataloader))

        inpts = {"X": batch["encoded_image"].transpose(0, 1)}
        self.network.run(inpts=inpts, time=self.time_max, input_time_dim=1)

        prediction = self.class_from_spikes(top_n=top_n, method=method)

        if not os.path.exists(f"reports//{self.name}"):
            os.makedirs(f"reports//{self.name}")
        print(f"Accuracy: {self.accuracy}")
        # XY weights
        f = self.plot_weights_XY()
        display.display(f)
        f.write_image(f"reports//{self.name}//weights_XY.pdf")

        # STDP and competition distribution
        if self.c_l:
            f = plot_STDP(self.A_pos, self.A_neg, self.tau_pos, self.tau_neg)
            display.display(f)
            f.write_image(f"reports//{self.name}//competitive_STDP.pdf")

            w, f = self.competition_distribution()
            display.display(f)
            f.write_image(f"reports//{self.name}//competitive_weights.pdf")

        # Input image
        f = plot_image(np.flipud(batch["image"][0, 0, :, :].numpy()))
        f.layout.title.text = f"Actual: {label}<br>Predicted: {prediction[0].item()}"
        display.display(f)
        f.write_image(f"reports//{self.name}//input_image.pdf")

        # Y spikes
        f = self.plot_best_spikes_Y()
        display.display(f)
        f.write_image(f"reports//{self.name}//best_Y_spikes.pdf")

        # Best neurons weights and voltages
        f1, f2 = self.plot_best_voters()
        display.display(f1)
        display.display(f2)
        f1.write_image(f"reports//{self.name}//best_voters_weights.pdf")

        f2.write_image(f"reports//{self.name}//best_voters_voltages.pdf")

        # Random neuron voltage
        i1 = random.randint(0, self.n_filters - 1)
        i2 = random.randint(0, self.conv_size - 1)
        i3 = random.randint(0, self.conv_size - 1)
        while i1 == self.best_voters_locations[i2, i3]:
            i1 = random.randint(0, self.n_filters - 1)
            i2 = random.randint(0, self.conv_size - 1)
            i3 = random.randint(0, self.conv_size - 1)

        f = self.plot_neuron_voltage(i1, i2, i3)
        display.display(f)
        f.write_image(f"reports//{self.name}//random_neuron_voltage.pdf")

        self.record_voltages(set_true)

        if pdf:
            # Generate a pdf report
            from pylatex import (
                Document,
                Section,
                Subsection,
                Tabular,
                Math,
                TikZ,
                Axis,
                Plot,
                Figure,
                Matrix,
                Alignat,
            )
            from pylatex.utils import verbatim
            import pathlib

            path = str(pathlib.Path().absolute()) + f"//reports//{self.name}"
            geometry_options = {
                "tmargin": "1.5cm",
                "bmargin": "1.5cm",
                "lmargin": "2cm",
                "rmargin": "1.5cm",
            }
            doc = Document(geometry_options=geometry_options)

            # Write network parameters in the document
            with doc.create(Section("Network parameters")):
                for key in self.parameters.keys():
                    text = f"{key}: {self.parameters[key]}\n"
                    doc.append(text)

            with doc.create(Section("Results")):
                doc.append("Accuracy: " + str(self.accuracy))

            with doc.create(Figure()) as pic:
                pic.add_image(f"{path}//weights_XY.pdf")
                # pic.add_caption('Weights')
            if self.c_l:
                with doc.create(Figure()) as pic:
                    pic.add_image(f"{path}//competitive_STDP.pdf")
                with doc.create(Figure()) as pic:
                    pic.add_image(f"{path}//competitive_weights.pdf")
            with doc.create(Figure()) as pic:
                pic.add_image(f"{path}//input_image.pdf")
            with doc.create(Figure()) as pic:
                pic.add_image(f"{path}//best_Y_spikes.pdf")
            with doc.create(Figure()) as pic:
                pic.add_image(f"{path}//best_voters_weights.pdf")
            with doc.create(Figure()) as pic:
                pic.add_image(f"{path}//best_voters_voltages.pdf")
            with doc.create(Figure()) as pic:
                pic.add_image(f"{path}//random_neuron_voltage.pdf")
            doc.generate_pdf(f"{path}//report", clean=True, clean_tex=True)

    def save(self):
        """
        Save network to disk.
        """
        path = f"networks//{self.name}"
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.network, path + "//network")
        torch.save(self.votes, path + "//votes")
        torch.save(self.conf_matrix, path + "//confusion_matrix")

        with open(path + "//parameters.json", "w") as file:
            json.dump(self.parameters, file)

        with open(path + "//score.json", "w") as file:
            json.dump(self.score, file)

        if not os.path.exists(r"networks/networks.db"):
            conn = sqlite3.connect(r"networks/networks.db")
            crs = conn.cursor()
            crs.execute(
                """CREATE TABLE networks(
                 name BLOB,
                 accuracy REAL,
                 type BLOB
                 )"""
            )
            conn.commit()
            conn.close()

        conn = sqlite3.connect(r"networks/networks.db")
        crs = conn.cursor()
        crs.execute("SELECT name FROM networks WHERE name = ?", (self.name,))
        result = crs.fetchone()
        if result:
            crs.execute("DELETE FROM networks WHERE name = ?", (self.name,))
            crs.execute(
                "INSERT INTO networks VALUES (?, ?, ?)",
                (self.name, self.accuracy, self.network_type),
            )
        else:
            crs.execute(
                "INSERT INTO networks VALUES (?, ?, ?)",
                (self.name, self.accuracy, self.network_type),
            )

        conn.commit()
        conn.close()

    def clear_activity(self, calibration=True, test=True):
        if calibration:
            if os.path.exists(f"activity//{self.name}//activity"):
                rmtree(f"activity//{self.name}//activity")
                print("Cleared calibration activity")
        if test:
            if os.path.exists(f"activity//{self.name}//activity_test"):
                rmtree(f"activity//{self.name}//activity_test")
                print("Cleared test activity")

    def rename(self, name=None):
        old_name = self.name
        if name is None:
            self.immutable_name = False
            self.foldername = None
            self.loaded_from_disk = False
        else:
            self.immutable_name = True
            self.foldername = str(name)

        if old_name != self.name:
            if os.path.exists(f"networks//{old_name}"):
                os.rename(f"networks//{old_name}", f"networks//{self.name}")

            if os.path.exists(f"activity//{old_name}"):
                os.rename(f"activity//{old_name}", f"activity//{self.name}")

        if os.path.exists(f"networks//{old_name}//parameters.json"):
            with open(f"networks//{self.name}//parameters.json", "w") as file:
                json.dump(self.parameters, file)

        if not os.path.exists(r"networks/networks.db"):
            conn = sqlite3.connect(r"networks/networks.db")
            crs = conn.cursor()
            crs.execute(
                """CREATE TABLE networks(
                 name BLOB,
                 accuracy REAL,
                 type BLOB
                 )"""
            )
            conn.commit()
            conn.close()

        conn = sqlite3.connect(r"networks//networks.db")
        crs = conn.cursor()
        crs.execute("SELECT name FROM networks WHERE name = ?", (old_name,))
        result = crs.fetchone()
        if result:
            crs.execute(
                """UPDATE networks
                set name = ?,
                accuracy = ?,
                type = ?
                WHERE name = ?""",
                (self.name, self.accuracy, self.network_type, old_name),
            )
            conn.commit()
            conn.close()
        else:
            crs.execute(
                "INSERT INTO networks VALUES (?, ?, ?)",
                (self.name, self.accuracy, self.network_type),
            )
            conn.commit()
            conn.close()

    def __str__(self):
        return f"Network with parameters:\n {self.parameters}"


########################################################################################################################
######################################  LOCALLY CONNECTED NETWORK  #####################################################
########################################################################################################################


class LC_SNN(AbstractSNN):
    accuracy_methods = ["patch_voting", "all_voting", "spikes_first", "lc"]

    def __init__(
        self,
        mean_weight=0.4,
        c_w=-100.0,
        time_max=250,
        crop=20,
        kernel_size=12,
        n_filters=25,
        stride=4,
        intensity=127.5,
        tau_pos=20.0,
        tau_neg=20.0,
        c_w_min=None,
        c_w_max=None,
        c_l=False,
        A_pos=None,
        A_neg=None,
        weight_decay=None,
        immutable_name=False,
        foldername=None,
        loaded_from_disk=False,
        n_iter=0,
    ):
        """
        Locally Connected network (https://arxiv.org/abs/1904.06269).
        :param mean_weight: a value for normalization of XY weights
        :param c_w: competitive YY weight value
        :param time_max: simulation time
        :param crop: the size in pixels MNIST images are cropped to
        :param kernel_size: kernel size for Local Connection
        :param n_filters: number of filter for each patch
        :param stride: stride for Local Connection
        :param intensity: intensity to use in Poisson Distribution to emit X spikes
        :param tau_pos: tau- parameter of STDP Y neurons.
        :param tau_neg: tau+ parameter of STDP Y neurons.
        :param c_w_min: minimum value of competitive YY weights
        :param c_w_max: maximum value of competitive YY weights
        :param c_l: To train or not to train YY connections
        :param A_pos: A- parameter of STDP for Y neurons.
        :param A_neg: A+ parameter of STDP for Y neurons.
        :param immutable_name: if True, then the network name will be `foldername`. If False, then the name will be
        generated from the network parameters.
        :param foldername: Name to use with `immutable_name` = True
        :param n_iter: number of complete training iterations.
        """
        super().__init__(
            mean_weight=mean_weight,
            c_w=c_w,
            time_max=time_max,
            crop=crop,
            kernel_size=kernel_size,
            n_filters=n_filters,
            stride=stride,
            intensity=intensity,
            c_l=c_l,
            A_pos=A_pos,
            A_neg=A_neg,
            weight_decay=weight_decay,
            tau_pos=tau_pos,
            tau_neg=tau_neg,
            c_w_min=c_w_min,
            c_w_max=c_w_max,
            immutable_name=immutable_name,
            foldername=foldername,
            loaded_from_disk=loaded_from_disk,
            n_iter=n_iter,
            type_="LC_SNN",
        )

    def create_network(self):
        """
        Builds network structure.
        """
        # Hyperparameters
        padding = 0
        conv_size = int((self.crop - self.kernel_size + 2 * padding) / self.stride) + 1
        tc_decay = 20.0
        thresh = -52
        refrac = 2
        self.wmin = 0
        self.wmax = 1

        # Network
        self.network = Network(learning=True)
        self.n_input = self.crop ** 2
        self.input_layer = Input(
            n=self.n_input, shape=(1, self.crop, self.crop), traces=True, refrac=refrac
        )
        self.n_output = self.n_filters * conv_size * conv_size
        self.output_shape = int(np.sqrt(self.n_output))
        self.output_layer = AdaptiveLIFNodes(
            n=self.n_output,
            shape=(self.n_filters, conv_size, conv_size),
            traces=True,
            thres=thresh,
            tc_trace_pre=self.tau_pos,
            tc_trace_post=self.tau_neg,
            tc_decay=tc_decay,
            theta_plus=0.05,
            tc_theta_decay=1e6,
            refrac=refrac,
        )

        self.kernel_prod = self.kernel_size ** 2

        norm = self.mean_weight * self.kernel_prod

        self.connection_XY = LocalConnection(
            self.input_layer,
            self.output_layer,
            n_filters=self.n_filters,
            kernel_size=self.kernel_size,
            stride=self.stride,
            update_rule=PostPre,
            norm=norm,  # 1/(kernel_size ** 2),#0.4 * self.kernel_size ** 2,  # norm constant - check
            nu=[1e-4, 1e-2],
            wmin=self.wmin,
            wmax=self.wmax,
        )

        w = self.connection_XY.w.view(self.input_layer.n, self.output_layer.n)
        w *= norm / self.connection_XY.w.sum(0).view(1, -1)

        if not self.c_l:
            # competitive connections
            w = torch.zeros(
                self.n_filters,
                conv_size,
                conv_size,
                self.n_filters,
                conv_size,
                conv_size,
            )
            mask = torch.zeros(w.shape)
            for fltr1 in range(self.n_filters):
                for fltr2 in range(self.n_filters):
                    if fltr1 != fltr2:
                        for i in range(conv_size):
                            for j in range(conv_size):
                                w[fltr1, i, j, fltr2, i, j] = self.c_w
                                mask[fltr1, i, j, fltr2, i, j] = 1
            self.connection_YY = Connection(self.output_layer, self.output_layer, w=w)
        else:
            w = torch.zeros(
                self.n_filters,
                conv_size,
                conv_size,
                self.n_filters,
                conv_size,
                conv_size,
            )
            mask = torch.zeros(w.shape)
            for fltr1 in range(self.n_filters):
                for fltr2 in range(self.n_filters):
                    if fltr1 != fltr2:
                        for i in range(conv_size):
                            for j in range(conv_size):
                                w[fltr1, i, j, fltr2, i, j] = random.random() * self.c_w
                                mask[fltr1, i, j, fltr2, i, j] = 1
            weight_decay = self.weight_decay
            if self.weight_decay == 0:
                weight_decay = None
            self.connection_YY = Connection(
                self.output_layer,
                self.output_layer,
                w=w,
                update_rule=PostPre,
                nu=[self.A_pos, self.A_neg],
                weight_decay=weight_decay,
                wmin=self.c_w_min,
                wmax=self.c_w_max,
            )
        self.mask_YY = mask

        self.network.add_layer(self.input_layer, name="X")
        self.network.add_layer(self.output_layer, name="Y")
        self.network.add_connection(self.connection_XY, source="X", target="Y")
        self.network.add_connection(self.connection_YY, source="Y", target="Y")

        self.spikes = {}
        self.spikes["Y"] = Monitor(
            self.network.layers["Y"], state_vars=["s"], time=self.time_max
        )
        self.network.add_monitor(self.spikes["Y"], name="Y_spikes")

        self.stride = self.stride
        self.conv_size = conv_size
        self.conv_prod = int(np.prod(conv_size))

        self.weights_XY = self.get_weights_XY()

        self.votes = None

    def class_from_spikes(self, top_n=None, method="patch_voting", spikes=None):
        """
        Predicts label from current Y spikes.
        :param top_n: how many labels can each neuron vote for
        :param method: voting method to use. Options: "patch_voting" and "all_voting"
        :param spikes: if None, use currently recorded spikes. Else use given spikes.
        :return: predicted label
        """
        if top_n == 0:
            raise ValueError("top_n can't be zero")
        if top_n is None:
            top_n = 10

        if spikes is None:
            spikes = self.spikes["Y"].get("s").sum(0).squeeze(0).float()

        if method == "patch_voting":
            res = spikes * self.votes
            res = res.max(1).values.sum(axis=[1, 2])

        elif method == "all_voting":
            res = spikes * self.votes
            res = res.sum(axis=[1, 2, 3])

        elif method == "spikes_first":
            res = torch.zeros(10)
            for i, row in enumerate(spikes.max(0).indices):
                for j, filter_number in enumerate(row):
                    res += (
                        spikes[filter_number, i, j] * self.votes[:, filter_number, i, j]
                    )

        elif method == "lc":
            return self.classifier.predict([spikes.flatten().numpy()])
        else:
            raise NotImplementedError(
                f"This voting method [{method}] is not implemented"
            )

        if res.sum(0).item() == 0:
            self.label = torch.tensor(-1)
            return torch.zeros(10).fill_(-1).type(torch.LongTensor)
        else:
            res = res.argsort(descending=True)
            self.label = res[0]
            return res

    def feed_label(
        self,
        label,
        top_n=None,
        k=1,
        to_print=True,
        plot=False,
        method="patch_voting",
        shuffle=True,
    ):
        """
        Inputs given label into the network, calculates network prediction.
        If plot=True will visualize information about the network. Use in Jupyter Notebook.
        :param label: input label (0 - 9)
        :param top_n: how many labels can each neuron vote for
        :param k: how many possible labels are returned in the prediction
        :param to_print: True - to print the prediction, False - not to print.
        :param plot: True - to plot info, False - no plotting.
        :return: torch.tensor with predictions in descending confidence order.
        """
        super().feed_label(
            label=label,
            top_n=top_n,
            k=k,
            to_print=to_print,
            plot=plot,
            method=method,
            shuffle=shuffle,
        )
        if plot:
            fig, fig2 = self.plot_best_voters()
            i1 = random.randint(0, self.n_filters - 1)
            i2 = random.randint(0, self.conv_size - 1)
            i3 = random.randint(0, self.conv_size - 1)
            while i1 == self.best_voters_locations[i2, i3]:
                i1 = random.randint(0, self.n_filters - 1)
                i2 = random.randint(0, self.conv_size - 1)
                i3 = random.randint(0, self.conv_size - 1)

            fig3 = self.plot_neuron_voltage(i1, i2, i3)
            display.display(fig)
            if fig2 is not None:
                display.display(fig2)
            if fig3 is not None:
                display.display(fig3)

    def get_weights_XY(self):
        """
        Get XY weights in a proper shape to plot them later.
        """
        weights_XY = reshape_locally_connected_weights(
            self.network.connections[("X", "Y")].w,
            n_filters=self.n_filters,
            kernel_size=self.kernel_size,
            conv_size=self.conv_size,
            locations=self.network.connections[("X", "Y")].locations,
            input_sqrt=self.n_input,
        )
        return weights_XY

    def draw_competitions(self, n, max_comp=None):
        """
        Draw competition weights between neurons on n an all other channels
        """
        w = self.network.connections[("Y", "Y")].w

        if max_comp is None:
            max_comp = torch.abs(w).max().item()

        fig = self.plot_weights_XY()
        fig.layout.showlegend = False
        k_ = self.kernel_size

        shape = self.network.connections[("Y", "Y")].w.size(1)
        shape_filters = int(self.n_filters ** 0.5)

        i_ = n % shape_filters
        j_ = n // shape_filters

        for i in range(shape):
            for j in range(shape):
                for k in range(shape_filters):
                    for l in range(shape_filters):
                        value = (
                            w[
                                l % shape_filters + shape_filters * k, j, i, n, j, i
                            ].item()
                            + w[
                                n, j, i, l % shape_filters + shape_filters * k, j, i
                            ].item()
                        ) / 2
                        color = f"RGBA(0,0,255,{0.1 + 0.9 * abs(round(value / max_comp, 2))})"
                        fig.add_scatter(
                            x=[
                                i * k_ * shape_filters + k_ * (i_ + 0.5) - 0.5,
                                i * k_ * shape_filters + k_ * k + k_ / 2 - 0.5,
                            ],
                            y=[
                                j * k_ * shape_filters + k_ * (j_ + 0.5) - 0.5,
                                j * k_ * shape_filters + k_ * l + k_ / 2 - 0.5,
                            ],
                            line=dict(color=color, width=4),
                            mode="lines",
                        )

        return fig

    def get_weights_YY(self):
        """
        Get YY weights in a proper shape to plot them later.
        """
        shape_YY = self.network.connections[("Y", "Y")].w.shape
        weights_YY = self.network.connections[("Y", "Y")].w.view(
            int(np.sqrt(np.prod(shape_YY))), int(np.sqrt(np.prod(shape_YY)))
        )
        return weights_YY

    def plot_spikes_Y(self, fig_spikes=None):
        """
        Plots all Y spikes.
        :return: plotly.graph_objs.Figure - Y spikes heatmap.
        """
        width = 1000
        height = 800
        spikes = (
            self.spikes["Y"]
            .get("s")
            .squeeze(1)
            .view(self.time_max, -1)
            .type(torch.LongTensor)
            .t()
        )
        active_spikes_indices = spikes.sum(1).nonzero().squeeze(1)
        active_spikes_locations = [
            self.index_to_location(index)
            for index in spikes.sum(1).nonzero().squeeze(1)
        ]
        for i, location in enumerate(active_spikes_locations):
            active_spikes_locations[
                i
            ] = f"Filter {location[0]},<br>patch ({location[1], location[2]})"

        if fig_spikes is None:
            fig_spikes = go.FigureWidget()
            fig_spikes.add_heatmap(
                z=spikes[active_spikes_indices, :], colorscale="YlOrBr"
            )
            tickvals = list(range(spikes.size(0)))
            fig_spikes.update_layout(
                width=width,
                height=height,
                title=go.layout.Title(text="Y neurons spikes", xref="paper",),
                xaxis=go.layout.XAxis(title_text="Time"),
                yaxis=go.layout.YAxis(
                    title_text="Neuron location",
                    tickmode="array",
                    tickvals=list(range(len(active_spikes_indices))),
                    ticktext=active_spikes_locations,
                    zeroline=False,
                ),
                showlegend=False,
            )
            return fig_spikes
        else:
            fig_spikes.data[0].z = spikes[active_spikes_indices, :]

    def location_to_index(self, location):
        shape = self.network.connections[("Y", "Y")].w.size(1)
        return location[0] * shape ** 2 + location[1] * shape + location[2]

    def index_to_location(self, index):
        shape = self.network.connections[("Y", "Y")].w.size(1)
        return [
            index // (shape ** 2),
            index % (shape ** 2) // shape,
            index % (shape ** 2) % shape,
        ]


########################################################################################################################
######################################  CONVOLUTION NETWORK  ###########################################################
########################################################################################################################


class C_SNN(AbstractSNN):
    accuracy_methods = ["patch_voting", "all_voting", "lc"]

    def __init__(
        self,
        mean_weight=0.4,
        c_w=-100.0,
        time_max=250,
        crop=20,
        kernel_size=12,
        n_filters=25,
        stride=4,
        intensity=127.5,
        c_l=False,
        A_pos=None,
        A_neg=None,
        tau_pos=20.0,
        tau_neg=20.0,
        weight_decay=None,
        n_iter=0,
        immutable_name=False,
        foldername=None,
        loaded_from_disk=False,
        c_w_min=None,
        c_w_max=None
    ):

        super().__init__(
            mean_weight=mean_weight,
            c_w=c_w,
            time_max=time_max,
            crop=crop,
            kernel_size=kernel_size,
            n_filters=n_filters,
            stride=stride,
            intensity=intensity,
            c_l=c_l,
            A_pos=A_pos,
            A_neg=A_neg,
            tau_pos=tau_pos,
            tau_neg=tau_neg,
            weight_decay=weight_decay,
            c_w_min=c_w_min,
            c_w_max=c_w_max,
            immutable_name=immutable_name,
            foldername=foldername,
            loaded_from_disk=loaded_from_disk,
            n_iter=n_iter,
            type_="C_SNN",
        )

    def create_network(self):
        # Hyperparameters
        padding = 0
        conv_size = int((self.crop - self.kernel_size + 2 * padding) / self.stride) + 1
        tc_decay = 20.0
        thresh = -52
        refrac = 2
        self.wmin = 0
        self.wmax = 1

        # Network
        self.network = Network(learning=True)
        self.n_input = self.crop ** 2
        self.input_layer = Input(
            n=self.n_input, shape=(1, self.crop, self.crop), traces=True, refrac=refrac
        )
        self.n_output = self.n_filters * conv_size * conv_size
        self.output_layer = AdaptiveLIFNodes(
            n=self.n_output,
            shape=(self.n_filters, conv_size, conv_size),
            traces=True,
            thres=thresh,
            tc_trace_pre=self.tau_pos,
            tc_trace_post=self.tau_neg,
            tc_decay=tc_decay,
            theta_plus=0.05,
            tc_theta_decay=1e6,
        )

        norm = self.mean_weight * self.kernel_size ** 2
        self.connection_XY = Conv2dConnection(
            self.input_layer,
            self.output_layer,
            kernel_size=self.kernel_size,
            stride=self.stride,
            update_rule=PostPre,
            norm=norm,
            nu=[1e-4, 1e-2],
            wmin=self.wmin,
            wmax=self.wmax,
        )

        # w = self.connection_XY.w.view(self.input_layer.n, self.output_layer.n)
        # w *= norm / self.connection_XY.w.sum(0).view(1, -1)

        # competitive connections
        w = torch.zeros(
            self.n_filters, conv_size, conv_size, self.n_filters, conv_size, conv_size
        )
        for fltr1 in range(self.n_filters):
            for fltr2 in range(self.n_filters):
                if fltr1 != fltr2:
                    # change
                    for i in range(conv_size):
                        for j in range(conv_size):
                            w[fltr1, i, j, fltr2, i, j] = self.c_w
        size = self.n_filters * conv_size ** 2

        if not self.c_l:
            self.connection_YY = Connection(self.output_layer, self.output_layer, w=w)
        else:
            weight_decay = self.weight_decay
            if self.weight_decay is None:
                weight_decay = 0
                self.weight_decay = 0
            self.connection_YY = Connection(
                self.output_layer,
                self.output_layer,
                w=w,
                update_rule=PostPre,
                nu=[self.A_pos, self.A_neg],
                weight_decay=weight_decay,
                wmin=self.c_w_min,
                wmax=self.c_w_max,
            )

        self.network.add_layer(self.input_layer, name="X")
        self.network.add_layer(self.output_layer, name="Y")
        self.network.add_connection(self.connection_XY, source="X", target="Y")
        self.network.add_connection(self.connection_YY, source="Y", target="Y")

        self.spikes = {}
        self.spikes["Y"] = Monitor(
            self.network.layers["Y"], state_vars=["s"], time=self.time_max
        )
        self.network.add_monitor(self.spikes["Y"], name="Y_spikes")

        self.stride = self.stride
        self.conv_size = conv_size
        self.conv_prod = int(np.prod(conv_size))
        self.kernel_prod = int(np.prod(self.kernel_size))
        self.output_shape = int(
            np.ceil(np.sqrt(self.network.connections[("X", "Y")].w.size(0)))
        )

        self.weights_XY = self.get_weights_XY()

    def class_from_spikes(self, top_n=None, method=None, spikes=None):
        if top_n == 0:
            raise ValueError("top_n can't be zero")
        if top_n is None:
            top_n = 10

        if spikes is None:
            spikes = self.spikes["Y"].get("s").sum(0).squeeze(0).float()

        if method == "patch_voting":
            res = spikes * self.votes
            res = res.max(1).values.sum(axis=[1, 2])

        elif method == "all_voting":
            res = spikes * self.votes.view([10] + list(spikes.shape))
            res = res.sum(axis=[1, 2, 3])

        elif method == "lc":
            return self.classifier.predict([spikes.flatten().numpy()])

        else:
            raise NotImplementedError(
                f"This voting method [{method}] is not implemented"
            )

        if res.sum(0).item() == 0:
            self.label = torch.tensor(-1)
            return torch.zeros(10).fill_(-1).type(torch.LongTensor)
        else:
            res = res.argsort(descending=True)
            self.label = res[0]
            return res

    def get_weights_XY(self):
        """
        Get XY weights in a proper shape to plot them later.
        """
        weights = self.network.connections[("X", "Y")].w
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
        """
        Get YY weights in a proper shape to plot them later.
        """
        shape_YY = self.network.connections[("Y", "Y")].w.shape
        weights_YY = self.network.connections[("Y", "Y")].w.view(
            int(np.sqrt(np.prod(shape_YY))), int(np.sqrt(np.prod(shape_YY)))
        )
        return weights_YY

    def plot_best_voters(self, fig1=None, fig2=None):
        return None, None


########################################################################################################################
######################################  FULLY CONNECTED NETWORK  #######################################################
########################################################################################################################


class FC_SNN(AbstractSNN):
    accuracy_methods = ["patch_voting", "all_voting", "lc"]

    def __init__(
        self,
        mean_weight=0.4,
        c_w=-100.0,
        time_max=250,
        crop=20,
        n_filters=25,
        intensity=127.5,
        tau_pos=20.0,
        tau_neg=20.0,
        n_iter=0,
        c_l=False,
        A_pos=None,
        A_neg=None,
        weight_decay=None,
        immutable_name=False,
        foldername=None,
        loaded_from_disk=False,
        c_w_min=None,
        c_w_max=None
    ):

        super().__init__(
            mean_weight=mean_weight,
            c_w=c_w,
            time_max=time_max,
            crop=crop,
            n_filters=n_filters,
            intensity=intensity,
            tau_pos=tau_pos,
            tau_neg=tau_neg,
            c_l=c_l,
            A_pos=A_pos,
            A_neg=A_neg,
            weight_decay=weight_decay,
            immutable_name=immutable_name,
            foldername=foldername,
            loaded_from_disk=loaded_from_disk,
            n_iter=n_iter,
            c_w_min=c_w_min,
            c_w_max=c_w_max,
            type_="FC_SNN",
        )

    def create_network(self):
        self.kernel_size = self.crop
        conv_size = 1

        # Hyperparameters
        tc_decay = 20.0
        thresh = -52
        refrac = 2
        self.wmin = 0
        self.wmax = 1

        # Network
        self.network = Network(learning=True)
        self.n_input = self.crop ** 2
        self.input_layer = Input(
            n=self.n_input, shape=(1, self.crop, self.crop), traces=True, refrac=refrac
        )
        self.n_output = self.n_filters
        self.output_shape = int(np.sqrt(self.n_output))
        self.output_layer = AdaptiveLIFNodes(
            n=self.n_output,
            shape=(self.n_output,),
            traces=True,
            thres=thresh,
            tc_trace_pre=self.tau_pos,
            tc_trace_post=self.tau_neg,
            tc_decay=tc_decay,
            theta_plus=0.05,
            tc_theta_decay=1e6,
        )

        self.kernel_prod = self.kernel_size ** 2

        norm = self.mean_weight * self.kernel_prod

        self.connection_XY = LocalConnection(
            self.input_layer,
            self.output_layer,
            n_filters=self.n_filters,
            kernel_size=self.kernel_size,
            stride=self.stride,
            update_rule=PostPre,
            norm=norm,  # 1/(kernel_size ** 2),#0.4 * self.kernel_size ** 2,  # norm constant - check
            nu=[1e-4, 1e-2],
            wmin=self.wmin,
            wmax=self.wmax,
        )

        w = self.connection_XY.w.view(self.input_layer.n, self.output_layer.n)
        w *= norm / self.connection_XY.w.sum(0).view(1, -1)

        # competitive connections
        w = torch.zeros(self.n_filters, self.n_filters)
        for fltr1 in range(self.n_filters):
            for fltr2 in range(self.n_filters):
                if fltr1 != fltr2:
                    w[fltr1, fltr2] = self.c_w

        # size = self.n_filters * conv_size ** 2
        # sparse_w = torch.sparse.FloatTensor(w.view(size, size).nonzero().t(), w[w != 0].flatten(),
        #                                     (size, size))

        if not self.c_l:
            self.connection_YY = Connection(self.output_layer, self.output_layer, w=w)
        else:
            weight_decay = self.weight_decay
            if self.weight_decay == 0:
                weight_decay = None
            self.connection_YY = Connection(
                self.output_layer,
                self.output_layer,
                w=w,
                update_rule=PostPre,
                nu=[self.A_pos, self.A_neg],
                weight_decay=self.weight_decay,
                wmin=self.c_w_min,
                wmax=self.c_w_max,
            )

        self.network.add_layer(self.input_layer, name="X")
        self.network.add_layer(self.output_layer, name="Y")
        self.network.add_connection(self.connection_XY, source="X", target="Y")
        self.network.add_connection(self.connection_YY, source="Y", target="Y")

        self.spikes = {}
        self.spikes["Y"] = Monitor(
            self.network.layers["Y"], state_vars=["s"], time=self.time_max
        )
        self.network.add_monitor(self.spikes["Y"], name="Y_spikes")

        self.stride = self.stride
        self.conv_size = 1
        self.conv_prod = int(np.prod(conv_size))

        self.weights_XY = self.get_weights_XY()

    def class_from_spikes(self, top_n=None, method=None, spikes=None):
        if top_n == 0:
            raise ValueError("top_n can't be zero")
        if top_n is None:
            top_n = 10

        if spikes is None:
            spikes = self.spikes["Y"].get("s").sum(0).squeeze(0).float()

        if method == "patch_voting":
            res = spikes * self.votes
            res = res.max(1).values

        elif method == "all_voting":
            res = spikes * self.votes.view([10] + list(spikes.shape))
            res = res.sum(axis=1)

        elif method == "lc":
            return self.classifier.predict([spikes.flatten().numpy()])

        else:
            raise NotImplementedError(
                f"This voting method [{method}] is not implemented"
                )

        if res.sum(0).item() == 0:
            self.label = torch.tensor(-1)
            return torch.zeros(10).fill_(-1).type(torch.LongTensor)
        else:
            res = res.argsort(descending=True)
            self.label = res[0]
            return res

    def plot_best_voters(self, fig1=None, fig2=None):
        """
        Plots information about best Y neurons from current spikes.
        :return: plotly.graph_objs.Figure - weights of best neurons;
        plotly.graph_objs.Figure - voltages of best neurons;
        """
        to_return = False
        if fig1 is None:
            to_return = True
        best_filter_index = self.best_voters.indices
        best_patch_value = self.best_voters.values

        text = f"Total spikes: {best_patch_value.item()}"
        title = f"Filter {best_filter_index.item()}<br>{text}"

        filter_ = (
            self.network.connections[("X", "Y")]
            .w[:, best_filter_index]
            .view(self.kernel_size, self.kernel_size)
        )

        if fig1 is None:
            fig1 = go.Figure(
                go.Heatmap(z=filter_.flip(0), zmin=0, zmax=1, colorscale="YlOrBr",),
            )

            fig1.update_layout(
                height=800,
                width=800,
                title=go.layout.Title(text=title, xref="paper", x=0),
            )
            fig1 = go.FigureWidget(fig1)
        else:
            fig1.data[0].z = filter_.flip(0)
        if self.voltages is not None:
            voltage = (
                self.voltages["Y"]
                .get("v")
                .squeeze(1)
                .view(self.time_max, self.n_filters, self.conv_size ** 2)[
                    :, best_filter_index, 0
                ]
            ).squeeze(1)

            spike_timings = (
                self.spikes["Y"]
                .get("s")
                .squeeze(1)[:, self.best_voters.indices]
                .nonzero()[:, 0]
            )

            voltage_plot = go.Scatter(
                x=list(range(self.time_max)),
                y=voltage,
                line=dict(color=colors[0]),
                opacity=1,
            )
            spike_plot = go.Scatter(
                x=spike_timings,
                y=voltage[spike_timings],
                mode="markers",
                marker=dict(color=colors[1]),
                opacity=1,
            )

            if fig2 is None:
                fig2 = go.Figure()
                fig2.add_trace(voltage_plot,)
                fig2.add_trace(spike_plot,)

                fig2.update_xaxes(title_text="Time")
                fig2.update_yaxes(title_text="Voltage")

                fig2.update_layout(
                    title_text=title, showlegend=False, height=1000, width=1000,
                )
                fig2 = go.FigureWidget(fig2)
            else:
                fig2.data[0].x = list(range(self.time_max))
                fig2.data[0].y = voltage

                fig2.data[1].x = spike_timings
                fig2.data[1].y = voltage[spike_timings]

                fig2.layout.title.text = title
        else:
            fig2 = None
        if to_return:
            return fig1, fig2

    def plot_best_spikes_Y(self, fig_spikes=None):
        """
        Plots Y spikes only for best neurons in each patch.
        :return: plotly.graph_objs.Figure - best Y spikes heatmap.
        """
        width = 1000
        height = 800
        best_spikes = (
            self.spikes["Y"]
            .get("s")
            .squeeze(1)[:, self.best_voters.indices]
            .type(torch.LongTensor)
            .t()
        )
        if fig_spikes is None:
            fig_spikes = go.Figure(data=go.Heatmap(z=best_spikes, colorscale="YlOrBr"))
            tickvals = list(range(best_spikes.size(0)))
            fig_spikes.update_layout(
                width=width,
                height=height,
                title=go.layout.Title(text="Best Y neurons spikes", xref="paper",),
                xaxis=go.layout.XAxis(title_text="Time"),
                yaxis=go.layout.YAxis(
                    title_text="Neuron location",
                    tickmode="array",
                    tickvals=tickvals,
                    ticktext=[f"Filter {self.best_voters.indices.item()}"],
                    zeroline=False,
                ),
                showlegend=False,
            )
            fig_spikes = go.FigureWidget(fig_spikes)
            return fig_spikes
        else:
            fig_spikes.data[0].z = best_spikes

    def feed_label(
        self, label, top_n=None, k=1, to_print=True, plot=False, shuffle=True
    ):
        super().feed_label(
            label=label, top_n=top_n, k=k, to_print=to_print, plot=plot, shuffle=shuffle
        )
        if plot:
            fig1, fig2 = self.plot_best_voters()
            fig1.show()
            if fig2 is not None:
                fig2.show()

    def get_weights_XY(self):
        """
        Get XY weights in a proper shape to plot them later.
        """
        weights = self.network.connections[("X", "Y")].w.view(
            self.crop, self.crop, self.n_filters
        )
        height = int(weights.size(0))
        width = int(weights.size(1))
        reshaped = torch.zeros(0, width * self.output_shape)
        m = 0
        for i in range(self.output_shape):
            row = torch.zeros(height, 0)
            for j in range(self.output_shape):
                if m < weights.size(-1):
                    row = torch.cat((row, weights[:, :, m]), dim=1)
                    m += 1
            reshaped = torch.cat((reshaped, row))

        return reshaped.flip(0)

    def get_weights_YY(self):
        """
        Get YY weights in a proper shape to plot them later.
        """
        shape_YY = self.network.connections[("Y", "Y")].w.shape
        weights_YY = self.network.connections[("Y", "Y")].w.view(
            int(np.sqrt(np.prod(shape_YY))), int(np.sqrt(np.prod(shape_YY)))
        )
        return weights_YY

    def competition_distribution(self, fig=None):
        w = self.network.connections[("Y", "Y")].w
        w_comp = []
        for fltr1 in range(w.size(0)):
            for fltr2 in range(w.size(1)):
                if fltr1 != fltr2:
                    w_comp.append(w[fltr1, fltr2])
        w_comp = torch.tensor(w_comp)
        if fig is None:
            fig = go.Figure(go.Histogram(x=w_comp))
            fig.update_layout(
                width=800,
                height=500,
                title=go.layout.Title(
                    text="Competition weights histogram", xref="paper"
                ),
                margin={"l": 20, "r": 20, "b": 20, "t": 40, "pad": 4},
                xaxis=go.layout.XAxis(title_text="Weight",),
                yaxis=go.layout.YAxis(title_text="N", zeroline=False,),
            )
            fig = go.FigureWidget(fig)
            return w_comp, fig
        else:
            fig.data[0].x = w_comp

    @property
    def parameters(self):
        parameters = {
            "network_type": self.network_type,
            "mean_weight": self.mean_weight,
            "n_iter": self.n_iter,
            "c_w": self.c_w,
            "c_w_min": self.c_w_min,
            "c_w_max": self.c_w_max,
            "time_max": self.time_max,
            "crop": self.crop,
            "n_filters": self.n_filters,
            "intensity": self.intensity,
            "dt": self.dt,
            "c_l": self.c_l,
            "A_pos": self.A_pos,
            "A_neg": self.A_neg,
            "tau_pos": self.tau_pos,
            "tau_neg": self.tau_neg,
            "weight_decay": self.weight_decay,
            "immutable_name": self.immutable_name,
        }
        return parameters


def best_avg_spikes(tau_pos, tau_neg, A_pos, A_neg):
    return (
        1
        / (tau_pos * tau_neg)
        * (A_neg * tau_neg - A_pos * tau_pos)
        / (A_pos - A_neg)
        * 250
    )


def plot_image(image, fig=None):
    width = 400
    height = int(width * image.shape[0] / image.shape[1])
    if fig is None:
        fig = go.Figure(data=go.Heatmap(z=image, colorscale="YlOrBr"))
        fig.update_layout(
            width=width,
            height=height,
            title=go.layout.Title(text="Image", xref="paper", x=0),
            xaxis=dict(scaleanchor="x", scaleratio=1, constrain="domain"),
            yaxis=dict(scaleanchor="x", scaleratio=1, constrain="domain"),
        )
        fig = go.FigureWidget(fig)
        return fig
    else:
        fig.data[0].z = image


def plot_STDP(A_pos, A_neg, tau_pos, tau_neg):
    layout = go.Layout(
        height=500,
        width=800,
        xaxis=dict(zeroline=True, zerolinecolor="#002c75"),
        yaxis=dict(zeroline=True, zerolinecolor="#002c75"),
        legend=dict(
            font=dict(size=12, color="black"),
            bgcolor="rgba(25,211,243,0.5)",
            bordercolor="Black",
            borderwidth=1,
        ),
    )
    t_neg = np.linspace(-100, 0, 100)
    t_pos = np.linspace(0, 100, 100)
    dw_neg = -A_neg * np.exp(t_neg / tau_neg)
    dw_pos = A_pos * np.exp(-t_pos / tau_pos)

    fig = go.Figure(layout=layout)
    fig.add_scatter(
        x=t_neg,
        y=dw_neg,
        line=dict(color=colors[0]),
        name="Negative update",
        line_shape="spline",
    )
    fig.add_scatter(
        x=t_pos,
        y=dw_pos,
        line=dict(color=colors[1]),
        name="Positive update",
        line_shape="spline",
    )
    fig.layout.title.text = "STDP"
    fig.layout.xaxis.title.text = "$t_{post} - t_{pre}, 10^{-3} s$"
    fig.layout.yaxis.title.text = "$\Delta w$"

    fig.layout.legend.y = 1
    fig.layout.legend.x = 0
    fig.layout.margin.t = 50
    fig.layout.margin.b = 20
    fig.layout.margin.r = 20

    fig = go.FigureWidget(fig)

    return fig
