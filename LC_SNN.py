import torch
import matplotlib.pyplot as plt
from torchvision import transforms

from time import time as t
import datetime
from tqdm import tqdm

from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.network import Network, load
from bindsnet.learning import PostPre, WeightDependentPostPre
from bindsnet.network.monitors import Monitor, NetworkMonitor
from bindsnet.network.nodes import AdaptiveLIFNodes, Input
from bindsnet.network.topology import LocalConnection, Connection
from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_conv2d_weights,
    plot_voltages,
    )

class LC_SNN:
    def __init__(self, norm=0.5, n_train=1):
        self.create_network(norm=norm)

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

            _spikes = {
                "X": self.spikes["X"].get("s").view(self.time_max, -1),
                "Y": self.spikes["Y"].get("s").view(self.time_max, -1),
                }
            _voltages = {"Y": self.voltages["Y"].get("v").view(self.time_max, -1)}

            inpt_axes, inpt_ims = plot_input(
                image, inpt, label=label, axes=inpt_axes, ims=inpt_ims
                )
            spike_ims, spike_axes = plot_spikes(_spikes, ims=spike_ims, axes=spike_axes)
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

    def create_network(self, norm=0.5):
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
                            w[fltr1, i, j, fltr2, i, j] = -100.0

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



    def train(self, n_train=1, n_iter=6000):
        self.network.train(True)
        n_train = n_train
        for epoch in range(n_train):
            train_dataloader = torch.utils.data.DataLoader(
                self.train_dataset, batch_size=1, shuffle=True)

            for i, batch in tqdm(list(zip(range(n_iter), train_dataloader))):
                #inpts = {"X": batch["encoded_image"]}
                inpts = {"X": batch["encoded_image"].transpose(0, 1)}

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
            self.network.reset_()  # Reset state variables

        self.network.train(False);
        #self.network.save(f'network_{str(datetime.datetime.today())}'[:-7].replace(' ', '_').replace(':', '-'))
        return self.network

    def weights_XY(self):
        weights_XY = self.connection_XY.w
        weights_to_display = torch.zeros(0, 28*25)
        i = 0
        while i < 625:
            for j in range(25):
                weights_to_display_row = torch.zeros(28, 0)
                for k in range(25):
                    weights_to_display_row = torch.cat((weights_to_display_row, weights_XY[:, :, i]), dim=1)
                    i += 1
                weights_to_display = torch.cat((weights_to_display, weights_to_display_row), dim=0)
        return weights_to_display.numpy()