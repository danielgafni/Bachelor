import torch
import matplotlib.pyplot as plt
from torchvision import transforms

from time import time as t
import datetime

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
    plot_spikes_display
    )

from IPython.display import clear_output

import streamlit as st

########################################################################################################################

st.title('Hahahhahahahahhahahaha')


time_max = 30
dt = 1
intensity = 127.5

with st.spinner('Loading dataset...'):
    train_dataset = MNIST(
        PoissonEncoder(time=time_max, dt=dt),
        None,
        "MNIST",
        download=False,
        train=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
            )
        )

st.write('Dataset loaded.')


@st.cache(ignore_hash=True)
def get_network(norm):
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

    network = Network(learning=True)
    GlobalMonitor = NetworkMonitor(network, state_vars=('v', 's', 'w'))

    input_layer = Input(n=784, shape=(1, 28, 28), traces=True)

    output_layer = AdaptiveLIFNodes(
        n=n_filters * conv_size * conv_size,
        shape=(n_filters, conv_size, conv_size),
        traces=True,
        thres=thresh,
        trace_tc=tc_trace,
        tc_decay=tc_decay,
        theta_plus=0.05,
        tc_theta_decay=1e6)


    connection_XY = LocalConnection(
        input_layer,
        output_layer,
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

    connection_YY = Connection(output_layer, output_layer, w=w)

    network.add_layer(input_layer, name='X')
    network.add_layer(output_layer, name='Y')

    network.add_connection(connection_XY, source='X', target='Y')
    network.add_connection(connection_YY, source='Y', target='Y')

    network.add_monitor(GlobalMonitor, name='Network')

    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=time_max)
        network.add_monitor(spikes[layer], name="%s_spikes" % layer)

    voltages = {}
    for layer in set(network.layers) - {"X"}:
        voltages[layer] = Monitor(network.layers[layer], state_vars=["v"], time=time_max)
        network.add_monitor(voltages[layer], name="%s_voltages" % layer)

    return network


st.sidebar.markdown('Hyperparameters')
norm = st.sidebar.slider('norm', 0.001, 0.5, 0.2315)
with st.spinner('Building network...'):
    network = get_network(norm=norm)
st.write('LC-SNN ready!')


########################################################################################################################

st.sidebar.markdown('Training parameters')
visualize = st.sidebar.checkbox('Visualize')
n_train = st.sidebar.slider('Number of passes through the training set', 1, 10, 1)
n_iter = int(st.sidebar.text_input('Number of images to feed the network with', '10'))
vis_interval = st.sidebar.slider('Time between visualisations, s', 1, 60, 10)


def train(n_train=1, n_iter=10, vis_interval=10):
    progress_bar = st.progress(0)
    vis_counter = 0
    t_start = t()

    for epoch in range(n_train):
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=True)

        #t_now = t_start + 0.5
        with st.spinner(f'Training...'):
            status_text = st.empty()
            weights_XY_plot = st.empty()
            weights_to_display_plot2 = st.empty()
            for i, batch in list(zip(range(n_iter), train_dataloader)):
                t_now = t()
                status_text.text(f'Time left: {round((n_iter - i - 1)/((i+1)/(t_now - t_start))/60, 1)}m, '
                                 f'speed: {round((i+1)/(t_now - t_start), 1)}it/s\n'
                                 f'Iteration: {i+1}/{n_iter}')
                progress_bar.progress(int(i/(n_iter-1)*100))
                inpts = {"X": batch["encoded_image"].transpose(0, 1)}

                network.run(inpts=inpts, time=time_max, input_time_dim=1)

                if visualize:

                    if (t_now - t_start)//vis_interval + 1 > vis_counter:
                        vis_counter += 1
                        weights_XY = network.connections[('X', 'Y')].w.reshape(28, 28, -1)
                        weights_to_display = torch.zeros(0, 28*25)
                        i = 0
                        while i < 625:
                            for j in range(25):
                                weights_to_display_row = torch.zeros(28, 0)
                                for k in range(25):
                                    weights_to_display_row = torch.cat((weights_to_display_row, weights_XY[:, :, i]), dim=1)
                                    i += 1
                                weights_to_display = torch.cat((weights_to_display, weights_to_display_row), dim=0)

                        #weights_XY_plot.image(weights_to_display.numpy(), caption='Weights XY', clamp=True)
                        f = plt.figure(figsize=(15, 15))
                        plt.title('Weights XY')
                        plt.imshow(weights_to_display.numpy(), cmap='YlOrBr')
                        weights_to_display_plot2.pyplot(f)
                        plt.close()

            status_text.text('Training is done.')
            network.reset_()  # Reset state variables


if_train = st.button('Train')

if if_train:
    train(n_train=n_train, n_iter=n_iter, vis_interval=vis_interval)