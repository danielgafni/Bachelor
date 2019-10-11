import torch
import os
import sys
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

from LC_SNN import LC_SNN

import streamlit as st


# @st.cache(ignore_hash=True, suppress_st_warning=True)
def create_network(norm, compettitive_weight, n_iter):
    with st.spinner('Creating network...'):
        net = LC_SNN(norm, compettitive_weight, n_iter)
        st.write('Network created')
        return net


def plot_weights(net):
    f = plt.figure(figsize=(15, 15))
    plt.imshow(net.weights_XY, cmap='YlOrBr')
    plt.colorbar()
    st.markdown('# Network weights')
    st.write(f)

def train_network(plot=False, vis_interval=10):
    net.train(n_iter=n_iter, plot=plot, vis_interval=vis_interval)
    st.write('Network trained')
    #if plot:
    #    plot_weights(net)

st.title('LC_SNN')
to_plot = st.checkbox('Visialize', True)
vis_interval = st.slider('Visualization interval, s', 1, 120, 60)
network_select = st.selectbox('Network source', ['Load network', 'Create network'])

if network_select == 'Create network':
    st.sidebar.markdown('# Parameters')
    norm = float(st.sidebar.text_input('norm', '0.2375'))
    compettitive_weight = float(st.sidebar.text_input('compettitive_weight', '-30'))
    n_iter = int(st.sidebar.text_input('n_iter', '100'))
    to_save = st.sidebar.checkbox('Save network after training', True)
    net = create_network(norm=norm, compettitive_weight=compettitive_weight, n_iter=n_iter)
    st.write(net)
    to_train = st.sidebar.button('Train network')
    if to_train:
        train_network(plot=to_plot, vis_interval=vis_interval)
        net.network.save(f'networks//norm={norm}_comp_weight={compettitive_weight}_n_iter={n_iter}')
        st.write(f'Network accuracy is {net.accuracy(1000)}')

if network_select == 'Load network':
    path = os.path.abspath(os.path.dirname(sys.argv[0]))
    filename = st.text_input('Enter netowork path', path+'\\network')
    to_load = st.button('Load')

    if to_load:
        net = LC_SNN(load=True)
        net.load(filename)
        st.write(net)
        st.write('Network loaded')
        if to_plot:
            plot_weights(net)