import torch
import os
import sys
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import numpy as np
import pandas as pd
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


#@st.cache(ignore_hash=True, suppress_st_warning=True)
def create_network(norm, compettitive_weight, n_iter, cropped_size, time_max, n_filters, stride, kernel_size):
    with st.spinner('Creating network...'):
        net = LC_SNN(norm, compettitive_weight, n_iter, cropped_size=cropped_size, time_max=time_max,
                     n_filters=n_filters, stride=stride, kernel_size=kernel_size)
        st.write('Network created')
        return net

def plot_weights(net):
    fig = net.plot_weights()
    st.plotly_chart(fig)

def plot_spikes(net):
    fig = net.plot_spikes()
    st.plotly_chart(fig)

def train_network(plot=False, vis_interval=10):
    net.train(n_iter=n_iter, plot=plot, vis_interval=vis_interval)


st.title('LC_SNN')
to_plot = st.checkbox('Visialize', True)
vis_interval = st.slider('Visualization interval, s', 1, 120, 10)
network_select = st.selectbox('Network source', ['Create network'])#['Load network', 'Create network'])

# Creating network
if network_select == 'Create network':
    st.sidebar.markdown('# Parameters')
    norm = float(st.sidebar.text_input('norm', '0.2'))
    compettitive_weight = float(st.sidebar.text_input('compettitive_weight', '-100'))
    n_iter = int(st.sidebar.text_input('n_iter', '1000'))
    time_max = int(st.sidebar.text_input('time_max', '250'))
    cropped_size = int(st.sidebar.text_input('cropped_size', '20'))
    n_filters = int(st.sidebar.text_input('n_filters', '25'))
    stride = int(st.sidebar.text_input('stride', '4'))
    kernel_size = int(st.sidebar.text_input('kernel_size', '12'))
    to_save = st.sidebar.checkbox('Save network after training', True)
    net = create_network(norm=norm, compettitive_weight=compettitive_weight, n_iter=n_iter, cropped_size=cropped_size,
                         time_max=time_max, n_filters=n_filters, stride=stride, kernel_size=kernel_size)
    st.write(net)
    to_train = st.sidebar.button('Train network')
    if to_train:
        if to_plot:
            st.write('# Visualization')
        train_network(plot=to_plot, vis_interval=vis_interval)

        if to_save:
            if not os.path.exists(f'networks'):
                os.makedirs(f'networks')
            net.network.save(f'networks//norm={norm}_comp_weight={compettitive_weight}_n_iter={n_iter}')
        #st.write(f'Network accuracy is {net.accuracy(1000)}')

# # Loading pre-trained network
# if network_select == 'Load network':
#     path = os.path.abspath(os.path.dirname(sys.argv[0]))
#     filename = st.text_input('Enter netowork path', path+'\\network')
#     to_load = st.button('Load')
#
#     if to_load:
#         net = LC_SNN(load=True)
#         net.load(filename)
#         st.write(net)
#         st.write('Network loaded')
#         if to_plot:
#             st.write('# Network weights')
#             net.visualize()