from .nets import LC_SNN, C_SNN
import os
import torch
import json
import pandas as pd
import plotly.graph_objs as go
from bindsnet.network.monitors import Monitor
from shutil import rmtree
from sqlite3 import connect

def view_network(name):
    if not os.path.exists(f'networks//{name}'):
        print('Network with such id does not exist')
        return None
    else:
        with open(f'networks//{name}//parameters.json', 'r') as file:
            parameters = json.load(file)
        return parameters


def view_database():
    database = pd.DataFrame(columns=[
        'name', 'accuracy', 'n_iter', 'mean_weight', 'n_filters', 'c_w', 'crop', 'kernel_size', 'stride', 'time_max'
        ])
    for name in os.listdir('networks'):
        if '.' not in name:
            parameters = view_network(name)
            if os.path.exists(f'networks//{name}//accuracy'):
                parameters['accuracy'] = torch.load(f'networks//{name}//accuracy')
            parameters['name'] = name
            database = database.append(parameters, ignore_index=True)

    return database


def plot_database(n_filters=None, network_type='LC_SNN'):
    data = view_database()
    data = data[data['type'] == network_type]
    if n_filters is None:
        color = data['n_filters']
        colorname = 'n_filters'
    else:
        data = view_database()[view_database()['n_filters'] == n_filters]
        color = data['n_iter']
        colorname = 'n_iter'


    data['error'] = ((data['accuracy'] * (1 - data['accuracy']) / data['n_iter']) ** 0.5).values

    fig = go.Figure(go.Scatter3d(x=data['c_w'], y=data['mean_weight'], z=data['accuracy'],
                                 error_z=dict(array=data['error'], visible=True, thickness=10, width=5, color='purple'),
                                 mode='markers', marker=dict(
            size=5,
            cmax=color.max(),
            cmin=0,
            color=color,
            colorbar=dict(title=colorname),
            colorscale='Viridis'
            )),

                    ).update_layout(height=1000, width=1000,
                                    margin={'l': 20, 'r': 20, 'b': 20, 't': 40, 'pad': 4},
                                    scene = dict(
                                        xaxis=dict(
                                            backgroundcolor='rgb(200,200, 230)',
                                            gridcolor='white',
                                            showbackground=True,
                                            title_text='c_w'
                                            ),
                                        yaxis=dict(
                                            backgroundcolor='rgb(230,200,230)',
                                            gridcolor='white',
                                            showbackground=True,
                                            title_text='mean_weight'
                                            ),
                                        zaxis=dict(
                                            backgroundcolor='rgb(230,230,200)',
                                            gridcolor='white',
                                            showbackground=True,
                                            title_text='accuracy'
                                            )
                                        ))

    return fig


def load_network(name):
    path = f'networks//{name}'
    try:
        with open(path + '//parameters.json', 'r') as file:
            parameters = json.load(file)
            mean_weight = parameters['mean_weight']
            c_w = parameters['c_w']
            n_iter = parameters['n_iter']
            time_max = parameters['time_max']
            crop = parameters['crop']
            kernel_size = parameters['kernel_size']
            n_filters = parameters['n_filters']
            stride = parameters['stride']
            intensity = parameters['intensity']
            type = parameters['type']
            c_l = False
            if 'c_l' in parameters.keys():
                c_l = parameters['c_l']
            nu = 0
            if nu in parameters.keys():
                nu = parameters['nu']
    except FileNotFoundError:
        raise FileNotFoundError

    accuracy = None
    votes = None
    conf_matrix = None

    if type == 'LC_SNN':
        net = LC_SNN(mean_weight=mean_weight, c_w=c_w, time_max=time_max, crop=crop,
                     kernel_size=kernel_size, n_filters=n_filters, stride=stride, intensity=intensity,
                     c_l=c_l, nu=nu)
        net.n_iter = n_iter
        if os.path.exists(path + '//votes'):
            votes = torch.load(path + '//votes')
            net.calibrated = True
        if os.path.exists(path + '//accuracy'):
            accuracy = torch.load(path + '//accuracy')
        if os.path.exists(path + '//confusion_matrix'):
            conf_matrix = torch.load(path + '//confusion_matrix')
        network = torch.load(path + '//network')
        net.network = network
        net.votes = votes
        net.accuracy = accuracy
        net.conf_matrix = conf_matrix
        net.spikes = {}
        for layer in set(net.network.layers):
            net.spikes[layer] = Monitor(net.network.layers[layer], state_vars=["s"], time=net.time_max)
            net.network.add_monitor(net.spikes[layer], name="%s_spikes" % layer)
        net._spikes = {
            "X": net.spikes["X"].get("s").view(net.time_max, -1),
            "Y": net.spikes["Y"].get("s").view(net.time_max, -1),
            }

        net.network.train(False)

    if type == 'C_SNN':
        net = C_SNN(mean_weight=mean_weight, c_w=c_w, time_max=time_max, crop=crop,
                    kernel_size=kernel_size, n_filters=n_filters, stride=stride, intensity=intensity)

        net.n_iter = n_iter
        if os.path.exists(path + '//votes'):
            votes = torch.load(path + '//votes')
            net.calibrated = True

        if os.path.exists(path + '//accuracy'):
            accuracy = torch.load(path + '//accuracy')

        if os.path.exists(path + '//confusion_matrix'):
            conf_matrix = torch.load(path + '//confusion_matrix')

        network = torch.load(path + '//network')

        net.network = network
        net.votes = votes
        net.accuracy = accuracy
        net.conf_matrix = conf_matrix

        net.spikes = {}
        for layer in set(net.network.layers):
            net.spikes[layer] = Monitor(net.network.layers[layer], state_vars=["s"], time=net.time_max)
            net.network.add_monitor(net.spikes[layer], name="%s_spikes" % layer)

        net._spikes = {
            "X": net.spikes["X"].get("s").view(net.time_max, -1),
            "Y": net.spikes["Y"].get("s").view(net.time_max, -1),
            }

        net.network.train(False)

    return net


def delete_network(name, sure=False):
    if not sure:
        print('Are you sure you want to delete the network? [Y/N]')
        if input() == 'Y':
            rmtree(f'networks//{name}')
            conn = connect(r'networks/networks.db')
            crs = conn.cursor()
            crs.execute(f'DELETE FROM networks WHERE id = ?', (name, ))
            conn.commit()
            conn.close()
            print('Network deleted!')
        else:
            print('Deletion canceled...')
    else:
        rmtree(f'networks//{name}')
        conn = connect(r'networks/networks.db')
        crs = conn.cursor()
        crs.execute(f'DELETE FROM networks WHERE id = ?', (name, ))
        conn.commit()
        conn.close()
        print('Network deleted!')


