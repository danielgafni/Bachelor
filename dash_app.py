# -*- coding: utf-8 -*-
import datetime
import json
from time import time as t
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import torch
from torchvision.transforms import transforms
from tqdm import tqdm

from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from thesis.utils import view_database, load_network
from thesis.nets import LC_SNN
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from flask_caching import Cache

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

net = LC_SNN()
training = True
current_iteration = 0
total_iterations = 100
sum_iterations = 0
time_left = '?'
time_from_start = 0
speed = 0

weights_XY = go.Figure(go.Heatmap()).update_layout(height=600, width=600)


class LC_SNN_app(LC_SNN):
    def train(self, n_iter=100., vis_interval=10.):
        global training
        train_dataset = MNIST(
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
        self.n_iter_state = 0
        if n_iter is None:
            n_iter = self.n_iter
        self.network.train(True)
        print('Training network...')
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=True)
        cnt = 0
        global total_iterations
        total_iterations = n_iter
        global sum_iterations
        t_start = t()
        for i, batch in tqdm(list(zip(range(n_iter), train_dataloader)), ncols=100):
            if training:
                global current_iteration
                current_iteration = i + 1
                global speed
                global time_left
                global time_from_start
                sum_iterations += 1
                t_now = t()
                time_from_start = str(datetime.timedelta(seconds=(int(t_now - t_start))))
                speed = (i + 1) / (t_now - t_start)
                time_left = str(datetime.timedelta(seconds=int((n_iter - i) / speed)))
                inpts = {"X": batch["encoded_image"].transpose(0, 1)}
                self.network.run(inpts=inpts, time=self.time_max, input_time_dim=1)

                self._spikes = {
                    "X": self.spikes["X"].get("s").view(self.time_max, -1),
                    "Y": self.spikes["Y"].get("s").view(self.time_max, -1),
                    }

                if (t_now - t_start) / vis_interval > cnt:
                    global weights_XY
                    weights_XY = self.plot_weights_XY().update_layout(height=600, width=600)
            else:
                current_iteration = 0
                time_left = '?'
                time_from_start = 0
                speed = 0

                break

        self.network.reset_()
        self.network.train(False)


networks_database = view_database()


CACHE_CONFIG = {
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'dash_cache'
    }
cache = Cache()
cache.init_app(app.server, config=CACHE_CONFIG)

app.layout = html.Div(children=[

    html.H1(children='SNN Tools'),

    html.Div(className='row', children=[

        html.H2(children='''
            Network source
            '''),

        dcc.Dropdown(
            id='network-source',
            options=[
                {'label': 'Load', 'value': 'load'},
                {'label': 'Create', 'value': 'create'}
                ],
            value='create', style={'width': '40%'}),

        html.Label(children='Refresh interval, s'),

        dcc.Slider(id='vis-interval', min=1,
                   max=120,
                   step=None,
                   marks={
                       1: '1', 5: '5', 10: '10', 30: '30', 60: '60', 120: '120'
                       },
                   value=10)
        ], style={'width': '50%', 'padding': '20 20 20 20'}
        ),

    html.Hr(),

    html.Div(children=[

        html.Div(children=[

            html.Div(id='box-load', children=[

                html.Label(id='label-load', children='network directory name'),

                dcc.Input(id='network-name',
                          value='2e762b53737d742e89a9295d927abecad0fe4502ca223245a74562b7',
                          type='text',
                          size='67'),
                ], style={'width': '100%'}),

            html.Div(id='box-load-button', children=[

                html.Label(children='Load network'),

                html.Button(id='load-network', children='Load network'),
                ])
            ], style={'width': '39%', 'display': 'inline-block'}),

        html.Div(children=[

            html.Div(id='box-norm', children=[
                html.Label(id='label-norm', children='norm'),
                dcc.Input(id='parameter-norm', type='number', value='0.45')
                ], style={'rowCount': '2'}, className='four columns'),

            html.Div(id='box-c_w', children=[
                html.Label(id='label-c_w', children='Competitive weight'),
                dcc.Input(id='parameter-c_w', type='number', value='-100')
                ], style={'rowCount': '2'}, className='four columns'),

            html.Div(id='box-n_filters', children=[
                html.Label(id='label-n_filters', children='Number of filters'),
                dcc.Input(id='parameter-n_filters', type='number', value='25')
                ], style={'rowCount': '2'}, className='four columns'),

            html.Div(id='box-create-network', children=[
                html.Label(id='label-create-network', children='Create network'),
                html.Button(id='create-network', children='Create network', className='three columns',
                            style={'width': '100%'}
                            ),
                ], style={'rowCount': '2'}, className='three columns')
            ], style={'width': '59%', 'display': 'inline-block', 'float': 'right'}
            )
        ]
        ),

    html.Hr(),

    html.Label(id='label-parameters', children='Network parameters:'),

    html.Div(id='network-parameters', children='Network not loaded'),

    html.Div(id='training-section',
             children=[
                dcc.Input(id='train-n_iter', value='100', style={'padding': '10 10'}),
                html.Button(id='train', children='Train network'),
                html.Button(id='stop', children='Stop training'),
                html.Label(id='n_iter-counter', children='n_iter: 0 / 1000, 0 it/s'),
                 ]),

    html.Div(children=[
        dcc.Graph(id='weights-xy', figure=go.Figure(go.Heatmap()).update_layout(width=600, height=600))],

        style={'display': 'inline-block', 'padding': '5 5 5 5', 'rowCount': 1}
        ),

    html.Div(id='net-input', style={'display': 'none'}, children=r'{"source": "create", "norm": 0.26, "c_w": -100, '
                                                                 r'"n_filters": 25}'),

    html.Div(id='stash', style={'display': 'none'}, children=''),

    html.Div(id='stash2', style={'display': 'none'}, children=''),


    dcc.Interval(
        id='refresh-plots',
        interval=10 * 1000,
        n_intervals=0
        ),

    dcc.Interval(
        id='refresh-n_iter',
        interval=1000,
        n_intervals=0
        )
    ])


@app.callback(
    Output('weights-xy', 'figure'),
    [Input('create-network', 'n_clicks'),
     Input('refresh-plots', 'n_intervals')]
    )
def update_xy(n_clicks, vis_interval):
    global weights_XY
    return weights_XY


@app.callback(
    Output('net-input', 'children'),
    [Input('load-network', 'n_clicks'),
     Input('create-network', 'n_clicks')],
    [State('network-source', 'value'),
     State('network-name', 'value'),
     State('parameter-norm', 'value'),
     State('parameter-c_w', 'value'),
     State('parameter-n_filters', 'value')]
    )
def update_net_input(n_clicks_load, n_clicks_create, network_source, name, norm, c_w, n_filters):
    if network_source == 'create':
        input_dict = {
            'source': 'create',
            'norm': norm,
            'c_w': c_w,
            'n_filters': n_filters
            }
    if network_source == 'load':
        input_dict = {
            'source': 'load',
            'name': name
            }

    return json.dumps(input_dict)


@cache.memoize()
def global_network(input_string):
    global net
    global weights_XY
    input_dict = json.loads(input_string)
    source = input_dict['source']
    if source == 'create':
        print('Creating network...')
        norm = float(input_dict['norm'])
        c_w = float(input_dict['c_w'])
        n_filters = int(input_dict['n_filters'])
        net = LC_SNN_app(norm=norm, c_w=c_w, n_filters=n_filters)
        weights_XY = net.plot_weights_XY().update_layout(height=600, width=600)
        return net

    if source == 'load':
        print('Loading network...')
        try:
            net = load_network(input_dict['name'])
        except FileNotFoundError:
            net = LC_SNN_app()
            return None
        return net


@app.callback([Output('network-parameters', 'children')],
              [Input('load-network', 'n_clicks'),
               Input('create-network', 'n_clicks')],
              [State('net-input', 'children')]
              )
def update_network_string(n_clicks_load, n_clicks_create, input_string):
    net = global_network(input_string=input_string)
    global weights_XY
    weights_XY = net.plot_weights_XY().update_layout(height=600, width=600)
    return [str(net)]


@app.callback([Output('load-network', 'disabled'),
               Output('network-name', 'disabled'),
               Output('create-network', 'disabled'),
               ],
              [Input('network-source', 'value')]
              )
def update_layout(network_source):
    if network_source == 'load':
        load_button_disabled = False
        network_name_disabled = False
        create_network_disabled = True
    else:
        load_button_disabled = True
        network_name_disabled = True
        create_network_disabled = False

    return load_button_disabled, network_name_disabled, create_network_disabled


@app.callback(
    Output('refresh-plots', 'interval'),
    [Input('vis-interval', 'value')]
    )
def update_vis_interval(input_time_interval):
    return input_time_interval * 1000


@app.callback(
    Output('stash', 'children'),
    [Input('train', 'n_clicks')],
    [State('train-n_iter', 'value'),
     State('net-input', 'children'),
     State('vis-interval', 'value')]
    )
def train_network(n_clicks_train, n_iter, input_string, vis_interval):
    global training
    training = True
    net = global_network(input_string=input_string)
    if n_clicks_train is not None:
        net.train(n_iter=int(n_iter), vis_interval=int(vis_interval)/1000)
    training = False
    return ''


@app.callback(
    Output('stash2', 'children'),
    [Input('stop', 'n_clicks')]
    )
def stop_training_network(n_clicks_train):
    global training
    training = False
    return ''


@app.callback(
    Output('n_iter-counter', 'children'),
    [Input('refresh-n_iter', 'n_intervals')],
    [State('train-n_iter', 'value')]
    )
def update_n_iter_counter(interval, n_iter):
    global training, current_iteration, total_iterations, time_from_start, time_left, speed, sum_iterations
    total_iterations = int(n_iter)
    return f'Training: {training}. [{current_iteration}/{total_iterations}], {time_from_start}->{time_left}, ' \
           f'{round(speed, 2)}it/s, total training iterations: {sum_iterations}'


if __name__ == '__main__':
    app.run_server(debug=True)
