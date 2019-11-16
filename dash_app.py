# -*- coding: utf-8 -*-
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from thesis.utils import view_database, load_network
from thesis.nets import LC_SNN
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

net = LC_SNN()

networks_database = view_database()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

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
            value='load', style={'width': '40%'}),

        html.Label(children='Refresh interval, s'),

        dcc.Slider(id='vis-interval', min=1,
                   max=120,
                   step=None,
                   marks={
                       1: '1', 5: '5', 10: '10', 30: '30', 60: '60', 120: '120'
                       },
                   value=30)
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
                ], style={'rowCount': '2'}, className='three columns'),

            html.Div(id='box-c_w', children=[
                html.Label(id='label-c_w', children='Competitive weight'),
                dcc.Input(id='parameter-c_w', type='number', value='-60')
                ], style={'rowCount': '2'}, className='three columns'),

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

    html.Div(id='n_iter-counter',
             children=[
                dcc.Input(id='train-n_iter', value='1000', style={'padding': '10 10'}),
                html.Button(id='train', children='Train network'),
                html.Button(id='stop', children='Stop training'),
                html.Button(id='update', children='Update plots'),
                html.Label(id='n_iter', children='n_iter: 0 / 1000, 0 it/s'),
                 ]),

    html.Div(children=[
        dcc.Graph(id='weights-xy', figure=go.Figure(go.Heatmap()).update_layout(width=600, height=600))],

        style={'display': 'inline-block', 'padding': '5 5 5 5', 'rowCount': 1}
        ),
    dcc.Interval(
        id='vis_interval',
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
    Output('network-parameters', 'children'),
    [Input('load-network', 'n_clicks'),
     Input('create-network', 'n_clicks')],
    [State('network-source', 'value'),
     State('network-name', 'value'),
     State('parameter-norm', 'value'),
     State('parameter-c_w', 'value')]
    )
def update_network(n_clicks_load, n_clicks_create, source, network_name, norm, c_w):
    global net

    if source == 'create':
        norm = float(norm)
        c_w = float(c_w)

        net = LC_SNN(norm=norm, c_w=c_w)

        return str(net)

    if source == 'load':
        if n_clicks_load is not None:
            try:
                net = load_network(network_name)
            except FileNotFoundError:
                net = LC_SNN()
                return 'Network not found. Created a network with default parameters.'

            return str(net)
        else:
            return 'Network not loaded'


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
    Output('weights-xy', 'figure'),
    [Input('load-network', 'n_clicks'),
     Input('create-network', 'n_clicks'),
     Input('update', 'n_clicks'),
     Input('vis_interval', 'n_intervals')]
    )
def update_weights_xy(n_clicks_load, n_clicks_create, n_clicks_update, n_intervals):
    global net
    fig = go.Figure(go.Heatmap()).update_layout(height=600, width=600)
    if n_clicks_load is not None or n_clicks_create is not None:
        return net.plot_weights_XY().update_layout(height=600, width=600)
    else:
        return fig


@app.callback(
    [Output('vis_interval', 'interval')],
    [Input('vis-interval', 'value')]
    )
def update_vis_interval(input_time_interval):
    return [input_time_interval * 1000]


@app.callback(
    [],
    [Input('train', 'n_clicks')],
    [State('n_iter', 'children')]
    )
def train_network(n_clicks_train, n_iter):
    global net
    net.train(n_iter=n_iter)


@app.callback(
    [Output('n_iter', 'children')],
    [Input('refresh-n_iter', 'interval')],
    [State('train-n_iter', 'value')]
    )
def update_n_iter_counter(interval, n_iter):
    global net
    return [f'{net.n_iter_counter} / {n_iter}']


if __name__ == '__main__':
    app.run_server(debug=True)
