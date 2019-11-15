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

    html.Label(children='''
        Choose network source
        '''),

    dcc.Dropdown(
        id='network-source',
        options=[
            {'label': 'Load', 'value': 'load'},
            {'label': 'Create', 'value': 'create'}
            ],
        value='load', style={'width': '25%'}),

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

    html.Label(id='label-parameters', children='Network parameters'),

    html.Div(id='network-parameters', children='Network not loaded'),

    html.Div(children=[
        dcc.Input(id='train-n_iter', value='1000', style={'padding': '10 10'}),
        html.Button(children='Train network'),
        html.Button(children='Stop training'),
        html.Label(children='n_iter: 0 / 1000, 0 it/s'),
        ]),

    html.Div(children=[
        dcc.Graph(id='weights-xy', figure=go.Figure(go.Heatmap()).update_layout(width=800, height=800)),

        dcc.Graph(id='weights-yy', figure=go.Figure(go.Heatmap()).update_layout(width=800, height=800)),
        ], style={'columnCount': '2'}
        )
    ])


@app.callback(
    [Output('network-parameters', 'children'),
     Output('weights-xy', 'figure')],
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

        return str(net), net.plot_weights_XY()

    if source == 'load':
        if n_clicks_load is not None:
            try:
                net = load_network(network_name)
            except FileNotFoundError:
                net = LC_SNN()
                return 'Network not found. Created a network with default parameters.', net.plot_weights_XY()
            return str(net), net.plot_weights_XY()
        else:
            return 'Network not loaded', go.Figure(go.Heatmap()).update_layout(width=800, height=800)



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

    return load_button_disabled, network_name_disabled, create_network_disabled#, parameters_disabled


if __name__ == '__main__':
    app.run_server(debug=True)
