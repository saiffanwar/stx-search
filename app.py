# Import required libraries
import dash
from dash import dcc, html, ctx
from dash.dependencies import Input, Output
import dash_daq as daq
import plotly.express as px
import pandas as pd
from visualisation_utils import Visualisation
import numpy as np
import dill as pck
import sys
import os
import argparse
import copy
from PIL import Image
# load_figure_template("darkly")


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--target_node', type=int,
                    default=12, help='Target node index for explanation')
parser.add_argument('-s', '--exp_size', type=int,
                    default=20, help='Size of the subgraph for explanation')
parser.add_argument('-m', '--mode', type=str, default='fidelity',
                    help='Mode for the simulated annealing algorithm')
parser.add_argument('--model', type=str,
                    default='TGCN', help='Which model to use')
parser.add_argument('-d', '--dataset', type=str,
                    default='METR_LA', help='Which dataset to use')
args = parser.parse_args()

logo_path = 'assets/logo.png'


directory = f'results/{args.dataset}/stx_search/'
all_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
available_target_results = [[a.split('_')[-2], a.split('_')[-1].split('.')[0]] for a in all_files]

args.target_node = available_target_results[0][0]
args.exp_size = int(available_target_results[0][1])

visualiser = Visualisation(model=args.model, dataset=args.dataset, event_idx=args.target_node,
                           exp_size=args.exp_size, mode=args.mode)
exp_graph_fig, exp_heatmap_fig, exp_progression_fig, exp_temporal_distribution_fig = visualiser.generate_plots()

# available_instances = 

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div(children=[
    html.Div([
        html.Img(src=app.get_asset_url('logo.png'), style={'width': '300px'}),
    ], style={
        'display': 'flex',
        'align-items': 'center',
        'justify-content': 'center',
        'padding-top': '20px'
    }),



html.Div(
    className='row',
    style={'display': 'flex', 'alignItems': 'center', 'gap': '10px', 'justifyContent': 'flex-start'},
    children=[
        html.Label(
            'Select Dataset:',
            style={'fontWeight': 'bold'}
        ),
        dcc.Dropdown(
            id='dataset-dropdown',
            options=[{'label': f'{d}', 'value': d} for d in ['METR_LA', 'PEMS_BAY']],
            value=args.dataset,
            style={'width': '250px'}
        ),
        html.Label(
            'Select target event and explanation size:',
            style={'fontWeight': 'bold'}
        ),
        dcc.Dropdown(
            id='target-node-dropdown',
            options=[{'label': f'Event {i[0]}, Exp Size: {i[1]}', 'value': i} for i in available_target_results],
            value=available_target_results[0],
            style={'width': '250px'}
        ),
        html.P(
            'Pause',
            style={'margin': '0'}
        ),
        daq.ToggleSwitch(
            id='pause-toggle',
            value=False,
            style={'width': '60px'}
        )
    ]),
    # Button to update the plotly
    dcc.Graph(
        figure=exp_progression_fig,
        id='exp-progression-fig',
    ),

    html.Button('Update Explanation Graph Plot',
                id='update-plot-btn', n_clicks=0),
    html.Div(
    style={'display': 'flex', 'width': '100%'},
    children=[
        dcc.Graph(
            figure=exp_graph_fig,
            id='explanation_graph',
            style={'width': '50%'}
        ),
        dcc.Graph(
            figure=exp_heatmap_fig,
            id='exp_heatmap_fig',
            style={'width': '30%'}
        ),
        dcc.Graph(
            figure=exp_temporal_distribution_fig,
            id='exp_temporal_distribution_fig',
            style={'width': '20%'}
        )
    ]),
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # in milliseconds
        n_intervals=0
    )
])


@app.callback(
    Output('explanation_graph', 'figure'),
    Output('exp_heatmap_fig', 'figure'),
    Output('exp_temporal_distribution_fig', 'figure'),
    Input('update-plot-btn', 'n_clicks')
)
def update_graphs(n_clicks):
    visualiser.reload_result_file()
    exp_graph_fig = visualiser.graph_visualiser()

    exp_heatmap_fig = visualiser.explanation_heatmap()
    exp_temporal_distribution_fig = visualiser.exp_temporal_distribution()

    return exp_graph_fig, exp_heatmap_fig, exp_temporal_distribution_fig


@app.callback(
    Output('exp-progression-fig', 'figure'),
    Input('interval-component', 'n_intervals'),
    Input('pause-toggle', 'value'),
)
def update_probs_plot(n_intervals, value):

    if value == False:
        visualiser.reload_result_file()
        exp_progression_fig = visualiser.exp_progression()
    else:
        exp_progression_fig = visualiser.exp_progression()

    return exp_progression_fig


@app.callback(
    Output('target-node-dropdown', 'options'),
    Output('target-node-dropdown', 'value'),
    Input('dataset-dropdown', 'value')
)
def update_target_node(dataset):
    directory = f'results/{dataset}/stx_search/'
    all_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    available_target_results = [(a.split('_')[-2], a.split('_')[-1].split('.')[0]) for a in all_files]
    available_options = [{'label': f'Event {i[0]}, Exp Size: {i[1]}', 'value': i} for i in available_target_results]

    target_exp = available_target_results[0]
    args.exp_size = int(target_exp[1])
    args.dataset = dataset
    target_event = target_exp[0]

    global visualiser
    visualiser = Visualisation(model=args.model, dataset=dataset, event_idx=target_event,
                               exp_size=args.exp_size, mode=args.mode)

    return available_options, target_exp

@app.callback(
    Input('target-node-dropdown', 'value')

)
def update_visualiser_from_target(value):
    if value is None:
        value = available_target_results[0]
    target_event = value[0]
    args.exp_size = int(value[1])

    global visualiser
    visualiser = Visualisation(model=args.model, dataset=args.dataset, event_idx=target_event,
                               exp_size=args.exp_size, mode=args.mode)

    raise dash.exceptions.PreventUpdate

# Run the app
if __name__ == '__main__':
    #    while True:
    #        try:
    app.run(debug=False)
#        except:
#            pass
