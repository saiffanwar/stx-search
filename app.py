# Import required libraries
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_daq as daq
import plotly.express as px
import pandas as pd
from visualisation_utils import graph_visualiser, annealing_progression, explanation_heatmap, exp_temporal_distribution
import numpy as np
import dill as pck
import sys
import argparse
import copy
#load_figure_template("darkly")


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--target_node', type=int, default=0, help='Target node index for explanation')
parser.add_argument('-s', '--subgraph_size', type=int, default=100, help='Size of the subgraph for explanation')
parser.add_argument('-m', '--mode', type=str, default='fidelity', help='Mode for the simulated annealing algorithm')
parser.add_argument('-d', '--dataset', type=str, default='GRID', help='Which dataset to use')
args = parser.parse_args()

results_dir = f'results/{args.dataset}/best_result_{args.target_node}_{args.subgraph_size}_{args.mode}.pck'

with open(results_dir, 'rb') as f:
    data = pck.load(f)
    explainer, sa = data

exp_vis_fig = graph_visualiser(sa, explainer)
exp_heatmap_fig = explanation_heatmap(sa, explainer)
annealing_progression_fig = annealing_progression(sa, explainer)
exp_temporal_distribution_fig = exp_temporal_distribution(sa, explainer)


# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div(children=[
    html.H1(children='Simulated Annealing to find an explanation subset of events in a graph'),


    html.Div(className='row', children=[
        html.P('Pause',
               style={'display': 'inline-block', 'width': '50px', 'margin-left': '0px'}
               ),
        daq.ToggleSwitch(
            id='pause-toggle',
            value=False,
            style={'display': 'inline-block', 'width': '100px', 'margin-left': '0px'}
            )]),
    # Button to update the plotly
    dcc.Graph(
        figure=annealing_progression_fig,
        id='annealing-progression-fig',
        ),

    html.Button('Update Explanation Graph Plot', id='update-plot-btn', n_clicks=0),
        # Plotly graph
    dcc.Graph(
        figure=exp_vis_fig,
        id='explanation_vis',
        ),
    html.Div(className='row', children=[
    # Plotly graph
    dcc.Graph(
        figure=exp_heatmap_fig,
        id='exp_heatmap_vis',
        style={'display': 'inline-block', 'width': '58vw', 'margin-left': '0px'}
    ),
    dcc.Graph(
        figure=exp_temporal_distribution_fig,
        id='exp_temporal_distribution_vis',
        style={'display': 'inline-block', 'width': '38vw', 'margin-right': '10px'}
        )
    ]),
    dcc.Interval(
        id='interval-component',
        interval=1*1000, # in milliseconds
        n_intervals=0
    )
])

@app.callback(
    Output('explanation_vis', 'figure'),
    Output('exp_heatmap_vis', 'figure'),
    Output('exp_temporal_distribution_vis', 'figure'),
    Input('update-plot-btn', 'n_clicks')
)
def update_graphs(n_clicks):
#    print(n_intervals)
#    sa.current_events, sa.current_score = sa.annealing_iteration(sa.current_events, sa.current_score)
    with open(results_dir, 'rb') as f:
        explainer, sa = pck.load(f)
    ''' plotting_data is a dictionary containing data for the full graph and the explanation graph with keys
        'full_input'  and 'explanation'. Each key contains a list of T lists where T is the length of the
        input window and each sublist contains [node longs, node lats, node timestamps, node indices] '''
    exp_vis_fig = graph_visualiser(sa, explainer)

    exp_heatmap_fig = explanation_heatmap(sa, explainer)
    exp_temporal_distribution_fig = exp_temporal_distribution(sa, explainer)

    return exp_vis_fig, exp_heatmap_fig, exp_temporal_distribution_fig

@app.callback(
    Output('annealing-progression-fig', 'figure'),
    Input('interval-component', 'n_intervals'),
    Input('pause-toggle', 'value')
)
def update_probs_plot(n_intervals, value):

    if not value:
        with open(results_dir, 'rb') as f:
            explainer, sa = pck.load(f)
        annealing_progression_fig = annealing_progression(sa, explainer)

    return annealing_progression_fig




# Run the app
if __name__ == '__main__':
#    while True:
#        try:
    app.run(debug=True)
#        except:
#            pass

