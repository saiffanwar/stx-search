# Import required libraries
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_daq as daq
import plotly.express as px
import pandas as pd
from visualisation_utils import Visualisation
import numpy as np
import dill as pck
import sys
import argparse
import copy
#load_figure_template("darkly")


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--target_node', type=int, default=10, help='Target node index for explanation')
parser.add_argument('-s', '--subgraph_size', type=int, default=50, help='Size of the subgraph for explanation')
parser.add_argument('-m', '--mode', type=str, default='fidelity+size', help='Mode for the simulated annealing algorithm')
parser.add_argument('-d', '--dataset', type=str, default='GRID', help='Which dataset to use')
args = parser.parse_args()


visualiser = Visualisation(dataset=args.dataset, target_idx=args.target_node, subgraph_size=args.subgraph_size, mode=args.mode)
exp_graph_fig, exp_heatmap_fig, exp_progression_fig, exp_temporal_distribution_fig = visualiser.generate_plots()

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
        figure=exp_progression_fig,
        id='exp-progression-fig',
        ),

    html.Button('Update Explanation Graph Plot', id='update-plot-btn', n_clicks=0),
        # Plotly graph
    dcc.Graph(
        figure=exp_graph_fig,
        id='explanation_graph',
        ),
    html.Div(className='row', children=[
    # Plotly graph
    dcc.Graph(
        figure=exp_heatmap_fig,
        id='exp_heatmap_fig',
        style={'display': 'inline-block', 'width': '58vw', 'margin-left': '0px'}
    ),
    dcc.Graph(
        figure=exp_temporal_distribution_fig,
        id='exp_temporal_distribution_fig',
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
    Output('explanation_graph', 'figure'),
    Output('exp_heatmap_fig', 'figure'),
    Output('exp_temporal_distribution_fig', 'figure'),
    Input('update-plot-btn', 'n_clicks')
)
def update_graphs(n_clicks):
#    print(n_intervals)
#    sa.current_events, sa.current_score = sa.annealing_iteration(sa.current_events, sa.current_score)
    visualiser.reload_result_file()
    exp_graph_fig = visualiser.graph_visualiser()

    exp_heatmap_fig = visualiser.explanation_heatmap()
    exp_temporal_distribution_fig = visualiser.exp_temporal_distribution()

    return exp_graph_fig, exp_heatmap_fig, exp_temporal_distribution_fig

@app.callback(
    Output('exp-progression-fig', 'figure'),
    Input('interval-component', 'n_intervals'),
    Input('pause-toggle', 'value')
)
def update_probs_plot(n_intervals, value):

    if value == False:
        visualiser.reload_result_file()
        exp_progression_fig = visualiser.exp_progression()
    else:
        exp_progression_fig = visualiser.exp_progression()


    return exp_progression_fig




# Run the app
if __name__ == '__main__':
#    while True:
#        try:
    app.run(debug=False)
#        except:
#            pass

