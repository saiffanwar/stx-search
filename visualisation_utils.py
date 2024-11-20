import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import json
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import pickle as pck
import torch

mapbox_access_token = open(".mapboxtoken").read()

with open('visualized_data/METR_LA_dyna.json') as json_data:
    d = json.load(json_data)
    json_data.close()
nodes = d['features']

lats = []
longs = []
geo_ids = []

for node in nodes:
    lats.append(node['geometry']['coordinates'][1])
    longs.append(node['geometry']['coordinates'][0])
    geo_ids.append(node['properties']['geo_id'])

def adj_mx_to_edges(adj_mx):
    edges = []
    for s in range(len(adj_mx)):
        for t in range(len(adj_mx)):
            if adj_mx[s,t] > 0:
                edges.append([s,t])

    return edges

def fetch_layer_edges(subgraph_nodes, adj_mx):
    edges = []
    for n in subgraph_nodes:
        node_edges = adj_mx[:, n]
        connected_nodes = list(np.argwhere(node_edges > 0).flatten())
        for c in connected_nodes:
            if c in subgraph_nodes:
                edges.append([n, c])
#    print(subgraph_nodes, edges)
    return edges



def graph_visualiser(explainer, sa):
# Create a random graph using networkx
#    edges = adj_mx_to_edges(adj_mx)

    exp_nodes = sa.indices_to_events(sa.best_events)
    exp_y, exp_graph = explainer.make_prediction_from_exp_nodes(exp_nodes)

    target_exp_y = explainer.model_output_to_target_prediction(exp_y)
    target_exp_y = explainer.scaler.inverse_transform(target_exp_y)


#    exp_y = sa.best_exp_y
#    exp_events = sa.best_exp_events
#    exp_graph = sa.best_exp_graph

    input_graph = explainer.batch['X'].cpu().numpy()
    model_y = explainer.model_y.detach().cpu().numpy()
    target_model_y = explainer.model_output_to_target_prediction(model_y)
    target_model_y = explainer.scaler.inverse_transform(target_model_y)
    adj_mx = explainer.adj_mx


# Layout for the plot
    # Combine the traces into a figure
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                        subplot_titles=[f'Model Prediction: {target_model_y}', f'Explanation Prediction: {target_exp_y}'],
                        vertical_spacing=0,
                        horizontal_spacing=0.05)
#    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]],)

    plotting_data = {'full input': [], 'explanation': []}

    plot_axes = [[1, 1], [1, 2]]

    graphs = [[input_graph, model_y], [exp_graph, exp_y]]
    max_val = max([np.max(np.array(g).flatten()) for g in graphs])
    min_val = min([np.min(np.array(g).flatten()) for g in graphs])

    plotting_data_num = 0
    for ax, graph in zip(plot_axes, graphs):
        row_num, col_num = ax

        for timestamp in range(explainer.input_window):
            for i, g in enumerate(graph):
                if not ((i == 1 ) and timestamp > 0):
                # Create edge lists: start and end points

                    # If it is th output graph, adjust the timestamp
                    plotting_timestamp = timestamp + (i*12)
                    # Extract node positions into arrays for easy manipulation

                    rescaled_input = explainer.scaler.inverse_transform(g[0][..., :explainer.output_window])
                    G = rescaled_input[timestamp]
                    G = g[0][timestamp]
                    G = [float(g[0]) for g in G]

                    # Normalize node values to 0-1 for colormap
                    norm = plt.Normalize(vmin=min_val, vmax=max_val)

                    # Use a colormap (here, we use 'viridis', but you can choose any matplotlib colormap)
                    if i == 0:
                        cmap = plt.cm.get_cmap('Greens')
                    else:
                        cmap = plt.cm.get_cmap('Reds')
#                    if timestamp == 0:
#                        continue
#                    else:
#                        break

                    # Map the node values to colors using the colormap
                    node_colors = []
                    x_nodes = []
                    y_nodes = []
                    z_nodes = []
                    node_values = []

                    subgraph_nodes = []
                    for j, g in enumerate(G):
                        # For all unmasked nodes
                        if g != 0:
                            subgraph_nodes.append(j)
                            node_values.append(g)
                            node_colors.append(cmap(norm(g)))
                            x_nodes.append(longs[j])
                            y_nodes.append(lats[j])
                            z_nodes.append(plotting_timestamp)

                    if i == 0:
                        plotting_data[list(plotting_data.keys())[plotting_data_num]].append([x_nodes, y_nodes, z_nodes, subgraph_nodes, node_values])

                    edges = fetch_layer_edges(subgraph_nodes, adj_mx)

                    x_edges = []
                    y_edges = []
                    z_edges = []
                    for edge in edges:
                        s,t = edge
                        x_edges += [longs[s], longs[t], None]  # x-coordinates of the edge start, end, and separator (None)
                        y_edges += [lats[s], lats[t], None]  # x-coordinates of the edge start, end, and separator (None)
                        z_edges += [plotting_timestamp, plotting_timestamp, None]  # x-coordinates of the edge start, end, and separator (None)
                    # Extract the RGB colors for plotly (plotly needs RGB in the form 'rgb(R,G,B)')
                    node_colors_rgb = [
                        f'rgb({int(255 * color[0])},{int(255 * color[1])},{int(255 * color[2])})'
                        for color in node_colors
                    ]

# Create a 3D scatter plot for the nodes
                    fig.add_trace(go.Scatter3d(
                        x=x_nodes,
                        y=y_nodes,
                        z=z_nodes,
                        mode='markers',
                        marker=dict(size=5, color=node_colors_rgb, opacity=0.8),
                        text=[f"Node: {n} \n Timestamp {plotting_timestamp} \n Value: {i}" for n,i in zip(subgraph_nodes, node_values)],
                        hoverinfo='text'
                    ), row=row_num, col=col_num)

# Create a 3D line plot for the edges
                    fig.add_trace(go.Scatter3d(
                        x=x_edges,
                        y=y_edges,
                        z=z_edges,
                        mode='lines',
                        line=dict(color='black', width=2),
                        hoverinfo='none'
                    ), row=row_num, col=col_num)

        plotting_data_num += 1

        fig.add_trace(go.Scatter3d(
            x=[ longs[explainer.target_index] ],
            y=[ lats[explainer.target_index] ],
            z=[ explainer.target_timestamp+12 ],
            mode='markers',
            marker=dict(size=20, color='orange', opacity=0.8),
            text=[f"Node: {n} \n Timestamp {timestamp} \n Value: {i}" for n,i in zip(subgraph_nodes, node_values)],
            hoverinfo='text'
            ), row=row_num, col=col_num)

# Display the plot

    fig.update_layout(
        height=750,
#        width=1000,

        showlegend=False,
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False)
            ),

        scene2=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False)
            ),

        scene3=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False)
            ),

        scene4=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False)
            )
        )

#    fig.show()
    return fig, plotting_data



def annealing_progression(sa, hide_rejected=False):
    probabilities = sa.acceptance_probabilities
    scores = sa.scores
    best_score = sa.best_score
    xs = np.arange(1, len(probabilities)+1, 1)
    temperatures =  [sa.starting_temperature * (sa.cooling_rate ** i) for i in range(1, len(xs) + 1)]
    ys = probabilities
    actions = ['Accepted move' if a else 'Rejected' for a in sa.actions]



    fig = make_subplots(rows=3, cols=2, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=['Current Score', 'Acceptance Probabilities', 'Explanation Size', 'Score Delta', 'Error', 'Temperature'])

    # Accepted Probabilities
    worse_score_xs = [i for i in range(len(probabilities)) if probabilities[i] is not None]
#    probabilities = [p for p in probabilities if p is not None]
    colours = ['black' if a else 'lightgray' for a in sa.actions]

    total_scores = [s[0] for s in scores]
    exp_percentage_error = [s[2] for s in scores]
    exp_percentage_size = [s[3] for s in scores]
    exp_absolute_error = [s[4] for s in scores]
    deltas = [s[1]-s[0] for s in scores]
    exp_sizes = sa.exp_sizes

    if hide_rejected:
        total_scores = [s[0] if a else None for s, a in zip(scores, sa.actions)]
        exp_percentage_error = [s[2] if a else None for s, a in zip(scores, sa.actions)]
        exp_percentage_size = [s[3] if a else None for s, a in zip(scores, sa.actions)]
        exp_absolute_error = [s[4] if a else None for s, a in zip(scores, sa.actions)]
        exp_sizes = [s if a else None for s, a in zip(exp_sizes, sa.actions)]
    hovertext = [f'Probability: {p} <br> Score: {s[0]} <br> Current Score: {s[1]} <br> Action: {a} <br> Exp Size: {e} <br> Exp Error: {err} ' for p, s, a, e, err in zip(probabilities, scores, actions, exp_sizes, exp_absolute_error)]

    fig.add_trace(go.Scatter(x=xs, y=probabilities, mode='markers', name='Acceptance Probabilities', marker=dict(color=colours), text=hovertext, hoverinfo='text'), row=1, col=2)

    # Scores
    fig.add_trace(go.Scatter(x=xs, y=total_scores, mode='lines', text=hovertext, hoverinfo='text', name='Current Score', line=dict(color='blue')), row=1, col=1)
    if len(np.unique(exp_percentage_error)) > 1:
        fig.add_trace(go.Scatter(x=xs, y=exp_percentage_error, mode='lines', name='Error', line=dict(color='pink'), yaxis='y2'), row=1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=exp_percentage_size, mode='lines', name='Exp Size', line=dict(color='purple'), yaxis='y2'), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y = exp_absolute_error, mode='lines', text=hovertext, hoverinfo='text', line=dict(color='red')), row=3, col=1)

    # Deltas
    fig.add_trace(go.Scatter(x=xs, y = deltas , mode='lines', text=hovertext, hoverinfo='text', name='Score Delta', line=dict(color='orange')), row=2, col=2)

    # temperatures
    fig.add_trace(go.Scatter(x=xs, y = exp_sizes, mode='lines', name='Explanation Size', text=hovertext, hoverinfo='text', line=dict(color='green')), row=2, col=1)
    fig.add_trace(go.Scatter(x=xs, y = temperatures, mode='lines', name='Temperature', text=hovertext, hoverinfo='text', line=dict(color='purple')), row=3, col=2)


    fig.update_layout(height=1000,
                  legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ))


    return fig


def explanation_heatmap(plotting_data, sa):

    target_index = sa.explainer.target_index

    data = plotting_data['explanation']
    x_nodes, y_nodes, z_nodes, subgraph_nodes, node_values = data[0]

    num_nodes = np.zeros(len(plotting_data['full input'][0][0]))

    all_xs = plotting_data['full input'][0][0]
    all_ys = plotting_data['full input'][0][1]
    all_subgraph_nodes = plotting_data['full input'][0][3]

    target_x = all_xs[target_index]
    target_y = all_ys[target_index]

    for t in data:
        for n in t[3]:
            num_nodes[n] += 1

    target_num_nodes = num_nodes[target_index]

    cmap = plt.cm.get_cmap('Greens')
    norm = plt.Normalize(vmin=min(num_nodes), vmax=max(num_nodes))
    node_colours = [cmap(norm(n)) for n in num_nodes]

    node_colours_rgb = [
        f'rgb({int(255 * color[0])},{int(255 * color[1])},{int(255 * color[2])})'
        for color in node_colours
    ]
    fig = go.Figure()

    mask = [True if n > 0  else False for n in num_nodes]

    all_ys = [y for y,m in zip(all_ys, mask) if m]
    all_xs = [x for x,m in zip(all_xs, mask) if m]
    colours = [c for c,m in zip(node_colours_rgb, mask) if m]
    subgraph_nodes = [n for n,m in zip(all_subgraph_nodes, mask) if m]


    num_nodes = [n for n in num_nodes if n > 0]

    fig.add_trace(go.Scattermapbox(
        lat=all_ys,
        lon=all_xs,
        mode='markers',
        name='Explanation Nodes',
        marker=go.scattermapbox.Marker(
            size=14,
            color=colours,
        ),
        text=[f"Node: {n} \n  Num Nodes: {i}" for n,i in zip(subgraph_nodes, num_nodes)],
        hoverinfo='text'
    ))

    fig.add_trace(go.Scattermapbox(
        lat=[ target_y ],
        lon=[ target_x ],
        name='Target Node',
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=14,
            color='orange',
            opacity=0.6
        ),
        text=f"Node: {target_index} \n  Num Nodes: {target_num_nodes}",
        hoverinfo='text'
    ))


    fig.update_layout(
#        mapbox_style="satellite-streets",
        title="Spatial Distribution of Nodes in Explanation",
        hovermode='closest',
        mapbox=dict(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=34.121990,
                lon=-118.2717
            ),
            pitch=0,
            zoom=10
        ),
        height=1000,
    )


    return fig


def exp_temporal_distribution(plotting_data, sa):
    target_index = sa.explainer.target_index
    exp_data = plotting_data['explanation']
    all_data = plotting_data['full input']
    num_timestamps = len(all_data)
    num_nodes = len(all_data[0][0])
    subgraph_nodes = [e[3] for e in exp_data]
    data = np.zeros((num_nodes, num_timestamps))

    for timestamp, nodes in enumerate(subgraph_nodes):
        for node in nodes:
            data[node, timestamp] = 1

    new_data = []
    nodes = []
    for node in range(num_nodes):
        if np.sum(data[node]) != 0:
            new_data.append(np.sum(data[node] for i in data[node] if i != 0))
            nodes.append(node)

    print(np.shape(new_data))
    data = np.array(new_data)
    occurences = np.sum(data, axis=1)
    order = np.argsort(occurences)
    data = data[order]
    nodes = [nodes[i] for i in order]
    print(nodes)
    fig = go.Figure(data=go.Heatmap(
        z=data,
        colorscale='Greens',  # White for 0, Black for 1
        text=[[f"Node: { nodes[i] }" for _ in range(num_timestamps)] for i in range(len(nodes))],
        hoverinfo='text',
        showscale=False,  # Hides color scale
        ))
    if target_index in nodes:
        target_row = nodes.index(target_index)


        fig.add_shape(
        type="rect",
        x0=-0.5, x1=num_timestamps-0.5,   # span the width of the row
        y0=target_row - 0.5, y1=target_row + 0.5,
        line=dict(color="orange", width=2)  # Outline color and width
        )

    fig.update_layout(
        height=1000,
        title='Temporal Distribution of Nodes in Explanation',
        xaxis_title="Timestamp",
        yaxis_title="Location Index"
    )

    return fig
