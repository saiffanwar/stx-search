import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import json
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import dill as pck
import torch
import copy
from collections import defaultdict

mapbox_access_token = open(".mapboxtoken").read()

dataset = 'GRID'
if dataset == 'METR_LA':
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
    x_coords = longs
    y_coords = lats
elif dataset == 'GRID':
    gridsize = 10
    coords = [[i, j] for i in range(gridsize) for j in range(gridsize)]
    x_coords = [i for i,j in coords]
    y_coords = [j for i,j in coords]

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


def events_to_coords(events, explainer):
    xs, ys, zs = [], [], []
    values = []
    for e in events:
        event_obj = explainer.events[e]
        e_t, e_idx = event_obj.timestamp, event_obj.node_idx
        x, y = x_coords[e_idx], y_coords[e_idx]
        z = e_t
        xs.append(x)
        ys.append(y)
        zs.append(z)
        values.append(event_obj.value)

    return xs, ys, zs, values


def graph_visualiser(sa, explainer):
# Create a random graph using networkx
    x = explainer.data['X'].detach().cpu().numpy()
    model_y = explainer.model_y.detach().cpu().numpy()
    target_model_y = model_y[0, 0, sa.target_idx, 0]

    exp_y = explainer.exp_prediction(sa.best_events).detach().cpu().numpy()
    target_exp_y = exp_y[0, 0, sa.target_idx, 0]
    adj_mx = explainer.adj_mx



# Layout for the plot
    # Combine the traces into a figure
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                        subplot_titles=[f'Model Prediction: {target_model_y}', f'Explanation Prediction: {target_exp_y}'],
                        vertical_spacing=0,
                        horizontal_spacing=0.05)

    plot_axes = [[1, 1], [1, 2]]

    events = [explainer.candidate_events, sa.best_events]
    data = [[explainer.candidate_events, model_y], [sa.best_events, exp_y]]
    _,_,_, all_values = events_to_coords(events[0], explainer)
    min_val = min(min(all_values), min(model_y.flatten()), min(exp_y.flatten()))
    max_val = max(max(all_values), max(model_y.flatten()), max(exp_y.flatten()))
#    max_val = max(all_values)
    norm = plt.Normalize(vmin=min_val, vmax=max_val)


    plotting_data_num = 0
    for p, plot_d in enumerate(zip(plot_axes, data)):
        ax, input_pred_data = plot_d
        row_num, col_num = ax
        for i, d in enumerate(input_pred_data):

            if i == 0:
                xs, ys, zs, values = events_to_coords(d, explainer)
                cmap = plt.cm.get_cmap('Greens')
                node_idxs = [explainer.events[e].node_idx for e in d]
                node_values = [explainer.events[e].value for e in d]
                node_timestamps = [explainer.events[e].timestamp for e in d]
            else:
                xs = x_coords
                ys = y_coords
                zs = [explainer.input_window+1 for _ in range(len(x_coords))]
                node_idxs = list(range(len(x_coords)))
                node_timestamps = zs
                node_values = d[0, 0, :, 0].flatten()
                cmap = plt.cm.get_cmap('Reds')

            # Normalize node values to 0-1 for colormap

            # Use a colormap (here, we use 'viridis', but you can choose any matplotlib colormap)
#        if i == 0:
#        else:
#                    if timestamp == 0:
#                        continue
#                    else:
#                        break
            node_colors = [cmap(norm(i)) for i in node_values]
            # Map the node values to colors using the colormap

#        edges = fetch_layer_edges(subgraph_nodes, adj_mx)

#                x_edges = []
#                y_edges = []
#                z_edges = []
#                for edge in edges:
#                    s,t = edge
#                    x_edges += [longs[s], longs[t], None]  # x-coordinates of the edge start, end, and separator (None)
#                    y_edges += [lats[s], lats[t], None]  # x-coordinates of the edge start, end, and separator (None)
#                    z_edges += [plotting_timestamp, plotting_timestamp, None]  # x-coordinates of the edge start, end, and separator (None)
            # Extract the RGB colors for plotly (plotly needs RGB in the form 'rgb(R,G,B)')
            node_colors_rgb = [
                f'rgb({int(255 * color[0])},{int(255 * color[1])},{int(255 * color[2])})'
                for color in node_colors
            ]

# Create a 3D scatter plot for the nodes
            fig.add_trace(go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode='markers',
                marker=dict(size=5, color=node_colors_rgb, opacity=0.8),
                text=[f"Node: {n} \n Timestamp {t} \n Value: {v}" for n,t,v in zip(node_idxs, node_timestamps, node_values)],
                hoverinfo='text'
            ), row=row_num, col=col_num)


# Create a 3D line plot for the edges
#                fig.add_trace(go.Scatter3d(
#                    x=x_edges,
#                    y=y_edges,
#                    z=z_edges,
#                    mode='lines',
#                    line=dict(color='black', width=2),
#                    hoverinfo='none'
#                ), row=row_num, col=col_num)

        plotting_data_num += 1

        fig.add_trace(go.Scatter3d(
            x=[ x_coords[sa.target_idx] ],
            y=[ y_coords[sa.target_idx] ],
            z=[13],
            mode='markers',
            marker=dict(size=20, color='orange', opacity=0.8),
            text=[f"Node: {sa.target_idx} \n Timestamp {13} \n Value: {explainer.model_y[0, 0, sa.target_idx, 0]}"],
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
    return fig



def annealing_progression(sa, explainer, hide_rejected=False):
    probabilities = sa.acceptance_probabilities
    scores = sa.scores
    errors = sa.errors
    sizes = sa.exp_sizes
#    best_score = sa.best_score
    xs = np.arange(1, len(probabilities)+1, 1)
#    temperatures =  [sa.starting_temperature * (sa.cooling_rate ** i) for i in range(1, len(xs) + 1)]
#    ys = probabilities
#    actions = ['Accepted move' if a else 'Rejected' for a in sa.actions]

    hovertext = [f'Probability: {p} <br> Score: {s} <br> Error: {e} <br> Exp Size: {e_s} ' for p, s, e, e_s in zip(probabilities, scores, errors, sizes)]


    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=['Current Score', 'Explanation Size', 'Error', 'Acceptance Probability'],)

    fig.add_trace(go.Scatter(x=xs, y = scores, name='Current Score', text=hovertext, hoverinfo='text', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y = sizes, name='Explanation Size', text=hovertext, hoverinfo='text', line=dict(color='green')), row=1, col=2)
    fig.add_trace(go.Scatter(x=xs, y = errors, name='Error', text=hovertext, hoverinfo='text', line=dict(color='red')), row=2, col=1)
    fig.add_trace(go.Scatter(x=xs, y = probabilities, mode='markers', name='Acceptance Probabilities', text=hovertext, hoverinfo='text', line=dict(color='purple')), row=2, col=2)


    fig.update_layout(height=1000,
                  legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ))
#
#
    return fig


def explanation_heatmap(sa, explainer):

    target_idx = sa.target_idx

    num_nodes = np.zeros(len(explainer.adj_mx))

    all_xs, all_ys, _, _ = events_to_coords(explainer.candidate_events, explainer)

    for e in sa.best_events:
        e_t, e_idx = explainer.events[e].timestamp, explainer.events[e].node_idx
        num_nodes[e_idx] += 1



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
#    subgraph_nodes = [n for n,m in zip(all_subgraph_nodes, mask) if m]


    num_nodes = [n for n in num_nodes if n > 0]

    if dataset == 'METR_LA':
        fig.add_trace(go.Scattermapbox(
            lat=all_ys,
            lon=all_xs,
            mode='markers',
            name='Explanation Nodes',
            marker=go.scattermapbox.Marker(
                size=14,
                color=colours,
            ),
            text=[f"Node: {i} \n  Num Nodes: {n}" for i, n in enumerate(num_nodes)],
            hoverinfo='text'
        ))

#    fig.add_trace(go.Scattermapbox(
#        lat=[ target_y ],
#        lon=[ target_x ],
#        name='Target Node',
#        mode='markers',
#        marker=go.scattermapbox.Marker(
#            size=14,
#            color='orange',
#            opacity=0.6
#        ),
#        text=f"Node: {target_idx} \n  Num Nodes: {target_num_nodes}",
#        hoverinfo='text'
#    ))


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

    else:
        fig.add_trace(go.Scatter(
            x=all_xs,
            y=all_ys,
            mode='markers',
            name='Explanation Nodes',
            marker=dict(size=14, color=colours),
            text=[f"Node: {i} \n  Num Nodes: {n}" for i, n in enumerate(num_nodes)],
            hoverinfo='text'
        ))

        fig.add_trace(go.Scatter(
            x=[x_coords[sa.target_idx]],
            y=[y_coords[sa.target_idx]],
            mode='markers',
            name='Target Node',
            marker=dict(size=14, color='orange', opacity=0.6),
            text=f"Node: {sa.target_idx}",
            hoverinfo='text'
        ))

        fig.update_layout(
            title="Spatial Distribution of Nodes in Explanation",
            height=1000,
        )

    return fig


def exp_temporal_distribution(sa, explainer):
    target_idx = sa.target_idx
    num_timestamps = explainer.input_window
    node_nums = list(range(len(x_coords)))
    node_timestamps = defaultdict(lambda: np.zeros(explainer.input_window))

    for e in sa.best_events:
        node_timestamps[explainer.events[e].node_idx][explainer.events[e].timestamp] = 1
#
#    new_data =
#    nodes = []
#    for node in range(num_nodes):
#        if np.sum(data[node]) != 0:
#            new_data.append(np.sum(data[node] for i in data[node] if i != 0))
#            nodes.append(node)

#    print(np.shape(new_data))
#    data = np.array(new_data)
#    occurences = np.sum(data, axis=1)
#    order = np.argsort(occurences)
#    data = data[order]
#    nodes = [nodes[i] for i in order]
#    print(nodes)
    fig = go.Figure(data=go.Heatmap(
        z=list(node_timestamps.values()),
        colorscale='Greens',  # White for 0, Black for 1
        text=[[f"Node: {k}" for _ in range(len(v))] for k, v in node_timestamps.items()],
        hoverinfo='text',
        showscale=False,  # Hides color scale
        ))
    if target_idx in node_timestamps.keys():
        target_row = list(node_timestamps.keys()).index(target_idx)


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
