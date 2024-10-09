import plotly.graph_objects as go
import numpy as np
import json
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import pickle as pck
import torch


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



def graph_visualiser(explainer, input_graph, masked_input, model_y, exp_y, adj_mx):
# Create a random graph using networkx
#    edges = adj_mx_to_edges(adj_mx)



# Layout for the plot
    # Combine the traces into a figure
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
#                        subplot_titles=['Input Graph', 'Masked Input', 'True Output', 'Predicted Output'],
                        vertical_spacing=0,
                        horizontal_spacing=0)
#    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]],)

    plot_axes = [[1, 1], [1, 2]]
    graphs = [[input_graph, model_y], [masked_input, exp_y]]
    max_val = max([np.max(np.array(g).flatten()) for g in graphs])
    min_val = min([np.min(np.array(g).flatten()) for g in graphs])

    for ax, graph in zip(plot_axes, graphs):
        row_num, col_num = ax

        for timestamp in range(explainer.input_window):
            # Create edge lists: start and end points

            for i, g in enumerate(graph):
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

                # Map the node values to colors using the colormap
                node_colors = []
                x_nodes = []
                y_nodes = []
                z_nodes = []
                node_values = []

                subgraph_nodes = []
                for i, g in enumerate(G):
                    if g != 0:
                        subgraph_nodes.append(i)
                        node_values.append(g)
                        node_colors.append(cmap(norm(g)))
                        x_nodes.append(lats[i])
                        y_nodes.append(longs[i])
                        z_nodes.append(plotting_timestamp)

                edges = fetch_layer_edges(subgraph_nodes, adj_mx)

                x_edges = []
                y_edges = []
                z_edges = []
                for edge in edges:
                    s,t = edge
                    x_edges += [lats[s], lats[t], None]  # x-coordinates of the edge start, end, and separator (None)
                    y_edges += [longs[s], longs[t], None]  # x-coordinates of the edge start, end, and separator (None)
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

    fig.add_trace(go.Scatter3d(
        x=[ lats[explainer.target_index] ],
        y=[ longs[explainer.target_index] ],
        z=[ explainer.target_timestamp+12 ],
        mode='markers',
        marker=dict(size=10, color='red', opacity=0.8),
        text=[f"Node: {n} \n Timestamp {timestamp} \n Value: {i}" for n,i in zip(subgraph_nodes, node_values)],
        hoverinfo='text'
        ), row=row_num, col=col_num)

# Display the plot

    fig.update_layout(
        title='3D Graph Network',
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

    fig.show()
