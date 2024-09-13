import networkx as nx
import plotly.graph_objects as go
import numpy as np
import json
from matplotlib import pyplot as plt


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



def graph_visualiser(explainer, input_graph, adj_mx):
# Create a random graph using networkx
    edges = adj_mx_to_edges(adj_mx)

    rescaled_input = explainer.scaler.inverse_transform(input_graph[0][..., :explainer.output_window])
# Layout for the plot
    layout = go.Layout(
        title='3D Graph Network',
        showlegend=False,
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False)
            )
        )

    # Combine the traces into a figure
    fig = go.Figure(layout=layout)
    all_values = np.array(input_graph).flatten()
    max_val = np.max(all_values)
    min_val = np.min(all_values)
    num_0s = 0


    for timestamp in range(np.array(input_graph).shape[1]):
        print(timestamp)
        # Extract node positions into arrays for easy manipulation
        x_nodes = longs  # y-coordinates of nodes
        y_nodes = lats  # x-coordinates of nodes
        z_nodes = [timestamp for l in lats]  # z-coordinates of nodes

        # Create edge lists: start and end points
        x_edges = []
        y_edges = []
        z_edges = []

        for edge in edges:
            s,t = edge
            x_edges += [x_nodes[s], x_nodes[t], None]  # x-coordinates of the edge start, end, and separator (None)
            y_edges += [y_nodes[s], y_nodes[t], None]  # x-coordinates of the edge start, end, and separator (None)
            z_edges += [timestamp, timestamp, None]  # x-coordinates of the edge start, end, and separator (None)
        G = rescaled_input[timestamp]
        print(G.shape)
        G = [g[0] for g in G]

        # Normalize node values to 0-1 for colormap
        norm = plt.Normalize(vmin=min_val, vmax=max_val)

        # Use a colormap (here, we use 'viridis', but you can choose any matplotlib colormap)
        cmap = plt.cm.get_cmap('Blues')

        # Map the node values to colors using the colormap
        node_colors = [cmap(norm(value)) for value in G]

        # Extract the RGB colors for plotly (plotly needs RGB in the form 'rgb(R,G,B)')
        node_colors_rgb = [
            f'rgb({int(255 * color[0])},{int(255 * color[1])},{int(255 * color[2])})'
            for color in node_colors
        ]
        for i in range(len(G)):
            if G[i] == 0:
                num_0s += 1


# Create a 3D scatter plot for the nodes
        fig.add_trace(go.Scatter3d(
            x=x_nodes,
            y=y_nodes,
            z=z_nodes,
            mode='markers',
            marker=dict(size=10, color=node_colors_rgb, opacity=0.8),
            text=[f"Node Value: {i}" for i in G],
            hoverinfo='text'
        ))

# Create a 3D line plot for the edges
        fig.add_trace(go.Scatter3d(
            x=x_edges,
            y=y_edges,
            z=z_edges,
            mode='lines',
            line=dict(color='black', width=2),
            hoverinfo='none'
        ))


# Display the plot
    print(num_0s)
    fig.show()

