import json
import plotly.express as px
import plotly.graph_objects as go
from pprint import pprint
import dash
from dash import dcc, html, Input, Output

app = dash.Dash(__name__)


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


fig = go.Figure()

fig.add_trace(go.Scattermapbox(
    lat=lats,
    lon=longs,
    mode='markers',
    marker=go.scattermapbox.Marker(
        size=14,
        color='red'
    ),
))


fig.update_layout(
    mapbox_style="satellite-streets",
    hovermode='closest',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=go.layout.mapbox.Center(
            lat=34.121990,
            lon=-118.2717
        ),
        pitch=0,
        zoom=11
    )
)

# Define the app layout
app.layout = html.Div([
    html.H1("Data Visualisation"),

    dcc.Input(
        id='input-text',
        type='number',
        placeholder='Node to view...',
        value=0  # Default value
    ),
    dcc.Input(
        id='input-geo-id',
        type='number',
        placeholder='Node to view...',
        value=773869  # Default value
    ),

    dcc.Graph(id='map', figure=fig, style={'width': '100%', 'height': '100vh'})
])

@app.callback(
    Output('map', 'figure', allow_duplicate=True),
    Input('input-text', 'value'),

    prevent_initial_call=True
)

def update_map(node):

    fig.data = []

    n_idxs = list(range(len(lats)))
    fig.add_trace(go.Scattermapbox(
        lat=lats,
        lon=longs,
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=14,
            color='red'
        ),
        text=list(zip(n_idxs, geo_ids)),
        hoverinfo='text+lon+lat',
    ))
    node = int(node)
    fig.add_trace(go.Scattermapbox(
        lat=[lats[node]],
        lon=[longs[node]],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=20,
            color='green'
        ),
        text=[str(node)+', '+str(geo_ids[node])],
        hoverinfo='text+lon+lat',
    ))
    return fig

@app.callback(
    Output('map', 'figure', allow_duplicate=True),
    Input('input-geo-id', 'value'),
    prevent_initial_call=True
)

def update_map_geo_id(geo_id):

    fig.data = []

    n_idxs = list(range(len(lats)))
    fig.add_trace(go.Scattermapbox(
        lat=lats,
        lon=longs,
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=14,
            color='red'
        ),
        text=list(zip(n_idxs, geo_ids)),
        hoverinfo='text+lon+lat',
    ))
    node = geo_ids.index(int(geo_id))
    fig.add_trace(go.Scattermapbox(
        lat=[lats[node]],
        lon=[longs[node]],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=20,
            color='green'
        ),
        text=[str(node)+', '+str(geo_ids[node])],
        hoverinfo='text+lon+lat',
    ))
    return fig
# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

