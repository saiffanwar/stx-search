import pickle as pck
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from numpy.random import f
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import ast
import json


mapbox_access_token = open(".mapboxtoken").read()

#results = pck.load(open('results.pkl', 'rb'))
#
#for k, v in results.items():
#    results[k] = v.cpu().numpy()
#    print(k, results[k].shape)

geofile = pd.read_csv('raw_data/PEMS_BAY/PEMS_BAY.geo')

longs = []
lats = []
for c in geofile['coordinates']:
    vals = c[1:-1].split(',')
    longs.append(float(vals[0]))
    lats.append(float(vals[1]))


fig = go.Figure(go.Scattermapbox(
        lat=lats,
        lon=longs,
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=14
        ),
        text=['Montreal'],
    ))

fig.update_layout(
    hovermode='closest',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=go.layout.mapbox.Center(
            lat=lats[0],
            lon=longs[0]
        ),
        pitch=0,
        zoom=10
    )
)

fig.show()
