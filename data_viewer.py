import pickle as pck
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from numpy.random import f
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

mapbox_access_token = open(".mapboxtoken").read()

geofile = pd.read_csv('raw_data/METR_LA/METR_LA.geo')
print(geofile.head())
location_coordinates = geofile['coordinates'].to_numpy()


#for batch in range(64):


fig = go.Figure(go.Scattermapbox(
        lat=['45.5017'],
        lon=['-73.5673'],
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
            lat=45,
            lon=-73
        ),
        pitch=0,
        zoom=5
    )
)

fig.show()
