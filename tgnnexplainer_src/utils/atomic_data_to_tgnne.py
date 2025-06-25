import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

dataset_name = 'PEMS_BAY'
recording_frequency = 300

def generate_csv_from_dyna():
    dyna_df = pd.read_csv(f'raw_data/{dataset_name}/{dataset_name}.dyna')

    time_to_datetime = lambda t: datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ")
    print('Converting to datetime ...')
    dyna_df['time'] = dyna_df['time'].apply(time_to_datetime)
    starting_time = min(dyna_df['time'])
    delta_seconds = lambda t: (t - starting_time).total_seconds()/recording_frequency
    dyna_df['time'] = dyna_df['time'].apply(delta_seconds)

    df = pd.DataFrame({})

    df['u'] = dyna_df['entity_id']
    df['ts'] = dyna_df['time']
    df['label'] = dyna_df['traffic_speed']
    df['f0'] = dyna_df['traffic_speed']

    print('Saving to CSV ...')
    df.to_csv(f'raw_data/{dataset_name}/ml_{dataset_name}.csv', index=False)


def edge_feat():
    rel_df = pd.read_csv(f'raw_data/{dataset_name}/{dataset_name}.rel')
    edge_df = pd.DataFrame({})
    edge_df['src'] = rel_df['origin_id']
    edge_df['destination'] = rel_df['destination_id']
    edge_df['f0'] = rel_df['cost']

    edge_df.to_csv(f'raw_data/{dataset_name}/ml_{dataset_name}_edge_feats.csv', index=False)
    # num_nodes = rel_df['origin_id'].nunique()
   # adj_mx = np.zeros((num_nodes, num_nodes), dtype=float)
   # for index, row in rel_df.iterrows():
   #     adj_mx[row['origin_id'], row['destination_id']] = 1
   #
   # with open(f'raw_data/{dataset_name}/ml_{dataset_name}_adj_mx.npy', 'wb') as f:
   #     np.save(f, adj_mx)
   #
   # edge_feats = defaultdict(float)
   # for index, row in rel_df.iterrows():
   #     edge_feats[(row['origin_id'], row['destination_id'])] = row['cost']
   # with open(f'raw_data/{dataset_name}/ml_{dataset_name}_edge_feats.npy', 'wb') as f:
   #     np.save(f, edge_feats)

def read_edge_feats():
    edge_feats = np.load(f'raw_data/{dataset_name}/ml_{dataset_name}_edge_feats.npy', allow_pickle=True).item()
    print(edge_feats)

# generate_csv_from_dyna()
edge_feat()
# read_edge_feats()

