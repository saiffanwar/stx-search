import numpy as np
import torch
import dill as pck
import os
import argparse
import copy
from matplotlib import pyplot as plt
import random
import pandas as pd

from libcity.data import get_dataset
from libcity.utils import get_model
from libcity.config import ConfigParser

import cProfile
import pstats
import time

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default='TGCN', help='Mode of operation: generate or visualise')
parser.add_argument('-t', '--target_idx', type=int, default=12, help='Target node index for explanation')
parser.add_argument('-s', '--subgraph_size', type=int, default=50, help='Size of the subgraph for explanation')
parser.add_argument('-d', '--dataset', type=str, default='METR_LA', help='Dataset name')
parser.add_argument('--mode', type=str, default='fidelity', help='Explanation Mode')
args = parser.parse_args()


class TGNNExplainer_LibCity:

    def __init__(self, model_name, dataset_name):
        self.task='traffic_state_pred'
        self.model_name=model_name
        self.dataset_name=dataset_name
        self.config_file=None
        self.saved_model=True
        self.train=False
        self.other_args={'exp_id': '1', 'seed': 0}
        self.config = ConfigParser(self.task, self.model_name, self.dataset_name, self.config_file, self.saved_model, self.train, self.other_args)
        self.config['weight_adj_epsilon'] = 0.000003
#        self.device = self.config.get('device', torch.device('cpu'))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.input_window = self.config.get('input_window', 3)
        self.output_window = self.config.get('output_window', 1)

        self.data_feature, self.train_data, self.valid_data, self.test_data = self.load_data()
        self.scaler = self.data_feature['scaler']
        model_path = os.getcwd()+f'/libcity/cache/1/model_cache/{self.model_name}_{self.dataset_name}.m'
        self.model = self.load_model(model_path, self.data_feature)
        self.adj_mx = self.data_feature['adj_mx']
        self.data = next(iter(self.train_data))
        self.data.to_tensor(self.device)
        self.model_y = self.model.predict(self.data).cpu()
        self.model_y = self.scaler.inverse_transform(self.model_y)

        self.lam = 0.8
        self.gamma = 1


    def set_computation_graph(self, all_events, edge_feat, target_idx):

        class Event:
            def __init__(self, timestamp, node_idx, value):
                self.timestamp = int(timestamp)
                self.node_idx = int(node_idx)
                self.value = value


        num_hops = 4
        self.all_events = all_events.copy()

        self.node_id_to_idx = pd.unique(self.all_events['u']).tolist()
        self.target_event = Event(self.all_events.iloc[target_idx]['ts'],
                                  self.node_id_to_idx.index(self.all_events.iloc[target_idx]['u']),
                                  self.scaler.transform(self.all_events.iloc[target_idx]['f0']))

        target_timestamp = self.all_events.iloc[target_idx]['ts']
        max_history = target_timestamp - self.input_window
        preceeding_events = self.all_events[self.all_events['ts'] < target_timestamp]
#        preceeding_events = preceeding_events.sort_values(by='ts', ascending=False)
        self.input_window_events = preceeding_events[preceeding_events['ts'] >= max_history].copy()
        self.input_window_events['relative_ts'] = (self.input_window_events['ts'] - max_history)


#        neighbouring_nodes = [self.all_events['u'][target_idx]]  # event_idx is 1-based, so we need to subtract 1
#        for n in range(num_hops):
#            cur_neighbours = []
#            for neighbour in neighbouring_nodes:
#                cur_neighbours.extend(edge_feat['destination'][edge_feat['src'] == neighbour].values)
#            neighbouring_nodes.extend(cur_neighbours)
#        neighbouring_nodes = list(set(neighbouring_nodes))
#        neighbouring_node_events = self.input_window_events[self.input_window_events['u'].isin(neighbouring_nodes)].copy()
# Pre-index edge_feat for fast lookup
        src_to_dst = edge_feat.groupby('src')['destination'].apply(list).to_dict()

# Initialize
        neighbouring_nodes = set([self.all_events['u'][target_idx]])

# Multi-hop expansion
        for _ in range(num_hops):
            new_neighbours = set()
            for node in neighbouring_nodes:
                new_neighbours.update(src_to_dst.get(node, []))
            neighbouring_nodes.update(new_neighbours)

# Filter input_window_events just once
        neighbouring_node_events = self.input_window_events[self.input_window_events['u'].isin(neighbouring_nodes)].copy()
        self.events = []
        for neighbouring_event in neighbouring_node_events.itertuples():
            timestamp = neighbouring_event.relative_ts
            node_idx = self.node_id_to_idx.index(neighbouring_event.u)
            value = self.scaler.transform(neighbouring_event.f0)
            self.events.append(Event(timestamp, node_idx, value))

        self.candidate_events = list(range(len( self.events )))
        unique_e_idx = None

        return self.candidate_events, unique_e_idx

    def load_data(self):
        single_batch_config = copy.deepcopy(self.config)
        single_batch_config['batch_size'] = 1
        dataset = get_dataset(single_batch_config)
        train_data, valid_data, test_data = dataset.get_data()
        self.input_sample_shape = np.array(next(iter(test_data))['X']).shape[1:]
        data_feature = dataset.get_data_feature()

        return data_feature, train_data, valid_data, test_data


    def load_model(self, model_path, data_feature):

        model = get_model(self.config, data_feature)
        model_state, optimizer_state = torch.load(model_path, map_location=self.device)
        model.load_state_dict(model_state)
        model = model.to(self.device)

        return model


    def generate_masked_input(self, exp_events):
        import copy
        masked_input = copy.deepcopy(self.data)
        for event_idx in self.candidate_events:
            if event_idx not in exp_events:
                event_timestamp, event_node_idx = self.events[event_idx].timestamp, self.events[event_idx].node_idx
                masked_input['X'][0, event_timestamp, event_node_idx, :] = 0
#        return self.data
        return masked_input

    def batch_to_cpu(self, batch):
        for key in batch.data:
            batch.data[key] = batch.data[key].to('cpu')


    def exp_prediction(self, exp_events):
        exp_masked_input = self.generate_masked_input(exp_events)
        self.best_exp_graph = exp_masked_input
#        exp_masked_input.to_tensor(self.device)
        exp_y = self.model.predict(exp_masked_input).cpu()
        exp_y = self.scaler.inverse_transform(exp_y)

        return exp_y

    def delta_fidelity(self, exp_events):
        exp_y = self.exp_prediction(exp_events)
        target_exp_y = exp_y[0, 0, self.target_event.node_idx, 0]

        target_model_y = self.model_y[0, 0, self.target_event.node_idx, 0]

        complement_events = [node for node in self.candidate_events if node not in exp_events]
        complement_masked_input = self.generate_masked_input(complement_events)
#        complement_masked_input.to_tensor(self.device)
        complement_exp_y = self.model.predict(complement_masked_input).cpu()
        complement_exp_y = self.scaler.inverse_transform(complement_exp_y)
        target_complement_exp_y = complement_exp_y[0, 0, self.target_event.node_idx, 0]

        if abs(target_exp_y - target_model_y) == 0:
            delta_fidelity = np.inf
        else:
            delta_fidelity = abs(target_complement_exp_y - target_model_y)/abs(target_exp_y - target_model_y)

#        print('Model Pred: ', target_model_y, 'Exp Pred: ', target_exp_y, 'Complement Pred: ', target_complement_exp_y, "Delta fidelity: ", delta_fidelity)

        return delta_fidelity, target_complement_exp_y, target_model_y, exp_y, target_exp_y


    def tgnne_score_func(self, exp_events):
        '''
        Calculate the fidelity of the model for a given subgraph. Fidelity is defined using the
        metric proposed in arXiv:2306.05760.

        Args:
            subgraph: A list of graphNode() objects that are part of the computation graph.

        Returns:
            fidelity: The fidelity of the model for the given subgraph.
        '''
        exp_y = self.exp_prediction(exp_events)
        target_exp_y = exp_y[0, 0, self.target_event.node_idx, 0]

        target_model_y = self.model_y[0, 0, self.target_event.node_idx, 0]

        fidelity_score = abs(target_model_y - target_exp_y)

        return fidelity_score

    def pg_ext_pred(self, mask_weights):
        # target_model_y should be a tensor and requires_grad=False is fine
        target_model_y = self.model_y[0, 0, self.target_event.node_idx, 0]

        input = copy.deepcopy(self.data)
        input['X'] = input['X'].clone().to(mask_weights.device)  # Ensure tensor on correct device

        for e, event in enumerate(self.candidate_events):
            event_ts = self.events[event].timestamp
            event_node_idx = self.events[event].node_idx
            event_mask_weight = mask_weights[e]

            input['X'][0, event_ts, event_node_idx, :] *= event_mask_weight

        # Forward pass
        masked_prediction = self.model(input)  # assumes model returns torch.Tensor
        target_masked_exp_y = masked_prediction[0, 0, self.target_event.node_idx, 0]

#        fidelity_score = torch.abs(target_model_y - target_masked_exp_y) * torch.sum(mask_weights)
        fidelity_score = torch.abs(target_model_y - target_masked_exp_y)

#        print('----------------')
#        print('LOSS: ', fidelity_score.item(), 'Model Pred: ', target_model_y.item(), 'Masked Pred: ', target_masked_exp_y.item())
#        print('----------------')

        return fidelity_score


