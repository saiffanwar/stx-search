import numpy as np
import torch
import pickle as pck
import argparse
import copy
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sa_explainer import SimulatedAnnealing

from libcity.data import get_dataset
from libcity.utils import get_model
from libcity.config import ConfigParser


class STX_Search_LibCity:

    def __init__(self, model_name, dataset_name, all_events):
        self.task = 'traffic_state_pred'
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.config_file = None
        self.saved_model = True
        self.train = False
        self.other_args = {'exp_id': '1', 'seed': 0}
        self.config = ConfigParser(self.task, self.model_name, self.dataset_name,
                                   self.config_file, self.saved_model, self.train, self.other_args)
        self.config['weight_adj_epsilon'] = 0.000003
#        self.device = self.config.get('device', torch.device('cpu'))
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.input_window = self.config.get('input_window', 3)
        self.output_window = self.config.get('output_window', 1)

        self.data_feature, self.train_data, self.valid_data, self.test_data = self.load_data()
        self.data = next(iter(self.train_data))
        self.data.to_tensor(self.device)
        self.scaler = self.data_feature['scaler']
        model_path = os.getcwd() + \
            f'/libcity/cache/1/model_cache/{self.model_name}_{self.dataset_name}.m'
        self.model = self.load_model(model_path, self.data_feature)
        self.adj_mx = self.data_feature['adj_mx']
        self.all_events = all_events.copy()

        self.lam = 0.8
        self.gamma = 1

    class Event:
        def __init__(self, timestamp, node_idx, scaled_value, original_value, event_idx=None):
            '''
            Args:
                timestamp (int): The timestamp of the event.
                node_idx (int): The index of the node where the event occurred.
                value (float): The value associated with the event.
                event_index (int, optional): The index of the event in the list of events where there are total num_ts*num_nodes events.
            '''
            self.timestamp = timestamp
            self.node_idx = node_idx
            self.event_idx = event_idx
            self.scaled_value = scaled_value
            self.original_value = original_value

    def _initialize(self, explaining_event_idx, exp_size):
        print('Initializing STX Search Explainer...')
        self.exp_size = exp_size
        with open(f'scaler.pck', 'wb') as f:
            pck.dump(self.scaler, f)
        self.construct_input_data_from_event(explaining_event_idx)
        # self.graph_to_events()
        self.set_computation_graph()

    def construct_input_data_from_event(self, explaining_event_idx):
        self.explaining_event_idx = explaining_event_idx
        self.explaining_event = self.all_events.iloc[explaining_event_idx]
        self.node_id_to_idx = pd.unique(self.all_events['u']).tolist()

        self.target_event = self.Event(timestamp = self.explaining_event['ts'],
                                       node_idx = self.node_id_to_idx.index(self.explaining_event['u']),
                                       scaled_value = self.scaler.transform(self.explaining_event['f0']),
                                       original_value = self.explaining_event['f0'],
                                       event_idx = explaining_event_idx)
        self.target_idx = self.target_event.node_idx

        target_timestamp = self.all_events.iloc[explaining_event_idx]['ts']
        max_history = target_timestamp - self.input_window

        preceeding_events = self.all_events[self.all_events['ts']
                                            < target_timestamp]
#        preceeding_events = preceeding_events.sort_values(by='ts', ascending=False)
        self.input_window_events = preceeding_events[preceeding_events['ts'] >= max_history].copy()
        self.input_window_events['relative_ts'] = (
            self.input_window_events['ts'] - max_history)
        print(self.input_window_events.head(25))
        print('Input window events shape: ', self.input_window_events.shape)


        self.events = []
        for i, event in enumerate(self.input_window_events.itertuples()):
            timestamp = int(event.relative_ts)
            node_idx = int(self.node_id_to_idx.index(event.u))
            original_value = event.f0
            scaled_value = self.scaler.transform(event.f0)
            self.events.append(self.Event(
                                          timestamp=timestamp, 
                                          node_idx=node_idx, 
                                          scaled_value=scaled_value, 
                                          original_value=original_value, 
                                          event_idx=i)
                                        )        
            self.data['X'][0, timestamp, node_idx, :] = scaled_value
            # self.data['X'][0, int(event.relative_ts), self.node_id_to_idx.index(
                # event.u), :] = self.scaler.transform(event.f0)
        print(len(self.events), ' events in the input window')
        self.model_y = self.model.predict(self.data).cpu()
        self.model_y = self.scaler.inverse_transform(self.model_y)
        self.target_model_y = self.model_y[0, 0, self.target_idx, 0]

    # def graph_to_events(self):
    #
    #     # rescaled_data = self.scaler.inverse_transform(self.data['X'])
    #
    #     self.events = []
    #
    #     # Create events from the input data
    #     for i, event in enumerate(self.input_window_events.itertuples()):
    #         timestamp = int(event.relative_ts)
    #         node_idx = self.node_id_to_idx.index(event.u)
    #         value = self.scaler.transform(event.f0)
    #         self.events.append(self.Event(
    #             timestamp=timestamp, node_idx=node_idx, value=value, event_idx=i))

    def load_data(self):
        single_batch_config = copy.deepcopy(self.config)
        single_batch_config['batch_size'] = 1
        dataset = get_dataset(single_batch_config)
        train_data, valid_data, test_data = dataset.get_data()
        self.input_sample_shape = np.array(
            next(iter(test_data))['X']).shape[1:]
        data_feature = dataset.get_data_feature()

        return data_feature, train_data, valid_data, test_data

    def load_model(self, model_path, data_feature):

        model = get_model(self.config, data_feature)
        model_state, optimizer_state = torch.load(
            model_path, map_location=self.device)
        model.load_state_dict(model_state)
        model = model.to(self.device)

        return model

    def generate_masked_input(self, exp_events):
        import copy
        masked_input = copy.deepcopy(self.data)
        masked_input['X'] = torch.zeros_like(masked_input['X'])
        for event_idx in exp_events:
            event_timestamp, event_node_idx = self.events[
                event_idx].timestamp, self.events[event_idx].node_idx
            masked_input['X'][0, event_timestamp, event_node_idx, :] = self.events[event_idx].scaled_value

        return masked_input


    def batch_to_cpu(self, batch):
        for key in batch.data:
            batch.data[key] = batch.data[key].to('cpu')

    def set_computation_graph(self):
        n_hop_neighbourhood = [self.target_event.node_idx]
        num_hops = 2
        print(np.argwhere(self.adj_mx[0] > 0).flatten())
        for n in range(num_hops):
            new_neighbours = []
            for neighbour in n_hop_neighbourhood:
                new_neighbours.extend(list(np.argwhere(self.adj_mx[neighbour] > 0).flatten()))

            n_hop_neighbourhood.extend(new_neighbours)
            n_hop_neighbourhood = list(set(n_hop_neighbourhood))
        print(n_hop_neighbourhood)

        self.candidate_events = [
            e.event_idx for e in self.events if e.node_idx in n_hop_neighbourhood]
        print(len(self.candidate_events), ' candidate events found for the target node: ',
              self.target_event.node_idx, ' within ', num_hops, ' hops')

        return self.candidate_events

    def exp_prediction(self, exp_events):
        exp_masked_input = self.generate_masked_input(exp_events)
#        exp_masked_input.to_tensor(self.device)
        exp_y = self.model.predict(exp_masked_input).cpu()
        exp_y = self.scaler.inverse_transform(exp_y)

        return exp_y

    def delta_fidelity(self, exp_events):
        exp_y = self.exp_prediction(exp_events)
        target_exp_y = exp_y[0, 0, self.target_idx, 0]


        complement_events = [
            node for node in self.candidate_events if node not in exp_events]
        complement_masked_input = self.generate_masked_input(complement_events)
#        complement_masked_input.to_tensor(self.device)
        complement_exp_y = self.model.predict(complement_masked_input).cpu()
        complement_exp_y = self.scaler.inverse_transform(complement_exp_y)
        target_complement_exp_y = complement_exp_y[0, 0, self.target_idx, 0]

        if abs(target_exp_y - self.target_model_y) == 0:
            delta_fidelity = np.inf
        else:
            delta_fidelity = abs(
                target_complement_exp_y - self.target_model_y)/abs(target_exp_y - self.target_model_y)

        # print('Model Pred: ', self.target_model_y, 'Exp Pred: ', target_exp_y, 'Complement Pred: ', target_complement_exp_y, "Delta fidelity: ", delta_fidelity)

        return delta_fidelity, target_complement_exp_y, self.target_model_y, exp_y, target_exp_y

    def score_func(self, exp_events):
        '''
        Calculate the fidelity of the model for a given subgraph. Fidelity is defined using the
        metric proposed in arXiv:2306.05760.

        Args:
            subgraph: A list of graphNode() objects that are part of the computation graph.

        Returns:
            fidelity: The fidelity of the model for the given subgraph.
        '''
        delta_fidelity, target_complement_exp_y, self.target_model_y, exp_y, target_exp_y = self.delta_fidelity(
            exp_events)

        exp_absolute_error = abs(self.target_model_y - target_exp_y)
        max_exp_size = self.exp_size**2
        exp_size_percentage = 100*len(exp_events)/max_exp_size
        exp_percentage_error = 100*abs(exp_absolute_error/self.target_model_y)

        lam = self.lam
        gam = self.gamma
        fidelity_size_score = gam*exp_percentage_error + lam*exp_size_percentage
#        print(exp_percentage_error, exp_size_percentage, fidelity_size_score)

        return exp_absolute_error, fidelity_size_score, delta_fidelity, self.model_y, self.target_model_y, exp_y, target_exp_y, target_complement_exp_y

    def explain(self, explaining_event_idx, exp_size=20, mode='fidelity', num_iter=10000):

        self._initialize(explaining_event_idx, exp_size)
        if self.dataset_name == 'PEMS_BAY' and self.explaining_event['u'] in [400001, 400017, 400240, 400560, 400677, 407202]:
            return None, None, None, None
        # print(self.candidate_events)
        sa = SimulatedAnnealing(self.data['X'].detach().cpu().numpy(), self.events, self.adj_mx, self.model_name, self.dataset_name,
                                self.target_idx, self.explaining_event_idx, self.candidate_events, exp_size, score_func=self.score_func, verbose=True)
#        score, exp, model_pred, exp_pred = sa.run(iterations=num_iter, expmode='fidelity')
        if mode != 'both':
            score, exp, model_pred, exp_pred = sa.run(
                iterations=num_iter, expmode=mode)
        else:
            score, exp, model_pred, exp_pred = sa.run(
                iterations=num_iter, expmode='fidelity', best_events=self.candidate_events, exp_size=exp_size)
            score, exp, model_pred, exp_pred = sa.run(
                iterations=num_iter, expmode='fidelity+size', best_events=exp, )

        return score, exp, model_pred, exp_pred
