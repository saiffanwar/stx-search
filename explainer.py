import numpy as np
import torch
import math
import os
import random
from tqdm import tqdm
from copy import deepcopy
from matplotlib import pyplot as plt
from pprint import pprint
import seaborn as sns
import pickle as pck

from libcity.data import get_dataset
from libcity.utils import get_model
from libcity.config import ConfigParser

from visualisation_utils import graph_visualiser
from mcts import MCTS
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, default='generate', help='Mode of operation: generate or visualise')
args = parser.parse_args()

def ncr(n, r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)


def flatten(l):
    return [item for sublist in l for item in sublist]

class graphNode:
    def __init__(self,node_index, timestamp, speed):
        self.node_index = node_index
        self.timestamp = timestamp
        self.speed = speed


class Explainer:

    def __init__(self, target_node_index=20, target_timestamp=0):
        self.task='traffic_state_pred'
        self.model_name='STGCN'
        self.dataset_name='METR_LA'
        self.config_file=None
        self.saved_model=True
        self.train=False
        self.other_args={'exp_id': '1', 'seed': 0}
        self.config = ConfigParser(self.task, self.model_name, self.dataset_name, self.config_file, self.saved_model, self.train, self.other_args)
#        self.device = self.config.get('device', torch.device('cpu'))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.input_window = self.config.get('input_window', 3)
        self.output_window = self.config.get('output_window', 1)
        self.target_index = 20
        self.target_timestamp = 0

    def generate_graph_nodes(self, x):
        x = x.cpu()
        window_size = np.array(x).shape[1]
        num_nodes = np.array(x).shape[2]
        nodes = [[] for i in range(window_size)]

        for t in range(window_size):
            for n in range(num_nodes):
                nodes[t].append(graphNode(n, t, x[0][t][n][0]))
        return nodes


    def load_data(self):

        dataset = get_dataset(self.config)
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

    def calculate_target_error(self, target_node, truth, pred):
        '''
        Calculate the absolute error of the target node.

        Args:
            target_node: graphNode() object of the target node.
            truth: The ground truth values of the batch (batch['y']) of size (batch_size, output_window, num_nodes, feature_dim).
            pred: The predicted values of the batch (batch['y']) of size (batch_size, output_window, num_nodes, feature_dim).

        Returns:
            error: The absolute error of the target node.
        '''

        truth = self.scaler.inverse_transform(truth.cpu())
        pred = self.scaler.inverse_transform(pred.cpu())
        error = torch.abs(truth[0][self.target_timestamp][self.target_index][0] - pred[0][self.target_timestamp][self.target_index][0])

        return error


    def fetch_computation_graph(self, x, adj_mx, target_node, spatial_neighbourhood_size=2, temporal_neighbourhood_size=3):
        '''
        --- Currently implemented for STGCN model ---
        This function fetches the set of events that the model uses to make a prediction.

        Args:
            x: The input tensor (batch_size, num_nodes, num_timesteps, num_features). For METR_LA (batch_size, 207, 12, 1)
            adj_mx: The adjacency matrix (num_nodes, num_nodes). For METR_LA (207, 207)
            target_node: graphNode() object of the target for which the computation graph is to be fetched.
            spatial_neighbourhood_size: The size of the spatial neighbourhood. Default is 2.
            temporal_neighbourhood_size: The size of the temporal neighbourhood. Default is 3.

        Returns:
            candidate_events: A list of graphNode() objects that are part of the computation graph.
        '''
        # Take from 1st index because the 0th will always be itself. The adjacency matrix is symmetric.
        node_adjacency = adj_mx[target_node.node_index][1:]
        immediate_spatial_neighbours = list((np.argwhere(node_adjacency > 0)).flatten())
        immediate_spatial_neighbours = list(range(0, 207))
#        immediate_temporal_neighbours = []

        N_hop_spatial_neighbours = [immediate_spatial_neighbours]
        for immediate_neighbour in immediate_spatial_neighbours:
            N_hop_spatial_neighbours.append(list((np.argwhere(adj_mx[immediate_neighbour][1:] > 0)).flatten()))

        two_hop_neighbours = list(set(sorted(flatten(N_hop_spatial_neighbours))))
        candidate_events = []
        for t in range(self.input_window):
            for n in two_hop_neighbours:
                candidate_events.append(self.all_nodes[t][n])
        return candidate_events

    def create_masked_input(self, subgraph_nodes, x, adj_mx):
        masked_adj_mx = torch.zeros(np.array(adj_mx).shape)
        masked_input = torch.zeros(np.array(x).shape)
        all_zeros = torch.zeros(np.array(x).shape)

        min_value = dir(self.scaler)

        subgraph_node_details = [[node.timestamp, node.node_index] for node in subgraph_nodes]
        inverse_scaled_input = self.scaler.inverse_transform(x[0])
        all_nodes = [[timestamp, index] for timestamp in range(self.input_window) for index in range(masked_input.shape[2])]
        for node in all_nodes:
            if node in subgraph_node_details:
                masked_adj_mx[node[1]][node[1]] = 1
                masked_input[0][node[0]][node[1]] = torch.FloatTensor(x[0][node[0]][node[1]])
#            else:
#                masked_input[0][node[0]][node[1]] = torch.FloatTensor([(-self.scaler.mean)/self.scaler.std])
        masked_adj_mx = adj_mx

        for t in range(self.input_window):
            timestamp_nodes = [s.node_index for s in subgraph_nodes if s.timestamp == t]

        return masked_input, masked_adj_mx

#    def calculate_fidelity(self, subgraph, batch, model):

    def exp_fidelity(self, exp_nodes):
        '''
        Calculate the fidelity of the model for a given subgraph. Fidelity is defined using the
        metric proposed in arXiv:2306.05760.

        Args:
            subgraph: A list of graphNode() objects that are part of the computation graph.

        Returns:
            fidelity: The fidelity of the model for the given subgraph.
        '''
        x = self.batch['X'].cpu()
        y = self.batch['y'].cpu()
        adj_mx = self.adj_mx
        explanation_graph, adj_mx = self.create_masked_input(exp_nodes, x, adj_mx)

        unimportant_nodes = [node for node in np.array(self.all_nodes).flatten() if node not in exp_nodes]
        non_explanation_graph, adj_mx = self.create_masked_input(unimportant_nodes, x, adj_mx)


        explanation_y = self.make_prediction_from_masked_input(explanation_graph, self.batch)
#        self.data_graph_visualisation(batch['X'].cpu(), explanation_graph, self.model_y, explanation_y.cpu(), adj_mx)
        non_explanation_y = self.make_prediction_from_masked_input(non_explanation_graph, self.batch)

        explanation_error = self.calculate_target_error(self.target_node, self.model_y, explanation_y)
        non_explanation_error = self.calculate_target_error(self.target_node, self.model_y, non_explanation_y)


        return explanation_error
#        if (explanation_error - non_explanation_error) == 0:
#            return np.inf
#        else:
#            fidelity =  1/(explanation_error - non_explanation_error)
##        print(explanation_error, non_explanation_error, fidelity)
#        return fidelity.float().numpy()


    def make_prediction_from_masked_input(self, masked_input, batch):

        masked_batch = deepcopy(batch)
        masked_batch['X'] = masked_input.cpu() # (1, 12, 207, 1)

        masked_batch.to_tensor(self.device)
#        loss = model.calculate_loss(masked_batch)
        y = self.model.predict(masked_batch) # (1, 12, 207, 1)

        return y

def batch_to_cpu(batch):
    for key in batch.data:
        batch.data[key] = batch.data[key].to('cpu')

def run_explainer():

    explainer = Explainer(target_node_index=20, target_timestamp=0)
    data_feature, train_data, valid_data, test_data = explainer.load_data()
    explainer.scaler = data_feature['scaler']

    model_path = os.getcwd()+'/libcity/cache/1/model_cache/STGCN_METR_LA.m'
    explainer.model = explainer.load_model(model_path, data_feature)
    adj_mx = data_feature['adj_mx']
    explainer.adj_mx = adj_mx


    with torch.no_grad():
        explainer.model.eval()
        batch = next(iter(test_data))
#            print(torch.tensor(batch['X']).shape)
#            print(dir(batch))
        batch['X'] = batch['X'][:1]
        batch['y'] = batch['y'][:1]


        batch.to_tensor(explainer.device)
        explainer.model_y = explainer.model.predict(batch)
        explainer.model_y.cpu()

        batch_to_cpu(batch)

        explainer.all_nodes = explainer.generate_graph_nodes(batch['X'])

        explainer.target_node = graphNode(explainer.target_index, explainer.target_timestamp, batch['y'][0][explainer.target_timestamp][explainer.target_index][0])

        if args.mode == 'generate':
            explainer.candidate_events = explainer.fetch_computation_graph(batch['X'], adj_mx, explainer.target_node)
            subgraph_sizes = [5,10,25,50,100,int(np.floor(len(explainer.candidate_events)/4)),int(np.floor(len(explainer.candidate_events)/2)),len(explainer.candidate_events)]
            subgraph_size = 50
#            random.seed(0)
            subgraph = random.sample(explainer.candidate_events, subgraph_size)
            explainer.batch = batch
            return explainer
#            mcts = MCTS(explainer)

#            mcts.tree_search_animation()
#            exp_subgraph = mcts.run_mcts(candidate_events=explainer.candidate_events, batch=batch)
#            mcts.self_play(candidate_events=explainer.candidate_events)

        elif args.mode == 'visualise':

            with open('results/METR_LA/best_exp.pck', 'rb') as f:
                exp_subgraph = pck.load(f)
            masked_input, masked_adj_mx = explainer.create_masked_input(exp_subgraph, batch['X'], explainer.adj_mx)
            masked_batch = deepcopy(batch)
            masked_batch['X'] = masked_input # (1, 12, 207, 1)
            masked_batch.to_tensor(explainer.device)
            exp_y = explainer.model.predict(masked_batch) # (1, 12, 207, 1)
            ### Inverse scaling of output to get traffic speed values.
            y_predicted = explainer.scaler.inverse_transform(exp_y[..., :explainer.output_window]) # (1, 12, 207, 1)
            graph_visualiser(explainer, batch['X'].cpu() ,masked_input, explainer.model_y.cpu(), exp_y.cpu(), adj_mx)


run_explainer()


