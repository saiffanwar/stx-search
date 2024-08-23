import numpy as np
import torch
import math
import os
import random
from tqdm import tqdm
from copy import deepcopy
from matplotlib import pyplot as plt

from libcity.data import get_dataset
from libcity.utils import get_model
from libcity.config import ConfigParser

def ncr(n, r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)


def flatten(l):
    return [item for sublist in l for item in sublist]

class graphNode():
    def __init__(self,node_index, timestamp, speed):
        self.node_index = node_index
        self.timestamp = timestamp
        self.speed = speed


class Explainer():

    def __init__(self):
        self.task='traffic_state_pred'
        self.model_name='STGCN'
        self.dataset_name='METR_LA'
        self.config_file=None
        self.saved_model=True
        self.train=False
        self.other_args={'exp_id': '1', 'seed': 0}
        self.config = ConfigParser(self.task, self.model_name, self.dataset_name, self.config_file, self.saved_model, self.train, self.other_args)
        self.device = self.config.get('device', torch.device('cpu'))
        self.input_window = self.config.get('input_window', 3)
        self.target_index = 50
        self.target_timestamp = 0



    def load_data(self):

        dataset = get_dataset(self.config)
        train_data, valid_data, test_data = dataset.get_data()
        data_feature = dataset.get_data_feature()
        return data_feature, train_data, valid_data, test_data


    def load_model(self, model_path, data_feature):

        model = get_model(self.config, data_feature)
        model_state, optimizer_state = torch.load(model_path)
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

        error = torch.abs(target_node.speed - pred[0][self.target_timestamp][self.target_index][0])

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
                candidate_events.append(graphNode(n, t, x[0][t][n][0]))
        return candidate_events

    def create_masked_input(self, subgraph_nodes, x):
        masked_input = torch.zeros(np.array(x).shape)

        for node in subgraph_nodes:
            masked_input[0][node.timestamp][node.node_index] = torch.FloatTensor(x[0][node.timestamp][node.node_index])

        return masked_input

    def calculate_fidelity(self, subgraph, batch, model):




    def main(self):

        data_feature, train_data, valid_data, test_data = self.load_data()
        model_path = os.getcwd()+'/libcity/cache/1/model_cache/STGCN_METR_LA.m'
        model = self.load_model(model_path, data_feature)
        adj_mx = data_feature['adj_mx']


        with torch.no_grad():
            model.eval()
            batch = next(iter(test_data))
#            print(torch.tensor(batch['X']).shape)
#            print(dir(batch))
            batch['X'] = batch['X'][:1]
            batch['y'] = batch['y'][:1]
            target_node = graphNode(self.target_index, self.target_timestamp, batch['y'][0][self.target_timestamp][self.target_index][0])

            self.candidate_events = self.fetch_computation_graph(batch['X'], adj_mx, target_node)

            fig, axes = plt.subplots(4,2, figsize=(10,10))

            subgraph_sizes = [5,10,25,50,100,int(np.floor(len(self.candidate_events)/4)),int(np.floor(len(self.candidate_events)/2)),len(self.candidate_events)]
            all_errors = {s: [] for s in subgraph_sizes}
            for subgraph_size, ax in zip(subgraph_sizes, fig.get_axes()):
                achieved_errors = []
                for i in tqdm(range(100)):
                    subgraph = random.sample(self.candidate_events, subgraph_size)
#            subgraph = self.candidate_events

                    masked_input = self.create_masked_input(subgraph, batch['X'])
#                batch['X'] = masked_input
                    masked_batch = deepcopy(batch)
                    masked_batch['X'] = masked_input

                    masked_batch.to_tensor(self.device)
                    loss = model.calculate_loss(masked_batch)
                    output = model.predict(masked_batch)
                    error = self.calculate_target_error(target_node, masked_batch['y'], output)
                    achieved_errors.append(error.item())
                all_errors[subgraph_size] = achieved_errors
#                len(self.candidate_events)\\4, (len(self.candidate_events)\\2), len(self.candidate_events)
                ax.hist(achieved_errors, bins=[0.01*i for i in range(0, 100)])

                print('Mean error: ', np.mean(achieved_errors))
#        ax.set_xlabel('Error')
        fig.savefig('error_histograms.png')
        plt.show()








exp = Explainer()
exp.main()

