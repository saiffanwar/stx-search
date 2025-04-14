import numpy as np
import torch
import dill as pck
import os
import argparse
import copy
from matplotlib import pyplot as plt
import random


from sa_explainer import SimulatedAnnealing

from libcity.data import get_dataset
from libcity.utils import get_model
from libcity.config import ConfigParser


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default='TGCN', help='Mode of operation: generate or visualise')
parser.add_argument('-t', '--target_node', type=int, default=12, help='Target node index for explanation')
parser.add_argument('-s', '--subgraph_size', type=int, default=50, help='Size of the subgraph for explanation')
parser.add_argument('-d', '--dataset', type=str, default='METR_LA', help='Dataset name')
args = parser.parse_args()


class STX_Search_LibCity:

    def __init__(self):
        self.task='traffic_state_pred'
        self.model_name=args.model
        self.dataset_name=args.dataset
        self.config_file=None
        self.saved_model=True
        self.train=False
        self.other_args={'exp_id': '1', 'seed': 0}
        self.config = ConfigParser(self.task, self.model_name, self.dataset_name, self.config_file, self.saved_model, self.train, self.other_args)
        self.config['weight_adj_epsilon'] = 0.000001
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

        self.lam = 0.9
        self.gamma = 0.1

    def graph_to_events(self):

        class Event:
            def __init__(self, timestamp, node_idx, value):
                self.timestamp = timestamp
                self.node_idx = node_idx
                self.value = value

        rescaled_data = self.scaler.inverse_transform(self.data['X'])

        self.events = []
        for timestamp in range(self.input_window):
            for node in range(self.data['X'].shape[2]):
                self.events.append(Event(timestamp, node, rescaled_data[0, timestamp, node, 0].detach().cpu().numpy()))

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

    def set_computation_graph(self, target_idx):
        candidate_events = [target_idx]


        for neighbour in candidate_events:
            candidate_events.extend(list(np.argwhere(self.adj_mx[neighbour] > 0).flatten()))
            candidate_events = list(set(candidate_events))

#        self.candidate_events = list(set(candidate_events))
        self.candidate_events = list(range(len(self.events)))

        return candidate_events

    def exp_prediction(self, exp_events):
        exp_masked_input = self.generate_masked_input(exp_events)
        self.best_exp_graph = exp_masked_input
#        exp_masked_input.to_tensor(self.device)
        exp_y = self.model.predict(exp_masked_input).cpu()
        exp_y = self.scaler.inverse_transform(exp_y)

        return exp_y

    def delta_fidelity(self, exp_events, target_node):
        exp_y = self.exp_prediction(exp_events)
        target_exp_y = exp_y[0, 0, target_node, 0]

        target_model_y = self.model_y[0, 0, target_node, 0]

        complement_events = [node for node in self.candidate_events if node not in exp_events]
        complement_masked_input = self.generate_masked_input(complement_events)
#        complement_masked_input.to_tensor(self.device)
        complement_exp_y = self.model.predict(complement_masked_input).cpu()
        complement_exp_y = self.scaler.inverse_transform(complement_exp_y)
        target_complement_exp_y = complement_exp_y[0, 0, target_node, 0]

        if abs(target_exp_y - target_model_y) == 0:
            delta_fidelity = np.inf
        else:
            delta_fidelity = abs(target_complement_exp_y - target_model_y)/abs(target_exp_y - target_model_y)

#        print('Model Pred: ', target_model_y, 'Exp Pred: ', target_exp_y, 'Complement Pred: ', target_complement_exp_y, "Delta fidelity: ", delta_fidelity)

        return delta_fidelity, target_complement_exp_y, target_model_y, target_exp_y


    def score_func(self, exp_events, target_node, mode='fidelity', d_gam=0.9, d_lam=0.1, max_delta_fidelity=None):
        '''
        Calculate the fidelity of the model for a given subgraph. Fidelity is defined using the
        metric proposed in arXiv:2306.05760.

        Args:
            subgraph: A list of graphNode() objects that are part of the computation graph.

        Returns:
            fidelity: The fidelity of the model for the given subgraph.
        '''
        delta_fidelity, target_complement_exp_y, target_model_y, target_exp_y  = self.delta_fidelity(exp_events, target_node)

        exp_absolute_error = abs(target_model_y - target_exp_y)
        max_exp_size = self.subgraph_size
        exp_size_percentage = 100*len(exp_events)/max_exp_size
        exp_percentage_error = 100*abs(exp_absolute_error/target_model_y)

        if mode == 'fidelity':
#            exp_score = exp_percentage_error
            exp_score = exp_absolute_error
#            exp_score = self.delta_fidelity(exp_events, self.target_index)
        elif mode == 'delta_fidelity':
            exp_score = delta_fidelity
        elif mode == 'fidelity+size':
            lam = self.lam
            gam = self.gamma
            exp_score = gam*exp_percentage_error + lam*exp_size_percentage
#            exp_score =  (delta_fidelity/max_delta_fidelity)+ lam*exp_size_percentage
        else:
            RuntimeError('Invalid mode. Choose either fidelity or fidelity+size')
        return exp_score, exp_absolute_error, target_model_y, target_exp_y, target_complement_exp_y


def plot_explanation(coords, target, exp):
    fig, ax = plt.subplots()
    ax.scatter(coords[:, 0], coords[:, 1], color='black', alpha=0.1)
    ax.scatter(coords[exp, 0], coords[exp, 1], color='g')
    ax.scatter(coords[target, 0], coords[target, 1], color='r')
    plt.show()


if __name__ == '__main__':

    with torch.no_grad():
        explainer = STX_Search_LibCity()

        explainer.graph_to_events()
        explainer.set_computation_graph(args.target_node)

        num_iter = 10000
        exp_size = args.subgraph_size
        explainer.subgraph_size = exp_size

        sa = SimulatedAnnealing(args.dataset, args.target_node, explainer.candidate_events, exp_size, score_func=explainer.score_func, verbose=True)
        score, exp, model_pred, exp_pred = sa.run(iterations=num_iter, expmode='fidelity', explainer=explainer)
        grid_size = int(len(explainer.adj_mx)**0.5)
        coords = [[i, j] for i in range(grid_size) for j in range(grid_size)]
        coords = np.array(coords)
#        plot_explanation(coords, args.target_node, exp)

        with open(f'results/{args.dataset}/best_result_{args.target_node}_{exp_size}_fidelity.pck', 'wb') as f:
            pck.dump([explainer, sa], f)
#        score, exp, model_pred, exp_pred = sa.run(iterations=num_iter, expmode='fidelity+size', best_events=exp)
#        if exp_size == 50:
#            g_score, g_exp, g_model_pred, g_exp_pred = copy.copy(score), copy.copy(exp), copy.copy(model_pred), copy.copy(exp_pred)



#        explainer.all_nodes = explainer.generate_graph_nodes(batch['X'])
#
#        explainer.target_node = graphNode(explainer.target_index, explainer.target_timestamp, batch['y'][0][explainer.target_timestamp][explainer.target_index][0])
