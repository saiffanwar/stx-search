import torch
from mcts import MCTS
import pickle as pck
import numpy as np
import matplotlib.pyplot as plt
import os

from torch import nn
import torch.nn.functional as F
from torchsummary import summary
from explainer import run_explainer
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp
import concurrent.futures


import warnings
warnings.filterwarnings("ignore")

results_dir = 'results/METR_LA/'

class AlphaZero:
    '''
    Class to run the AlphaZero algorithm
    '''
    def __init__(self, device, explainer):
        '''
        Args:
            model: nn.Module() object
            device: torch.device() object
            explainer: explainer object needed to make predictions for evaluation explanation fidelity.
        '''
        self.device = device
        self.explainer = explainer
        self.num_workers = 10
        self.input_window = 12
        self.num_nodes = 207
        self.feature_dim = 1
        self.model = AlphaZeroModel(input_window=self.input_window, num_nodes=self.num_nodes, feature_dim=self.feature_dim).to(device)
        self.mcts = MCTS(self.explainer, self.model, expansion_protocol = 'single_child')
        self.num_simulations=5000
        self.depth = 1


    def self_play(self, worker_num=1, round=1):
        print(f'Starting worker {worker_num}')
        x_train = []
        action_probs_ys = []
        vals_ys = []

        all_paths = []

        # Resest the tree before self play. avoids magnifying random biases
        mcts = MCTS(self.explainer, self.model, expansion_protocol='single_child')
        current_node = mcts.root


        for i in range(self.depth):
            print(f'Worker {worker_num} on depth {i}')
            action_probs, reward, paths = mcts.search(current_node, self.num_simulations)
            [all_paths.append(p) for p in paths]
#            plt.bar(range(len(action_probs)), action_probs)
#            with open('results/probabilities.pck', 'wb') as f:
#                pck.dump(action_probs, f)
#                f.close()
#            plt.savefig(f'results/METR_LA/action_probs_{worker_num}.png')
#            plt.close()

#            mcts.visualise_exp_evolution(all_paths)
            x_train.append(mcts.node_to_event_matrix(current_node))

#            action_probs = mcts.generate_probability_matrix_for_children(current_node)

            action_probs_ys.append(action_probs)
            vals_ys.append(reward)
            action = None
            while action not in current_node.taken_actions:
                action = np.random.choice(range(len(action_probs)), p=action_probs)
            current_node = current_node.children[current_node.taken_actions.index(action)]


            # Select the next node based on the action probabilities
#            selected_child = np.random.choice(list(range(len(action_probs))), p=action_probs)
#            if mcts.expansion_protocol == 'single_child':
#                current_node = mcts.expansion(current_node, action_probs)
#            else:
#                current_node = np.random.choice(current_node.children, p=action_probs)


#            print(f'Size of tree: {len(self.mcts.node_ids)}')

        print(f'{ results_dir }{self.num_simulations}/{str(mcts.selection_policy[0])[-1]}training_data_worker_{worker_num}_{round}.pck', 'wb')

        if not os.path.exists(results_dir+f'{self.num_simulations}/'):
            os.makedirs(results_dir+f'{self.num_simulations}/')
#        print(results_dir+f'/{num_simulations}/{mcts.selection_policy[0]}training_data_worker_{worker_num}.pck')
        with open(f'{ results_dir }{self.num_simulations}/{str(mcts.selection_policy[0])[-1]}training_data_worker_{worker_num}_{round}.pck', 'wb') as file:
            pck.dump([np.array(x_train), np.array(action_probs_ys), np.array(vals_ys), all_paths], file)

        return np.array(x_train), np.array(action_probs_ys), np.array(vals_ys)

    def train(self, x_train, action_probs_ys, vals_ys, optimizer, round):

        x_train = self.mcts.event_matrix_to_model_input(x_train)
        probs_y, vals_y = self.mcts.prob_val_to_model_output(action_probs_ys, vals_ys)
        print(probs_y.shape, vals_y.shape)


        for i in tqdm(range(1000)):
            policy, value = self.model(x_train)
            policy_loss = F.cross_entropy(policy, probs_y)
            value_loss = F.mse_loss(value, vals_y)
            loss = policy_loss + value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

#        value = list(value.detach().cpu().numpy())
#        value = [v.item() for v in value]
#        vals_y = list(vals_y.detach().cpu().numpy())
#        vals_y = [v.item() for v in vals_y]
#        print(list(zip(value, vals_y)))
##        print(np.sum(value), np.sum(vals_y))

            print(probs_y)
            print(policy)
        with open('results/METR_LA/model_output.pck', 'wb') as f:
            pck.dump([policy, probs_y, value, vals_y], f)

        print(policy_loss, value_loss, loss)
        policy = policy.detach().cpu().numpy()
        probs_y = probs_y.detach().cpu().numpy()
        value = value.detach().cpu().numpy()
        vals_y = vals_y.detach().cpu().numpy()

        print(np.unique(policy))
#action_probs = [p*100 for p in action_probs]
        torch.save(self.model.state_dict(), f'saved/models/policy_model_{round}')

#    def learn(self):
#        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
##        optimizer.zero_grad()
#        for epoch in range(100):
#
#            with Pool(processes=self.num_workers) as pool:
#                pool.map(self.self_play(), list(range(self.num_workers)))
#
#            print(f'Self Play...')
#            x_train, action_probs_ys, vals_ys = self.self_play()
#            print(f'Training...')
#            self.train(x_train, action_probs_ys, vals_ys, optimizer)


    def combine_training_data(self, round):

        all_x_train = []
        all_y_probs = []
        all_vals_y = []
        for w in range(self.num_workers):
#            with open(f'{ results_dir }{self.num_simulations}/{str(self.mcts.selection_policy[0])[-1]}training_data_worker_{w}_{round}.pck', 'rb') as file:
            with open(f'{ results_dir }{self.num_simulations}/{str(self.mcts.selection_policy[0])[-1]}training_data_worker_{w}_{round}.pck', 'rb') as file:
#                data = pck.load(file)
                x_train, action_probs_y, vals_y, all_paths = pck.load(file)
            [all_x_train.append(x) for x in x_train]
            [all_y_probs.append(p) for p in action_probs_y]
            [all_vals_y.append(v) for v in vals_y]

            with open(f'{ results_dir }{self.num_simulations}/all_{str(self.mcts.selection_policy[0])[-1]}training_data_round_{round}.pck', 'wb') as file:
                pck.dump([np.array(all_x_train), np.array(all_y_probs), np.array(all_vals_y)], file)

        return np.array(all_x_train), np.array(all_y_probs), np.array(all_vals_y)





class AlphaZeroModel(nn.Module):
    def __init__(self, input_window, num_nodes, feature_dim):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_window = input_window
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.num_hidden = 32

        self.startBlock = nn.Sequential(
            nn.Conv3d(in_channels=1,
                      out_channels=self.num_hidden,
                      kernel_size=(3, 1, 3),
                      stride=1,
                      padding=(1, 0, 1)),
            nn.BatchNorm3d(self.num_hidden),
            nn.ReLU()
                )

        self.resBlock1 = ResBlock(self.num_hidden)
        self.resBlock2 = ResBlock(self.num_hidden)
        self.resBlock3 = ResBlock(self.num_hidden)
#        self.resBlock4 = ResBlock(self.num_hidden)

        self.policyHead = PolicyHead(self.input_window, self.num_nodes, self.num_hidden)
        self.valueHead = ValueHead(self.input_window, self.num_nodes, self.num_hidden)

    def forward(self, x):
        x = self.startBlock(x)
        x = self.resBlock1(x)
        x = self.resBlock2(x)
        x = self.resBlock3(x)
#        x = self.resBlock4(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value


class ValueHead(nn.Module):
    def __init__(self, input_window, num_nodes, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=num_hidden,
                  out_channels=1,
                  kernel_size=(1, 1, 1),
                  stride=1)
        self.bn1 = nn.BatchNorm3d(1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.ln = nn.Linear(input_window*num_nodes, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.ln(x)
        return x

class PolicyHead(nn.Module):
    def __init__(self, input_window, num_nodes, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=num_hidden,
                  out_channels=1,
                  kernel_size=(1, 1, 1),
                  stride=1)
#        self.ln1 = nn.Linear(input_window*num_nodes*num_hidden, input_window*num_nodes)
        self.bn1 = nn.BatchNorm3d(1)
#        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.flatten = nn.Flatten()
        self.ln2 = nn.Linear(input_window*num_nodes, input_window*num_nodes)
#        self.sfm = nn.Softmax(dim=1)
#
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.flatten(x)
#        x = self.ln1(x)
#        print(x.shape)
        x = self.leaky_relu(x)
#        x = self.flatten(x)
        x = self.ln2(x)
        x = self.leaky_relu(x)
#        print(x.shape)
        x = self.sfm(x)
        return x



class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=num_hidden,
                               out_channels=num_hidden,
                               kernel_size=(3, 1, 3),
                               stride=1,
                               padding=(1, 0, 1))
        self.bn1 = nn.BatchNorm3d(num_hidden)
        self.leaky_relu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(in_channels=num_hidden,
                               out_channels=num_hidden,
                               kernel_size=(3, 1, 3),
                               stride=1,
                               padding=(1, 0, 1))
        self.bn2 = nn.BatchNorm3d(num_hidden)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.leaky_relu(out)
        return out




#def load_model(epoch_num):
#    with open(f'saved/models/policy_model_{epoch_num}', 'rb')


if __name__ == '__main__':
        print(torch.cuda.is_available())  # Should return True if CUDA is available
        print(torch.version.cuda)  # Prints the version of CUDA PyTorch is using
        print(torch.__version__)
        print(torch.backends.cudnn.enabled)  # Checks if cuDNN is enabled (used for faster convolution)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        explainer = run_explainer()
        alpha_zero = AlphaZero(device, explainer)
#    alpha_zero.learn()

        optimizer = torch.optim.Adam(alpha_zero.model.parameters(), lr=0.00001)
#        optimizer.zero_grad()
#    with Pool(5) as p:
#        p.map(alpha_zero.self_play, [1])

        for round in range(100):
#            alpha_zero.model.eval()
#            with concurrent.futures.ProcessPoolExecutor(max_workers=alpha_zero.num_workers, mp_context=mp.get_context("spawn")) as executor:
#                futures = [executor.submit(alpha_zero.self_play, i, round) for i in range(alpha_zero.num_workers)]
#                results = [future.result() for future in concurrent.futures.as_completed(futures)]
##
#            torch.cuda.empty_cache()

#
            x_train, action_probs_y, vals_y = alpha_zero.combine_training_data(round)
#        summary(alpha_zero.model, input_size=(1, 12, 1, 207))
            print(x_train.shape, action_probs_y.shape, vals_y.shape)
            alpha_zero.train(x_train, action_probs_y, vals_y, optimizer, round)
#    for epoch in range(100):
#
#        print(f'Self Play...')
#        x_train, action_probs_ys, vals_ys = alpha_zero.self_play()
#        print(f'Training...')


