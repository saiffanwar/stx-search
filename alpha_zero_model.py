import torch
from mcts import MCTS
import pickle as pck
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
import torch.nn.functional as F
#from torchsummary import summary
from explainer import run_explainer
from tqdm import tqdm
from multiprocessing import Pool


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
        self.num_workers = 1
        self.model = PolicyModel(input_window=12, num_nodes=207, feature_dim=1).to(device)
#        self.mcts = MCTS(self.explainer, self.model)


    def self_play(self, worker_num=1):
        print(f'Starting worker {worker_num}')
        x_train = []
        action_probs_ys = []
        vals_ys = []

        all_paths = []

        # Resest the tree before self play. avoids magnifying random biases
        self.mcts = MCTS(self.explainer, self.model)
        current_node = self.mcts.root

        for i in tqdm(range(10)):
            action_probs, reward, paths = self.mcts.search(current_node)
#            [all_paths.append(p) for p in paths]
#            plt.bar(range(len(action_probs)), action_probs)
#            with open('results/probabilities.pck', 'wb') as f:
#                pck.dump(action_probs, f)
#                f.close()
            print(np.unique(action_probs, return_counts=True))
#            plt.savefig(f'results/METR_LA/action_probs_{worker_num}.png')
#            plt.close()

#            self.mcts.visualise_exp_evolution(all_paths)

            x_train.append(self.mcts.node_to_event_matrix(current_node))
            action_probs_ys.append(action_probs)
            vals_ys.append(reward)

            # Select the next node based on the action probabilities
#            selected_child = np.random.choice(list(range(len(action_probs))), p=action_probs)
            if self.mcts.expansion_protocol == 'single_child':
                current_node = self.mcts.expansion(current_node, action_probs)
            else:
                current_node = np.random.choice(current_node.children, p=action_probs)


#            print(f'Size of tree: {len(self.mcts.node_ids)}')


#        with open(f'training_data_worker_{worker_num}.pck', 'wb') as file:
#            pck.dump([np.array(x_train), np.array(action_probs_ys), np.array(vals_ys)], file)
#            file.close()

        return np.array(x_train), np.array(action_probs_ys), np.array(vals_ys)

    def train(self, x_train, action_probs_ys, vals_ys, optimizer):

        x_train = self.mcts.event_matrix_to_model_input(x_train)
        print(x_train.shape)
        probs_y, vals_y = self.mcts.prob_val_to_model_output(action_probs_ys, vals_ys)

        for i in tqdm(range(100)):
            policy, value = self.model(x_train)
            policy_loss = F.cross_entropy(policy, probs_y)
            value_loss = F.mse_loss(value, vals_y)
            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()
        print(loss)

    def test_func(self, w):
        print(f'Running worker {w}')
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






class PolicyModel(nn.Module):
    def __init__(self, input_window, num_nodes, feature_dim):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_window = input_window
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.num_hidden = 16

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
#        self.resBlock3 = ResBlock(self.num_hidden)
#        self.resBlock4 = ResBlock(self.num_hidden)

        self.policyHead = nn.Sequential(
            nn.Conv3d(in_channels=self.num_hidden,
                      out_channels=1,
                      kernel_size=(1, 1, 1),
                      stride=1),
            nn.BatchNorm3d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.input_window*self.num_nodes, self.input_window*self.num_nodes),
            nn.ReLU(),
            nn.Softmax(dim=1)
#            nn.Unflatten(1, (self.input_window, self.num_nodes, 1))
        )

        self.valueHead = nn.Sequential(
            nn.Conv3d(in_channels=self.num_hidden,
                      out_channels=1,
                      kernel_size=(1, 1, 1),
                      stride=1),
            nn.BatchNorm3d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.input_window*self.num_nodes, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.startBlock(x)
        x = self.resBlock1(x)
        x = self.resBlock2(x)
#        x = self.resBlock3(x)
#        x = self.resBlock4(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value



class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=num_hidden,
                               out_channels=num_hidden,
                               kernel_size=(3, 1, 3),
                               stride=1,
                               padding=(1, 0, 1))
        self.bn1 = nn.BatchNorm3d(num_hidden)
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
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out



#def load_model(epoch_num):
#    with open(f'saved/models/policy_model_{epoch_num}', 'rb')


results_dir = 'results/METR_LA/'

if __name__ == '__main__':

    print(torch.cuda.is_available())  # Should return True if CUDA is available
    print(torch.version.cuda)  # Prints the version of CUDA PyTorch is using
    print(torch.__version__)
    print(torch.backends.cudnn.enabled)  # Checks if cuDNN is enabled (used for faster convolution)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#    summary(model, input_size=x_train.shape[1:])
    explainer = run_explainer()
    alpha_zero = AlphaZero(device, explainer)
#    alpha_zero.learn()

    optimizer = torch.optim.Adam(alpha_zero.model.parameters(), lr=0.0001)
#        optimizer.zero_grad()
#    for epoch in range(100):

    with Pool(processes=alpha_zero.num_workers) as pool:
#        pool.map(alpha_zero.self_play, list(range(alpha_zero.num_workers)), chunksize=1)
        pool.map(alpha_zero.test_func, list(range(alpha_zero.num_workers)), chunksize=1)
        pool.close()

#        print(f'Self Play...')
#        x_train, action_probs_ys, vals_ys = alpha_zero.self_play()
#        print(f'Training...')
#        alpha_zero.train(x_train, action_probs_ys, vals_ys, optimizer)


