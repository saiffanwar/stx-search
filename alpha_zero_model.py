import torch
from mcts import MCTS
import pickle as pck
import numpy as np

from torch import nn
import torch.nn.functional as F
from torchsummary import summary
from explainer import run_explainer



class AlphaZero:
    '''
    Class to run the AlphaZero algorithm
    '''
    def __init__(self, model, device, explainer):
        '''
        Args:
            model: nn.Module() object
            device: torch.device() object
            explainer: explainer object needed to make predictions for evaluation explanation fidelity.
        '''
        self.model = model
        self.device = device
        self.explainer = explainer
        self.mcts = MCTS(self.explainer, self.model)


    def self_play(self, ):
        x_train = []
        y_train = []

        current_node = self.mcts.root
        for i in range(10):
            action_probs, reward = self.mcts.search(current_node)
            x_train.append(self.mcts.node_to_event_matrix(current_node))
            y_train.append([action_probs, reward])

            # Select the next node based on the action probabilities
            selected_child = np.random.choice(list(range(len(action_probs))), p=action_probs)
            current_node = current_node.children[selected_child]

            print(f'Size of tree: {len(self.mcts.node_ids)}')

        return x_train, y_train
#            x_train.append(self.mcts.node_to_event_matrix(current_node))
#            if current_node.expanded == False:
#                self.mcts.expansion(current_node)
#            y_train.append([self.mcts.generate_probability_matrix_for_children(current_node), 0])

    def train(self, x_train, y_train):
        x_train = x_train[0].view(1, 1, self.model.input_window, self.model.num_nodes, self.model.feature_dim).to(self.model.device)
#        y_train = y_train[0].view(1, 1, self.model.input_window, self.model.num_nodes, self.model.feature_dim).to(self.model.device)
        vals_y = torch.tensor([y_train[0][1]]).to(self.model.device)
        probs_y = torch.tensor([y_train[0][0]]).to(self.model.device)
        print(probs_y.shape)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        for i in range(100):
            optimizer.zero_grad()
            policy, value = self.model(x_train)
            policy_loss = F.cross_entropy(policy, probs_y)
            value_loss = F.mse_loss(value, vals_y)
            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()
        print(loss)

    def learn(self):
        for epoch in range(100):
            x_train, y_train = self.self_play()
            self.train(x_train, y_train)






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
        self.resBlock3 = ResBlock(self.num_hidden)
        self.resBlock4 = ResBlock(self.num_hidden)

        self.policyHead = nn.Sequential(
            nn.Conv3d(in_channels=self.num_hidden,
                      out_channels=1,
                      kernel_size=(1, 1, 1),
                      stride=1),
            nn.BatchNorm3d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.input_window*self.num_nodes, self.input_window*self.num_nodes),
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
            nn.Linear(self.input_window*self.num_nodes, 1)
        )

    def forward(self, x):
        x = self.startBlock(x)
        x = self.resBlock1(x)
        x = self.resBlock2(x)
        x = self.resBlock3(x)
        x = self.resBlock4(x)
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




results_dir = 'results/METR_LA/'

def main():
#
    print(torch.cuda.is_available())  # Should return True if CUDA is available
    print(torch.version.cuda)  # Prints the version of CUDA PyTorch is using
    print(torch.__version__)
    print(torch.backends.cudnn.enabled)  # Checks if cuDNN is enabled (used for faster convolution)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(results_dir+'x_train.pck', 'rb') as f:
        x_train = pck.load(f)[0]
    input_window, num_nodes, feature_dim = x_train.shape

    print(f'Original Shape: {x_train.shape}')
    x_train = x_train.view(1, 1, input_window, feature_dim, num_nodes).cuda()
#    print(f'Input Shape: {x_train.shape}')
    model = PolicyModel(input_window=input_window, num_nodes=num_nodes, feature_dim=feature_dim).to(device)
    summary(model, input_size=x_train.shape[1:])
#    policy, value = model(x_train)
#    print(f'Output Shape: {policy.shape}, {value.shape}')
#    print(value)
    explainer = run_explainer()
    alpha_zero = AlphaZero(model, device, explainer)
    alpha_zero.learn()

if __name__ == '__main__':
    main()

