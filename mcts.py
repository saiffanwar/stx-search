import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
import pickle as pck
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import torch
from scipy.special import softmax


unvisited_reward = 0
np.random.seed(0)

class treeNode:
    def __init__(self, events, children=[], parent=None, prior=0):
        '''
        Args:
            parent: treeNode() object for the parent of the current node.
            children: list of treeNode() objects for the children of the current node.
            events: list of integers representing the indices of the graphNode() objects that are in the explanation subset where
                    the indices are stored in mcts.all_event_graph_nodes
            prior: float representing the prior probability of selecting this node. Init with 0 and updated
                    with Alpha Zero model policy output.
        '''
        self.parent = parent
        self.children = children
        self.events = events
        self.visits = 0
        self.cumulative_reward = 0
        self.value = 0
        self.expanded = False
        self.node_id = np.random.randint(0, 1000000)
        self.prior = prior.cpu().detach().numpy() if isinstance(prior, torch.Tensor) else prior
        self.last_updated = None

class MCTS:

    def  __init__(self, explainer, model, exp_size=100):
        '''
        Args:
            candidate_events: list of graphNode() objects that are possible candidates for the explanation subset
        '''
#        self.candidate_events = candidate_events
        self.node_ids = []
        self.nodes_expanded = 0
        self.nodes_visited = 0
        self.explainer = explainer
        self.model = model
        self.exp_size = exp_size
        self.best_exp = None
        self.best_exp_reward = unvisited_reward
        self.leaves = []
        self.results_dir = 'results/METR_LA/'

        self.all_event_graph_nodes = {e: event for e, event in enumerate(self.explainer.candidate_events)}
        self.all_event_indices = [e for e in range(len(self.explainer.candidate_events))]
        self.root = treeNode(events=self.all_event_indices)
        self.leaves.append(self.root)
        self.node_ids.append(self.root.node_id)
        self.expansion_protocol = 'single_child'

    def uct(self, tree_node, c=0.4):
        '''
        Args:
            tree_node: treeNode() object
        '''
#        if tree_node.visits == 0:
#            return unvisited_reward
#        else:
#            print((tree_node.cumulative_reward / tree_node.visits) , c*np.sqrt(tree_node.parent.visits / tree_node.visits))
        return tree_node.value + c*tree_node.prior*np.sqrt(tree_node.parent.visits / (1+tree_node.visits))
#        return tree_node.prior


    def selection(self, tree_node, policy):
#        print('Selection...')
        # If we only expand a single child at a time, sometimes we force to exploit rather than expand a new child.
        if self.expansion_protocol == 'single_child':
            explore = np.random.choice([True, False], p=[0.5, 0.5])
        else:
            explore = False
        if explore:
            new_child = self.expansion(tree_node, policy)
            return new_child
        else:
            scores = [self.uct(child) for child in tree_node.children]
#        ordered_scores = sorted(scores)
#        print(ordered_scores[:5])
#            print(np.unique(scores, return_counts=True))
            best_node = tree_node.children[np.argmax(scores)]
            return best_node


    def expansion(self, tree_node, policy=None):
        '''
        Expands a new child for the given tree node based on the policy provided by the model. Masks out the policy to only
        allow actions that haven't been taken yet and then selects from the remaining distribution.

        Args:
            tree_node: treeNode() object
            policy: np.array of shape (num_actions) containing the probabilities of each action

        Returns:
        '''
#        print('Expansion...')
        # If no policy, then select a random action by assigning everything same probability.
#        if policy is None:
#            policy = np.ones(len(self.all_event_indices))

        if self.expansion_protocol == 'single_child':
            masked_policy = policy.copy()
            state_events = tree_node.events.copy()
            for i in range(len(self.all_event_indices)):
                if i not in state_events:
                    masked_policy[i] = 0

#        masked_policy_norm = masked_policy / np.sum(masked_policy)
            policy_as_probs = softmax(masked_policy)
            action = np.random.choice(range(len(masked_policy)), p=policy_as_probs)
            new_child_events = [e for e in state_events if e != action]
            new_child = treeNode(events=new_child_events, children=[], parent=tree_node, prior=masked_policy[action])

            # If it is the first expansion, remove the parent from the leaves list.
            if len(tree_node.children) == 0:
                self.leaves.remove(tree_node)
            self.leaves.append(new_child)
            tree_node.children.append(new_child)

            # Check if the node is now fully expanded. There should be a child for each action (removing each event).
            if len(tree_node.children) == len(tree_node.events):
                tree_node.expanded = True

        elif self.expansion_protocol == 'all_children':
            for i in tree_node.events:
                new_child_events = tree_node.events.copy()
                new_child_events.remove(i)
                new_child = treeNode(events=new_child_events, children=[], parent=tree_node, prior=policy[i])
                tree_node.children.append(new_child)
                self.leaves.append(new_child)
            self.leaves.remove(tree_node)
            tree_node.expanded = True

        return new_child



    def simulation(self, leaf_node):
        '''
        Args:
            tree_node: treeNode() object
        '''
#        print('Simulation...')
        possible_events = leaf_node.events.copy()
        while len(possible_events) > self.exp_size:
            random_event = np.random.choice(possible_events)
            possible_events.remove(random_event)

        exp_events = [self.all_event_graph_nodes[e] for e in possible_events]
        reward = self.explainer.exp_fidelity(exp_events)
        if reward < self.best_exp_reward:
            self.best_exp = possible_events
            self.best_exp_reward = reward
            exp_subgraph = [self.all_event_graph_nodes[e] for e in self.best_exp]
            with open(self.results_dir+'best_exp.pck', 'wb') as f:
                pck.dump(exp_subgraph , f)
        return reward, possible_events


    def backpropagate(self, reward, ancestors, iteration):
#        print('Backpropagation...')
        for leaf_node in ancestors:
            current_node = leaf_node
            while current_node.parent != None:
                if current_node.last_updated != iteration:
#                print(f'Backpropagating node: {current_node.node_id} with visits: {current_node.visits}')
                    self.nodes_visited += 1
                    current_node.visits += 1
                    current_node.cumulative_reward = (current_node.cumulative_reward + reward)
                    current_node.value = current_node.cumulative_reward / current_node.visits
                    current_node.last_updated = iteration
                current_node = current_node.parent
            # update root node
            if current_node.last_updated != iteration:
                current_node.visits += 1
                current_node.cumulative_reward = (current_node.cumulative_reward + reward)
                current_node.value = current_node.cumulative_reward / current_node.visits
                current_node.last_updated = iteration


    def find_all_ancestors(self, exp_events):
        '''
        Args:
            exp_events: list of graphNode() object indices to reference self.all_event_graph_nodes
        '''
        ancestors = []
        for leaf in self.leaves:
            if set(exp_events).issubset(set(leaf.events)):
                ancestors.append(leaf)
        return ancestors

    def visualise_exp_evolution(self, all_paths):

        fig, ax = plt.subplots()
        node_locations = {}
        for n, path in enumerate( all_paths ):
            xs, ys = [], []
            for node in path:
                if node in node_locations.keys():
                    pass
                else:
                    node_locations[node] = [path.index(node), n]
                    xs.append(node_locations[node][0])
                    ys.append(node_locations[node][1])

                if node != path[0]:
                    prev_node = path[path.index(node)-1]
                    start_x, start_y = node_locations[prev_node]
                    end_x, end_y = node_locations[node]

                    ax.plot([start_x, end_x], [start_y, end_y], 'k-', zorder=0, alpha=0.5)

                ax.scatter(xs, ys, s=100, zorder=1, )

        fig.savefig(f'{ self.results_dir }exp_evolution_iter_{len(all_paths)}.png')
        plt.close()

    def tree_search_animation(self):
        with open(self.results_dir+'all_paths.pck', 'rb') as f:
            all_paths = pck.load(f)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(-1, len(all_paths[-1])*1.1)
        ax.set_ylim(-1, len(all_paths)*1.1)
        node_locations = {}
        def animate(n):
            path = all_paths[n]
            xs, ys = [], []
            for node in path:
                if node in node_locations.keys():
                    pass
                else:
                    node_locations[node] = [path.index(node), n]
                    xs.append(node_locations[node][0])
                    ys.append(node_locations[node][1])

                if node != path[0]:
                    prev_node = path[path.index(node)-1]
                    start_x, start_y = node_locations[prev_node]
                    end_x, end_y = node_locations[node]

                    ax.plot([start_x, end_x], [start_y, end_y], 'k-', zorder=0, alpha=0.5)

                ax.scatter(xs, ys, s=100, zorder=1, )
        ani = animation.FuncAnimation(fig, animate, frames=len(all_paths), interval=100)
        ani.save(f'{ self.results_dir }exp_evolution_iter_{len(all_paths)}.mp4', writer='ffmpeg')

    def event_indices_to_graph_nodes(self, event_indices):
        return [self.all_event_graph_nodes[e] for e in event_indices]
#
#
#    def run_mcts(self, num_iterations=100):
#
#        os.system(f'rm { self.results_dir }exp_evolution_iter_*.png')
#        all_paths = []
#        for i in tqdm(range(num_iterations)):
#            leaf_node=False
#            current_node = self.root
#            self.node_to_event_matrix(current_node)
##            for child in self.root.children:
##                print(f'Visits: {child.visits}, Value: {child.cumulative_reward}, Layer: {len(self.root.events) - len(child.events)}, Expanded: {child.expanded}')
#            path = []
#            while not leaf_node:
#                if current_node.expanded:
#                    current_node = self.selection(current_node)
#                else:
#                    leaf_node = True
#                    self.expansion(current_node)
#                    self.leaves.remove(current_node)
#                print(f'Visits: {current_node.visits}, Value: {current_node.cumulative_reward}, Layer: {len(self.root.events) - len(current_node.events)}, Expanded: {current_node.expanded}')
#                path.append(current_node.node_id)
#
#            print(path)
#            all_paths.append(path)
#            reward, exp_events = self.simulation(current_node)
#            ancestors = self.find_all_ancestors(exp_events)
#
#            self.backpropagate(reward, ancestors, iteration=i)
#            print('Best Explanation Fidelity: {}'.format(self.best_exp_reward))
#            self.visualise_exp_evolution(all_paths)
#            with open(self.results_dir+'all_paths.pck', 'wb') as f:
#                pck.dump(all_paths, f)
#
#        exp_subgraph = [self.all_event_graph_nodes[e] for e in self.best_exp]
#        return exp_subgraph

    def node_to_event_matrix(self, node):
        events = self.event_indices_to_graph_nodes(node.events)
        input_window, num_nodes, feature_dim = self.explainer.input_sample_shape
        event_presence_matrix = np.zeros((self.explainer.input_sample_shape), dtype=np.float32)

        for event in events:
            event_presence_matrix[event.timestamp, event.node_index, :] = 1

#        event_presence_matrix = np.reshape(event_presence_matrix, (1, input_window, num_nodes, feature_dim))
        # Leave at cpu
        return event_presence_matrix

    def event_matrix_to_model_input(self, event_presence_matrix):
        event_presence_matrix = event_presence_matrix.reshape(len(event_presence_matrix), 1, self.model.input_window, self.model.feature_dim, self.model.num_nodes)
        event_presence_matrix = torch.tensor(event_presence_matrix, dtype=torch.float32).to(self.model.device)
        return event_presence_matrix

    def prob_val_to_model_output(self, action_probabilities, values):
        action_probabilities = action_probabilities.reshape(len(action_probabilities), self.model.num_nodes*self.model.input_window)
        action_probabilities = torch.tensor(action_probabilities, dtype=torch.float32).to(self.model.device)

        values = torch.tensor(values, dtype=torch.float32).to(self.model.device)
        return action_probabilities, values

    def generate_probability_matrix_for_children(self, tree_node):
#        events = self.event_indices_to_graph_nodes(tree_node.events)
        probabilities = np.zeros((self.explainer.input_sample_shape))
        for child in tree_node.children:
            # Find the event that was removed to create the child node, i.e., the action taken
            removed_event = list(set(tree_node.events) - set(child.events))[0]
            removed_event_timestamp = self.all_event_graph_nodes[removed_event].timestamp
            removed_event_node_index = self.all_event_graph_nodes[removed_event].node_index
            child_score = self.uct(child)
            # Record the score of taking that action in the probability matrix.
            # Actions that don't exist keep a score of 0
#            probabilities[removed_event_timestamp, removed_event_node_index, :] = child_score

            probabilities[removed_event_timestamp, removed_event_node_index, :] = child.cumulative_reward / child.visits

        probabilities = softmax(probabilities.flatten())
        # Currently a lower score is a more favourable node, so need to inverse this to imply probability.
        # Subtract all values from the max value to inverse the probabilities.
#        probabilities = [np.max(probabilities) - x if x != 0 else 0 for x in probabilities ]
#        probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities))

        return probabilities

    def is_terminal(self, tree_node):
        if len(tree_node.events) <= self.exp_size:
            return True
        else:
            return False

    def search(self, state, num_searches=100):
        '''
        Args:
            state: np.array of shape (input_window, num_nodes, feature_dim)

            maybe the search is only going down 1 level, need to check this.
            What if the current_node is a leaf node, then which action probs do we return?
        '''
        all_paths = []
        print(f'Starting search with state: {len(state.events)}')
        for i in tqdm(range(num_searches)):
            path = []
            current_node = state

#            while current_node.expanded:
            # Currently the policy is not used in the UCT formula to help with the selection.
            while current_node not in self.leaves:
#                print('Selection...')
                input_sample = np.array([self.node_to_event_matrix(current_node)])
                input_sample = self.event_matrix_to_model_input(input_sample)
                policy, predicted_reward = self.model(input_sample)
                policy = policy.cpu().detach().numpy()[0]
                predicted_reward = int(predicted_reward.cpu().detach().numpy()[0])
                reward = predicted_reward
#                print(np.unique(policy, return_counts=True))
                current_node = self.selection(current_node, policy)
#                path.append(current_node.node_id)


            # Every 10th iteration, force a real simulated reward to propogate back up the tree.
            if not self.is_terminal(current_node):
#                print('Expansion...')
                input_sample = np.array([self.node_to_event_matrix(current_node)])
                input_sample = self.event_matrix_to_model_input(input_sample)
                policy, predicted_reward = self.model(input_sample)
                predicted_reward = int(predicted_reward.cpu().detach().numpy()[0])
#                reward, _ = self.simulation(current_node)
                reward = predicted_reward
                policy = policy.cpu().detach().numpy()[0]
#                print(np.unique(policy, return_counts=True))
#                print(policy)
                current_node = self.expansion(current_node, policy)
                ancestors = self.find_all_ancestors(current_node.events)
                if i%10 == 0:
#                    print('Simulation...')
                    reward, possible_events = self.simulation(current_node)
                    ancestors = self.find_all_ancestors(possible_events)
#            ancestors = self.find_all_ancestors(exp_events)
            else:
                exp_events = [self.all_event_graph_nodes[e] for e in current_node.events]
                reward = self.explainer.exp_fidelity(exp_events)
                if reward < self.best_exp_reward:
                    self.best_exp = current_node.events
                    self.best_exp_reward = reward
                    exp_subgraph = [self.all_event_graph_nodes[e] for e in self.best_exp]
                    with open(self.results_dir+'best_exp.pck', 'wb') as f:
                        pck.dump(exp_subgraph , f)
                ancestors = self.find_all_ancestors(current_node.events)
            print(len(ancestors))
#            ancestors = [current_node]
            self.backpropagate(reward, ancestors, iteration=i)
#            all_paths.append(path)
#            self.visualise_exp_evolution(all_paths)
#            with open(self.results_dir+'all_paths.pck', 'wb') as f:
#                pck.dump(all_paths, f)

        action_probs = self.generate_probability_matrix_for_children(state)
        reward = state.cumulative_reward/state.visits

        return action_probs, reward, all_paths

