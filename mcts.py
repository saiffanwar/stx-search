import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
import pickle as pck
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import torch


class treeNode:
    def __init__(self, current_node_ids, events, children=[], parent=None, prior=0):
        self.parent = parent
        self.children = children
        self.events = events
        self.visits = 0
        self.cumulative_reward = 1e-10
        self.expanded = False
        self.node_id = np.random.randint(0, 1000000)
        self.prior = prior.cpu().detach().numpy() if isinstance(prior, torch.Tensor) else prior

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
        self.best_exp_reward = np.inf
        self.leaves = []
        self.results_dir = 'results/METR_LA/'

        self.all_event_graph_nodes = {e: event for e, event in enumerate(self.explainer.candidate_events)}
        self.all_event_indices = [e for e in range(len(self.explainer.candidate_events))]
        self.root = treeNode(current_node_ids=self.node_ids, events=self.all_event_indices)
        self.leaves.append(self.root)
        self.node_ids.append(self.root.node_id)

    def uct(self, tree_node, c=0.01):
        '''
        Args:
            tree_node: treeNode() object
        '''
        if tree_node.visits == 0:
            return 1e-10
        else:
#            print((tree_node.cumulative_reward / tree_node.visits) , c*np.sqrt(tree_node.parent.visits / tree_node.visits))
            return (tree_node.cumulative_reward / tree_node.visits) + c*np.sqrt(tree_node.parent.visits / tree_node.visits)*tree_node.prior



    def selection(self, tree_node):
#        print('Selection...')
        scores = [self.uct(child) for child in tree_node.children]
#        ordered_scores = sorted(scores)
#        print(ordered_scores[:5])
        best_node = tree_node.children[np.argmin(scores)]
        return best_node

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

    def backpropagate(self, reward, ancestors):
#        print('Backpropagation...')
        for leaf_node in ancestors:
            current_node = leaf_node
            while current_node.parent != None:
#                print(f'Backpropagating node: {current_node.node_id} with visits: {current_node.visits}')
                self.nodes_visited += 1
                current_node.visits += 1
                if current_node.cumulative_reward == 1e-10:
                    current_node.cumulative_reward = reward
                else:
                    current_node.cumulative_reward = (current_node.cumulative_reward + reward)
                current_node = current_node.parent
            # update root node
            current_node.visits += 1
            if current_node.cumulative_reward == 1e-10:
                current_node.cumulative_reward = reward
            else:
                current_node.cumulative_reward = (current_node.cumulative_reward + reward)



    def expansion(self, tree_node, policy):
        '''
        Args:
            tree_node: treeNode() object
        '''
#        print('Expansion...')
        tree_node.children = []
        for event_num, e in enumerate(tree_node.events):
#            childs_events = [x for x in tree_node.events if x != e]
            childs_events = tree_node.events[:event_num] + tree_node.events[event_num+1:]
            child_prior = policy[e]
            tree_node.children.append(treeNode(current_node_ids=self.node_ids, events=childs_events, parent=tree_node, prior=child_prior))
            self.node_ids.append(tree_node.children[-1].node_id)
            self.leaves.append(tree_node.children[-1])
#        print('Expanding node with {} children'.format(len(tree_node.children)))
        self.nodes_expanded += 1

#        print('Nodes expanded: {}'.format(self.nodes_expanded))
        tree_node.expanded = True

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


    def run_mcts(self, candidate_events, batch, num_iterations=100):

        self.all_event_graph_nodes = {e: event for e, event in enumerate(candidate_events)}
        self.all_event_indices = [e for e in range(len(candidate_events))]
        self.root = treeNode(current_node_ids=self.node_ids, events=self.all_event_indices)
        self.leaves.append(self.root)
        self.node_ids.append(self.root.node_id)
        self.batch = batch

        os.system(f'rm { self.results_dir }exp_evolution_iter_*.png')
        all_paths = []
        for i in tqdm(range(num_iterations)):
            leaf_node=False
            current_node = self.root
            self.node_to_event_matrix(current_node)
#            for child in self.root.children:
#                print(f'Visits: {child.visits}, Value: {child.cumulative_reward}, Layer: {len(self.root.events) - len(child.events)}, Expanded: {child.expanded}')
            path = []
            while not leaf_node:
                if current_node.expanded:
                    current_node = self.selection(current_node)
                else:
                    leaf_node = True
                    self.expansion(current_node)
                    self.leaves.remove(current_node)
                print(f'Visits: {current_node.visits}, Value: {current_node.cumulative_reward}, Layer: {len(self.root.events) - len(current_node.events)}, Expanded: {current_node.expanded}')
                path.append(current_node.node_id)

            print(path)
            all_paths.append(path)
            reward, exp_events = self.simulation(current_node)
            ancestors = self.find_all_ancestors(exp_events)

            self.backpropagate(reward, ancestors)
            print('Best Explanation Fidelity: {}'.format(self.best_exp_reward))
            self.visualise_exp_evolution(all_paths)
            with open(self.results_dir+'all_paths.pck', 'wb') as f:
                pck.dump(all_paths, f)

        exp_subgraph = [self.all_event_graph_nodes[e] for e in self.best_exp]
        return exp_subgraph

    def node_to_event_matrix(self, node):
        events = self.event_indices_to_graph_nodes(node.events)
        input_window, num_nodes, feature_dim = self.explainer.input_sample_shape
        event_presence_matrix = np.zeros((self.explainer.input_sample_shape), dtype=np.float32)

        for event in events:
            event_presence_matrix[event.timestamp, event.node_index, :] = 1

#        event_presence_matrix = np.reshape(event_presence_matrix, (1, input_window, num_nodes, feature_dim))
        # Leave at cpu
        event_presence_matrix = torch.tensor(event_presence_matrix)
        return event_presence_matrix


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
            probabilities[removed_event_timestamp, removed_event_node_index, :] = child_score

        probabilities = probabilities.flatten()
        probabilities = [np.max(probabilities) - x if x != 0 else 0 for x in probabilities ]
        probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities))
        # Currently a lower score is a more favourable node, so need to inverse this to imply probability.
        # Subtract all values from the max value to inverse the probabilities.
#        max_score = np.max(probabilities)
#        print(max_score)

        return probabilities

    def is_leaf(self, tree_node):
        if len(tree_node.events) <= self.exp_size:
            print('Leaf node reached')
            return True
        else:
            return False

    def search(self, state, num_searches=10):
        '''
        Args:
            state: np.array of shape (input_window, num_nodes, feature_dim)

            maybe the search is only going down 1 level, need to check this.
            What if the current_node is a leaf node, then which action probs do we return?
        '''
        for i in tqdm(range(num_searches)):
            current_node = state
            while current_node.expanded:
                # Currently the policy is not used in the UCT formula to help with the selection.
                current_node = self.selection(current_node)

            if not self.is_leaf(current_node):
                input = self.node_to_event_matrix(current_node)
                input = input.view(1, 1, self.model.input_window, self.model.num_nodes, self.model.feature_dim).to(self.model.device)
                policy, reward = self.model(input)
                reward = reward.cpu().detach().numpy()[0]
                policy = policy.cpu().detach().numpy()[0]
                self.expansion(current_node, policy)
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
#            ancestors = self.find_all_ancestors(current_node.events)
            self.backpropagate(reward, ancestors=[current_node])

        action_probs = self.generate_probability_matrix_for_children(state)
        reward = state.cumulative_reward/state.visits

        return action_probs, reward



#        current_node = self.root
#        while True:
#            self.expansion(current_node)
#            self.leaves.remove(current_node)
#            x_train.append(self.node_to_event_matrix(current_node))
#            with open(self.results_dir+'x_train.pck', 'wb') as f:
#                pck.dump(x_train, f)
#            y_train.append([self.generate_probability_matrix_for_children(current_node), 0])
#
#            random_event = np.random.choice(current_node.events)
#            current_node.events.remove(random_event)
#            if len(current_node.events) == self.exp_size:
#                exp_events = [self.all_event_graph_nodes[e] for e in current_node.events]
#                reward = self.explainer.exp_fidelity(exp_events, self.explainer.batch)
#                for x in x_train:
#                    y_train.append(reward)
#
#        return x_train, y_train
#
#
#


'''
Best error so far with 2000 exp size is 0.305 after 50 iterations of MCTS upto a depth of at least 40
Best error so far with 1000 exp size is 2.188 after 50 iterations of MCTS upto a depth of at least 42
Best error so far with 500 exp size is 2.719 after 50 iterations of MCTS upto a depth of at least 27
'''



