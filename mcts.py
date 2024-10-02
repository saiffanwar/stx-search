import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
import pickle as pck
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


class treeNode():
    def __init__(self, current_node_ids, events, children=[], parent=None):
        self.parent = parent
        self.children = children
        self.events = events
        self.visits = 0
        self.cumulative_reward = 1e-10
        self.expanded = False
        self.node_id = np.random.randint(0, 1000000)

class MCTS():

    def  __init__(self, candidate_events, explainer, batch, exp_size=100):
        '''
        Args:
            candidate_events: list of graphNode() objects that are possible candidates for the explanation subset
        '''
#        self.candidate_events = candidate_events
        self.all_event_graph_nodes = {e: event for e, event in enumerate(candidate_events)}
        self.all_event_indices = [e for e in range(len(candidate_events))]
        self.node_ids = []
        self.root = treeNode(current_node_ids=self.node_ids, events=self.all_event_indices)
        self.node_ids.append(self.root.node_id)
        self.nodes_expanded = 0
        self.nodes_visited = 0
        self.explainer = explainer
        self.batch = batch
        self.exp_size = exp_size
        self.best_exp = None
        self.best_exp_reward = np.inf
        self.leaves = [self.root]
        self.results_dir = 'results/METR_LA/'

    def uct(self, tree_node, c=0.01):
        '''
        Args:
            tree_node: treeNode() object
        '''
        if tree_node.visits == 0:
            return 1e-10
        else:
#            print((tree_node.cumulative_reward / tree_node.visits) , c*np.sqrt(tree_node.parent.visits / tree_node.visits))
            return (tree_node.cumulative_reward / tree_node.visits) + c*np.sqrt(tree_node.parent.visits / tree_node.visits)

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
        reward = self.explainer.exp_fidelity(exp_events, self.batch)
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



    def expansion(self, tree_node):
        '''
        Args:
            tree_node: treeNode() object
        '''
#        print('Expansion...')
        tree_node.children = []
        for event_num, e in enumerate(tree_node.events):
#            childs_events = [x for x in tree_node.events if x != e]
            childs_events = tree_node.events[:event_num] + tree_node.events[event_num+1:]
            tree_node.children.append(treeNode(current_node_ids=self.node_ids, events=childs_events, parent=tree_node))
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

#        plt.show()
#        fig.savefig(f'{ self.results_dir }exp_evolution_iter_{len(all_paths)}.png')

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


    def run_mcts(self, num_iterations=100):
        os.system(f'rm { self.results_dir }exp_evolution_iter_*.png')
        all_paths = []
        for i in tqdm(range(num_iterations)):
            leaf_node=False
            current_node = self.root
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
#            print('Best Explanation: {}'.format(self.best_exp))
            print('Best Explanation Fidelity: {}'.format(self.best_exp_reward))
            self.visualise_exp_evolution(all_paths)
            with open(self.results_dir+'all_paths.pck', 'wb') as f:
                pck.dump(all_paths, f)
        print(self.best_exp, len(self.best_exp))
        exp_subgraph = [self.all_event_graph_nodes[e] for e in self.best_exp]
        return exp_subgraph


'''
Best error so far with 2000 exp size is 0.305 after 50 iterations of MCTS upto a depth of at least 40
Best error so far with 1000 exp size is 2.188 after 50 iterations of MCTS upto a depth of at least 42
Best error so far with 500 exp size is 2.719 after 50 iterations of MCTS upto a depth of at least 27
'''



