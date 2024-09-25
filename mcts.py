import numpy as np
import pandas as pd
from tqdm import tqdm
import copy


class treeNode():
    def __init__(self, events, children=[], parent=None):
        self.parent = parent
        self.children = children
        self.events = events
        self.visits = 0
        self.cumulative_reward = np.inf
        self.expanded = False

class MCTS():

    def  __init__(self, candidate_events, explainer, batch, exp_size=2000):
        '''
        Args:
            candidate_events: list of graphNode() objects that are possible candidates for the explanation subset
        '''
#        self.candidate_events = candidate_events
        self.all_event_graph_nodes = {e: event for e, event in enumerate(candidate_events)}
        self.all_event_indices = [e for e in range(len(candidate_events))]
        self.root = treeNode(events=self.all_event_indices)
        self.nodes_expanded = 0
        self.nodes_visited = 0
        self.explainer = explainer
        self.batch = batch
        self.exp_size = exp_size
        self.best_exp = None
        self.best_exp_reward = np.inf
        self.leaves = [self.root]

    def uct(self, tree_node, c=np.sqrt(2)):
        '''
        Args:
            tree_node: treeNode() object
        '''
        if tree_node.visits == 0:
            return np.inf
        else:
            return tree_node.cumulative_reward / tree_node.visits + c*np.sqrt(np.log(tree_node.parent.visits) / tree_node.visits)

    def selection(self, tree_node):
#        print('Selection...')
        return min(tree_node.children, key=lambda x: self.uct(x))

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
                self.nodes_visited += 1
                current_node.visits += 1
                if current_node.cumulative_reward == np.inf:
                    current_node.cumulative_reward = reward
                else:
                    current_node.cumulative_reward = (current_node.cumulative_reward + reward)
                current_node = current_node.parent
            # update root node
            current_node.visits += 1
            if current_node.cumulative_reward == np.inf:
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
            tree_node.children.append(treeNode(events=childs_events, parent=tree_node))
            self.leaves.append(tree_node.children[-1])
#        print('Expanding node with {} children'.format(len(tree_node.children)))
        self.nodes_expanded += 1

#        print('Nodes expanded: {}'.format(self.nodes_expanded))
        tree_node.expanded = True


    def run_mcts(self, num_iterations=100):
        for i in tqdm(range(num_iterations)):
            leaf_node=False
            current_node = self.root
            while not leaf_node:
                print(f'Visits: {current_node.visits}, Value: {current_node.cumulative_reward}, Layer: {len(self.root.events) - len(current_node.events)}, Expanded: {current_node.expanded}')
                if current_node.expanded:
                    current_node = self.selection(current_node)
                else:
                    leaf_node = True
                    self.expansion(current_node)
                    self.leaves.remove(current_node)

            reward, exp_events = self.simulation(current_node)
            ancestors = self.find_all_ancestors(exp_events)
            self.backpropagate(reward, ancestors)
#            print('Best Explanation: {}'.format(self.best_exp))
            print('Best Explanation Fidelity: {}'.format(self.best_exp_reward))


'''
Best fidelity so far is 12.05
Best error so far with 1000 exp size is 2.188 after 50 iterations of MCTS upto a depth of at least 42
Best error so far with 500 exp size is 3.208 after 50 iterations of MCTS upto a depth of at least 27
'''


