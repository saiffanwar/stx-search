import numpy as np
import math
from explainer import run_explainer
from tqdm import tqdm
import pickle as pck
import time
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--target_node', type=int, default=0, help='Target node index for explanation')
parser.add_argument('-s', '--subgraph_size', type=int, default=50, help='Size of the subgraph for explanation')
parser.add_argument('-m', '--mode', type=str, default='fidelity+size', help='Mode for the simulated annealing algorithm')
args = parser.parse_args()

results_dir = 'results/METR_LA/simulated_annealing/'

class SimulatedAnnealing:
    def __init__(self, explainer, candidate_events, exp_size):
        self.explainer = explainer
        self.candidate_events = candidate_events
        self.best_events = []
        self.best_score = np.inf
        if args.mode == 'fidelity':
            self.starting_temperature = 1
        elif args.mode == 'fidelity+size':
            self.starting_temperature = 0.01
        self.temperature = self.starting_temperature
        self.cooling_rate = 0.9995
        self.exp_size = exp_size
        self.objective_function = explainer.exp_fidelity
        self.best_exp_graph = None
        self.acceptance_probabilities = []
        self.scores = []
        self.actions = []
        self.exp_sizes = []

#    def events_to_indices(self, events):
#        return [events.index(event) for event in events]
    def acceptance_probability(self, delta, temperature, alpha=1):
        return np.exp(-alpha*(delta)/temperature)

    def indices_to_events(self, indices):
        f = lambda idx: self.candidate_events[idx]
        return list(map(f, indices))

#    def annealing_iteration(self, current_events, current_score, current_error, current_exp_size, current_absolute_error, iteration, mode):
    def annealing_iteration(self, *args, mode):
#        new_events = self.perturb(current_events,num_events=int((self.temperature/100)*self.exp_size))
        if mode == 'fidelity':
            current_events, current_score, current_absolute_error = args
        elif mode == 'fidelity+size':
            current_events, current_score, current_percentage_error, current_percentage_size, current_absolute_error = args



        new_events = self.perturb(current_events, mode)
        if mode == 'fidelity':
            new_score, new_graph, new_absolute_error = self.objective_function(self.indices_to_events(new_events), mode)
        elif mode == 'fidelity+size':
            new_score, new_percentage_error, new_percentage_size, new_graph, new_absolute_error = self.objective_function(self.indices_to_events(new_events), mode)

        delta = new_score - current_score

#        print('Exp Size: ', len(new_events))
#        print("New score: ", new_score)
#        print("Current score: ", current_score)
#        print("Delta: ", delta)
#        print("Temperature: ", self.temperature)
        if delta < 0:
            current_score = new_score.copy()
            current_events = new_events.copy()
            current_absolute_error = new_absolute_error
            if mode == 'fidelity+size':
                current_percentage_error = new_percentage_error
                current_percentage_size = new_percentage_size
            if new_score < self.best_score:
                self.best_score = new_score.copy()
                self.best_events = new_events.copy()
                self.best_exp_graph = new_graph
            self.acceptance_probabilities.append(None)
            self.actions.append(True)
        else:
            acceptance_probability = self.acceptance_probability(delta, self.temperature)
            self.acceptance_probabilities.append(acceptance_probability)
            #            acceptance_probability = np.exp(-(new_score - current_score)/self.temperature)
#            print('Acceptance Probability: ', acceptance_probability)
            rand_val = np.random.rand()
            accept = rand_val < acceptance_probability
            if accept:
#                print(f'Accepted with probability {acceptance_probability} at iteration {iteration} with value {rand_val}')
                current_score = new_score.copy()
                current_events = new_events.copy()
                self.actions.append(True)
            else:
                self.actions.append(False)
        self.exp_sizes.append(len(current_events))
        self.temperature *= self.cooling_rate
#        print("Best score: ", self.best_score)

        with open(f'{results_dir}best_result_{self.explainer.target_index}_{self.exp_size}_{mode}.pck.tmp', 'wb') as f:
            pck.dump((self.explainer, self), f)
        if mode == 'fidelity':
            self.scores.append([current_score, new_score, 0, 0, current_absolute_error])
            return current_events, current_score, current_absolute_error
        elif mode == 'fidelity+size':
            self.scores.append([current_score, new_score, current_percentage_error, current_percentage_size, current_absolute_error])
            return current_events, current_score, current_percentage_error, current_percentage_size, current_absolute_error





    def run(self, iterations=10000, mode=args.mode):
        self.all_event_indices = list(range(len(self.candidate_events)))
        # Initialize with random events
        if mode == 'fidelity':
            initial_events = list(np.random.choice(self.all_event_indices, self.exp_size))
        elif mode == 'fidelity+size':
            explainer, sa = pck.load(open(f'{results_dir}best_result_{self.explainer.target_index}_{self.exp_size}_fidelity.pck', 'rb'))
            initial_events = sa.best_events

        if mode == 'fidelity':
            initial_score, exp_graph, initial_absolute_error = self.objective_function(self.indices_to_events(initial_events), mode)
        elif mode == 'fidelity+size':
            initial_score, initial_percentage_error, initial_percentage_size, exp_graph, initial_absolute_error = self.objective_function(self.indices_to_events(initial_events), mode)
        self.best_exp_graph = exp_graph
#        print("Initial score: ", initial_score)

        current_events = initial_events
        current_score = initial_score
        current_absolute_error = initial_absolute_error
        if mode == 'fidelity+size':
            current_percentage_error = initial_percentage_error
            current_percentage_size = initial_percentage_size


        for i in tqdm(range(iterations)):
            if mode == 'fidelity':
                args = (current_events, current_score, current_absolute_error)
                current_events, current_score, current_absolute_error = self.annealing_iteration(*args, mode=mode)
            elif mode == 'fidelity+size':
                args = (current_events, current_score, current_percentage_error, current_percentage_size, current_absolute_error)
                current_events, current_score, current_percentage_error, current_percentage_size, current_absolute_error = self.annealing_iteration(*args, mode=mode)
            os.rename(f'{results_dir}best_result_{explainer.target_index}_{self.exp_size}_{mode}.pck.tmp', f'{results_dir}best_result_{explainer.target_index}_{self.exp_size}_{mode}.pck')

        return self.best_score, self.indices_to_events(self.best_events)

    def add_event(self, current_events, num_events=2):
        new_events = current_events.copy()
        available_events = list(set(self.all_event_indices) - set(new_events))
        new_event = np.random.choice(available_events, num_events, replace=False)
        [new_events.append(e) for e in new_event]
        return new_events

    def remove_event(self, current_events, num_events=2):
        new_events = current_events.copy()
        events_to_remove = np.random.choice(new_events, num_events, replace=False)
        [new_events.remove(e) for e in events_to_remove]
        return new_events

    def replace_event(self, current_events, num_events=2):
        new_events = current_events.copy()
        available_events = list(set(self.all_event_indices) - set(new_events))
        new_event = np.random.choice(available_events, num_events, replace=False)
        events_to_remove = np.random.choice(new_events, num_events, replace=False)
        for i in range(num_events):
            new_events.remove(events_to_remove[i])
            new_events.append(new_event[i])
#        new_events.remove(events_to_remove)
#        new_events.append(new_event)
        return new_events

    def perturb(self, current_events, mode, num_events=5):
        if mode == 'fidelity':
            return self.replace_event(current_events, num_events)
        elif mode == 'fidelity+size':
            if len(current_events) <= 11:
                move = np.random.choice([self.add_event, self.replace_event])
            else:
                move = np.random.choice([self.add_event, self.remove_event, self.replace_event])
            return move(current_events)

if __name__ == '__main__':
    tic = time.time()
    explainer = run_explainer(target_node_index=args.target_node)
    toc = time.time()-tic
#    print("Time taken: ", toc)
    best_exps = []
    best_scores = []

#    with open('results/METR_LA/simulated_annealing/all_results.pck', 'rb') as f:
#        best_exps, best_scores = pck.load(f)
#    best_exps = best_exps[:2]
#    best_scores = best_scores[:2]

    if len(best_exps) == 0:
        candidate_events = explainer.candidate_events
    else:
        candidate_events = best_exps[-1]
    sa = SimulatedAnnealing(explainer, candidate_events, args.subgraph_size)
    num_iter=10000
    score, exp = sa.run(iterations=num_iter)
    best_scores.append(score)
    best_exps.append(exp)
#    with open(f'{results_dir}final_result_{num_iter}_iter_{self.explainer.target_index}_{self.exp_size}.pck', 'wb') as f:
#        pck.dump((best_exps, best_scores), f)

    print("Best scores: ", best_scores)



