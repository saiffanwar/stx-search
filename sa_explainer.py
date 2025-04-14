import numpy as np
import torch
import math
import os
import random
from tqdm import tqdm
import copy
from matplotlib import pyplot as plt
from pprint import pprint
import dill as pck
import time

#from visualisation_utils import graph_visualiser

class SimulatedAnnealing:
    def __init__(self, dataset_name, target_idx, candidate_events, exp_size, score_func, expmode='fidelity', verbose=False):
        self.expmode = expmode
        self.candidate_events = candidate_events
        self.target_idx = target_idx
        self.best_events = []
        if self.expmode == 'fidelity':
            self.starting_temperature = 1
        elif self.expmode == 'fidelity+size':
            self.starting_temperature = 1
        self.temperature = self.starting_temperature
        self.cooling_rate = 0.99
        self.exp_size = exp_size
        self.objective_function = score_func
        self.dataset_name = dataset_name
        self.results_dir = f'{os.getcwd()}/results/{dataset_name}/'
        self.verbose = verbose
        self.max_delta_fidelity = 0

        ## Results Recording
        self.scores = []
        self.errors = []
        self.acceptance_probabilities = []
        self.actions = []
        self.exp_sizes = []


        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)


    def acceptance_probability(self, delta, temperature, alpha=10):
        if delta < 0:
            delta = -delta
        return np.exp(-alpha*(delta)/temperature)


    def annealing_iteration(self, *args, mode, iteration=0):
        current_events, current_score = args

        new_events = self.perturb(current_events)
        new_score, new_absolute_error, model_pred, exp_pred, new_unimportant_y_pred = self.objective_function(new_events, self.target_idx, self.expmode, max_delta_fidelity=self.max_delta_fidelity)
        delta_fidelity = self.objective_function(new_events, self.target_idx, mode='delta_fidelity', max_delta_fidelity=self.max_delta_fidelity)[0]
        if delta_fidelity > self.max_delta_fidelity:
            self.max_delta_fidelity = delta_fidelity

        delta = new_score - current_score

        # Minimising objective function for error and exp size, Maximising for delta fidelity
        if mode in ['fidelity','fidelity+size']:
            delta_criteria = lambda d : d < 0
            score_criteria = lambda s, b_s : s < b_s
        elif mode in ['delta_fidelity']:
            delta_criteria = lambda d : d > 0
            score_criteria = lambda s, b_s : s > b_s

        if delta_criteria(delta):
            current_score = copy.copy(new_score)
            current_events = copy.copy(new_events)

            if score_criteria(new_score, self.best_score):
                self.best_score = copy.copy(new_score)
                self.best_events = copy.copy(new_events)
                self.best_pred = exp_pred
                self.best_delta_fidelity = delta_fidelity
                self.best_unimportant_pred = new_unimportant_y_pred
            self.acceptance_probabilities.append(None)
            self.actions.append(True)

        else:
            acceptance_probability = self.acceptance_probability(delta, self.temperature)
            self.acceptance_probabilities.append(acceptance_probability)
            rand_val = np.random.rand()
            accept = rand_val < acceptance_probability
            if accept:
                current_score = copy.copy(new_score)
                current_events = copy.copy(new_events)
                self.actions.append(True)
            else:
                self.actions.append(False)

        self.exp_sizes.append(len(current_events))
        self.scores.append(float(current_score))
        self.errors.append(abs(model_pred - exp_pred))


        if self.verbose == True:
            if iteration%100==0:
                print('Exp Size: ', len(new_events))
                print("New score: ", new_score)
                print("Current score: ", current_score)
                print("Delta: ", delta)
                print("Temperature: ", self.temperature)
                print('Model pred:', model_pred, 'Exp pred:', exp_pred, 'Unimportant pred:', new_unimportant_y_pred)
                print('Best score: ', self.best_score, 'Best pred:', self.best_pred, 'Best delta fidelity:', self.best_delta_fidelity, 'Best unimportant pred:', self.best_unimportant_pred)


        self.temperature *= self.cooling_rate
#        if iteration%50==0:
#        with open(f'{self.results_dir}best_result_{self.target_idx}_{self.exp_size}_{self.expmode}.pck.tmp', 'wb') as f:
#            pck.dump((self.explainer, self), f)
        return current_events, current_score


    def add_event(self, current_events, num_events=2):
        new_events = copy.copy(current_events)
        available_events = list(set(self.candidate_events) - set(new_events))
        new_event = np.random.choice(available_events, num_events, replace=False)
        [new_events.append(e) for e in new_event]
        return new_events

    def remove_event(self, current_events, num_events=1):
        new_events = copy.copy(current_events)
        events_to_remove = np.random.choice(new_events, num_events, replace=False)
        [new_events.remove(e) for e in events_to_remove]
        return new_events

    def replace_event(self, current_events, num_events=2):
        new_events = copy.copy(current_events)
        available_events = list(set(self.candidate_events) - set(new_events))
        if len(available_events) < num_events:
            return new_events
        new_event = np.random.choice(available_events, num_events, replace=False)
        events_to_remove = np.random.choice(new_events, num_events, replace=False)
        for i in range(num_events):
            new_events.remove(events_to_remove[i])
            new_events.append(new_event[i])
#        new_events.remove(events_to_remove)
#        new_events.append(new_event)
        return new_events

    def perturb(self, current_events, num_events=2):
        if self.expmode in ['fidelity', 'delta_fidelity']:
            return self.replace_event(current_events, num_events)
        elif self.expmode in ['fidelity+size']:
            if len(current_events) <= 2:
                move = np.random.choice([self.add_event, self.replace_event])
            else:
                move = np.random.choice([self.add_event, self.remove_event, self.replace_event])
            try:
                return move(current_events)
            except:
                return self.perturb(current_events)

    def run(self, iterations=10000, expmode='fidelity', best_events=None, explainer=None):
        self.expmode = expmode
        # Initialize with random events if only optimising fidelity
        if self.exp_size >= len(self.candidate_events):
            self.exp_size = int(len(self.candidate_events))

        if self.expmode == 'fidelity':
            initial_events = list(np.random.choice(self.candidate_events, self.exp_size, replace=False))
            initial_events = [int(e) for e in initial_events]
        elif self.expmode in ['fidelity+size', 'delta_fidelity']:
            initial_events = best_events


        initial_score, initial_absolute_error, self.model_pred, exp_pred, unimportant_explanation_y = self.objective_function(initial_events, self.target_idx, self.expmode, max_delta_fidelity=self.max_delta_fidelity)
        self.best_score = initial_score
        self.best_events = initial_events
        self.best_pred = exp_pred
        self.best_score = initial_score
        self.best_pred = exp_pred
        self.best_delta_fidelity = self.objective_function(initial_events, self.target_idx, mode='delta_fidelity', max_delta_fidelity=self.max_delta_fidelity)[0]
        self.best_unimportant_pred = unimportant_explanation_y

        print('Initial score: ', initial_score)
        print('Model pred:', self.model_pred)
        print('Best score: ', self.best_score, 'Best pred:', self.best_pred, 'Best delta fidelity:', self.best_delta_fidelity, 'Best unimportant pred:', self.best_unimportant_pred)
        current_events = initial_events
        current_score = initial_score


        if len(self.best_events) != int(len(self.candidate_events)):
            for i in tqdm(range(iterations)):
                args = (current_events, current_score)
                current_events, current_score = self.annealing_iteration(*args, mode=self.expmode, iteration=i)
                if i%100 == 0:
                    with open(f'results/{self.dataset_name}/best_result_{self.target_idx}_{self.exp_size}_fidelity.pck', 'wb') as f:
                        pck.dump([explainer, self], f)
        self.best_events = [int(e) for e in self.best_events]
        print('Best events: ', sorted(self.best_events))

        return self.best_score, self.best_events, self.model_pred, self.best_pred
#
#class SA_Explainer:
#
#    def __init__(self, model, tgnnexplainer=None, dataset_name='wikipedia', model_name='tgat'):
#        self.task='traffic_state_pred'
#        self.model = model
#        self.model_name=model_name
#        self.dataset=dataset_name
#        self.saved_model=True
#        self.train=False
#        self.other_args={'exp_id': '1', 'seed': 0}
#        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#        self.target_timestamp = 0
#        self.tgnnexplainer = tgnnexplainer
#        self.subgraph_size=50
#
#    def delta_fidelity(self, exp_events, target_index, gamma=0.9, lam=0.1):
#        target_model_y = self.tgnnexplainer.tgnn_reward_wraper.original_scores
#        target_explanation_y = self.tgnnexplainer.tgnn_reward_wraper._get_model_prob(target_index, exp_events, num_neighbors=200)
#
#        remaining_events = list(set(self.tgnnexplainer.computation_graph_events) - set(exp_events))
#        remaining_explanation_y = self.tgnnexplainer.tgnn_reward_wraper._get_model_prob(target_index, remaining_events, num_neighbors=200)
#
##        delta_fidelity = lam*abs(remaining_explanation_y - target_model_y) - gamma*abs(target_explanation_y - target_model_y)
#        if abs(target_explanation_y - target_model_y) == 0:
#            delta_fidelity = np.inf
#        else:
#            delta_fidelity = abs(remaining_explanation_y - target_model_y)/abs(target_explanation_y - target_model_y)
#
#        return delta_fidelity, remaining_explanation_y, target_model_y, target_explanation_y
#
#    def score_func(self, exp_events, mode='fidelity', d_gam=0.9, d_lam=0.1, max_delta_fidelity=None):
#        '''
#        Calculate the fidelity of the model for a given subgraph. Fidelity is defined using the
#        metric proposed in arXiv:2306.05760.
#
#        Args:
#            subgraph: A list of graphNode() objects that are part of the computation graph.
#
#        Returns:
#            fidelity: The fidelity of the model for the given subgraph.
#        '''
#        delta_fidelity, unimportant_explanation_y, target_model_y, target_explanation_y  = self.delta_fidelity(exp_events, self.target_index, d_gam, d_lam)
#
#        exp_absolute_error = abs(target_model_y - target_explanation_y)
#        max_exp_size = self.subgraph_size
#        exp_size_percentage = 100*len(exp_events)/max_exp_size
#        exp_percentage_error = 100*abs(exp_absolute_error/target_model_y)
#
#        if mode == 'fidelity':
##            exp_score = exp_percentage_error
#            exp_score = exp_absolute_error
##            exp_score = self.delta_fidelity(exp_events, self.target_index)
#        elif mode == 'delta_fidelity':
#            exp_score = delta_fidelity
#        elif mode == 'fidelity+size':
#            lam = self.lam
#            gam = self.gamma
#            exp_score = gam*exp_percentage_error + lam*exp_size_percentage
##            exp_score =  (delta_fidelity/max_delta_fidelity)+ lam*exp_size_percentage
#        else:
#            RuntimeError('Invalid mode. Choose either fidelity or fidelity+size')
#        return exp_score, exp_absolute_error, target_model_y, target_explanation_y, unimportant_explanation_y
#
#    def __call__(self, event_idxs, num_iter=500, results_dir=None, results_batch=None):
#        testing_gammas = True
#        testing_sparsity = False
#
#        rb = [str(results_batch) if results_batch is not None else ''][0]
#        print(f'Running results batch {rb} with {len(event_idxs)} events')
#
#        if testing_gammas:
#            gammas = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
#            sa_results_gammas = {g: {'target_idxs': [], 'explanations': [], 'explanation_predictions': [], 'model_predictions': [], 'delta_fidelity': []} for g in gammas}
#            filename = f'/sa_results_{self.dataset}_{self.model_name}_gammas_{rb}'
##            exp_sizes = [self.subgraph_size]
#
#        if testing_sparsity:
##            exp_sizes = [10,20,30,40, 50, 60, 70, 80, 90, 100]
#            exp_sizes = [10,25,50,75,100]
#            sa_results_exp_sizes = {s: {'target_idxs': [], 'explanations': [], 'explanation_predictions': [], 'model_predictions': [], 'delta_fidelity': []} for s in exp_sizes}
#            filename = f'/sa_results_{self.dataset}_{self.model_name}_exp_sizes_{rb}'
#
#
#        for target_index in event_idxs:
##            try:
#                print(f'---- Explaining event: {event_idxs.index(target_index)} out of {len(event_idxs)} ------')
#                self.target_index = target_index
##            self.tgnnexplainer.model.eval()
#
#                self.tgnnexplainer._initialize(self.target_index)
#                self.candidate_events = self.tgnnexplainer.computation_graph_events
#                print(len(self.candidate_events))
#                original_score = self.tgnnexplainer.tgnn_reward_wraper.compute_original_score(self.candidate_events, self.target_index)
#
#                if testing_sparsity:
#                    if len(self.candidate_events) > max(exp_sizes):
#                        for exp_size in exp_sizes:
#                            sa = SimulatedAnnealing(self, self.target_index, self.candidate_events, exp_size, score_func=self.score_func, verbose=True)
##                sa.reinitialize()
#                            score, exp, model_pred, exp_pred = sa.run(iterations=num_iter, expmode='fidelity')
##                    sa.reinitialize()
#                            score, exp, model_pred, exp_pred = sa.run(iterations=num_iter, expmode='delta_fidelity', best_events=exp)
#                            if exp_size == 50:
#                                g_score, g_exp, g_model_pred, g_exp_pred = copy.copy(score), copy.copy(exp), copy.copy(model_pred), copy.copy(exp_pred)
#
#                            delta_fidelity = self.delta_fidelity(exp, self.target_index)[0]
#                            print('Score: ', score, 'Exp Length: ', len(exp), 'Model Pred: ', model_pred, 'Exp Pred: ', exp_pred, 'Delta Fidelity: ', delta_fidelity)
#
#                            sa_results_exp_sizes[exp_size]['target_idxs'].append(target_index)
#                            sa_results_exp_sizes[exp_size]['explanations'].append(exp)
#                            sa_results_exp_sizes[exp_size]['explanation_predictions'].append(exp_pred)
#                            sa_results_exp_sizes[exp_size]['model_predictions'].append(model_pred)
#                            sa_results_exp_sizes[exp_size]['delta_fidelity'].append(delta_fidelity)
#
#                        with open(results_dir + f'/intermediate_results/sa_results_{self.dataset}_{self.model_name}_exp_sizes_{rb}.pkl', 'wb') as f:
#                            pck.dump(sa_results_exp_sizes, f)
#                else:
#                    sa = SimulatedAnnealing(self, self.target_index, self.candidate_events, self.subgraph_size, score_func=self.score_func, verbose=True)
#                    score, exp, model_pred, exp_pred = sa.run(iterations=num_iter, expmode='fidelity')
#                    score, exp, model_pred, exp_pred = sa.run(iterations=num_iter, expmode='delta_fidelity', best_events=exp)
#                    g_score, g_exp, g_model_pred, g_exp_pred = copy.copy(score), copy.copy(exp), copy.copy(model_pred), copy.copy(exp_pred)
#
#                if testing_gammas:
#                    for gamma in gammas:
#
#                        print(f'----- Explaining with Gamma: {gamma} -----')
#                        self.gamma=gamma
#                        self.lam=1-gamma
##                        sa.reinitialize()
#                        t_score, t_exp, t_model_pred, t_exp_pred = sa.run(iterations=num_iter, expmode='fidelity+size', best_events=g_exp)
#                        delta_fidelity = self.delta_fidelity(t_exp, self.target_index)[0]
#                        print('Score: ', t_score, 'Exp Length: ', len(t_exp), 'Model Pred: ', t_model_pred, 'Exp Pred: ', t_exp_pred, 'Delta Fidelity: ', delta_fidelity)
#                        if abs(t_exp_pred - t_model_pred) != 0:
#                            sa_results_gammas[gamma]['target_idxs'].append(target_index)
#                            sa_results_gammas[gamma]['explanations'].append(t_exp)
#                            sa_results_gammas[gamma]['explanation_predictions'].append(t_exp_pred)
#                            sa_results_gammas[gamma]['model_predictions'].append(t_model_pred)
#                            sa_results_gammas[gamma]['delta_fidelity'].append(delta_fidelity)
#
#                    with open(results_dir + f'/intermediate_results/sa_results_{self.dataset}_{self.model_name}_gammas_{rb}.pkl', 'wb') as f:
#                        pck.dump(sa_results_gammas, f)
#



#            except:
#                pass

#        with open(results_dir + f'/sa_results_{self.dataset}_{self.model_name}_gammas_{rb}.pkl', 'wb') as f:
#            pck.dump(sa_results_gammas, f)
