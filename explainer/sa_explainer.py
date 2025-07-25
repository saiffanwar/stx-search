import copy
import os
import pickle as pck

import numpy as np
from tqdm import tqdm

# from visualisation_utils import graph_visualiser


class SimulatedAnnealing:
    def __init__(
        self,
        data,
        graph_events,
        adj_mx,
        model_name,
        dataset_name,
        target_idx,
        explaining_event_idx,
        candidate_events,
        exp_size,
        score_func,
        expmode="fidelity",
        verbose=False,
    ):
        self.data = data
        self.graph_events = graph_events
        self.input_window = np.shape(data)[1]
        self.adj_mx = adj_mx
        self.model_name = model_name
        self.dataset_name = dataset_name

        self.expmode = expmode
        self.candidate_events = candidate_events
        self.target_idx = target_idx
        self.explaining_event_idx = explaining_event_idx
        self.best_events = []
        if self.expmode == "fidelity":
            self.starting_temperature = 1
        elif self.expmode == "fidelity+size":
            self.starting_temperature = 1
        self.temperature = self.starting_temperature
        self.cooling_rate = 0.99
        self.exp_size = exp_size
        self.objective_function = score_func
        self.results_dir = f"{os.getcwd()}/results/{dataset_name}/"
        self.verbose = verbose

        # Results Recording
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
        return np.exp(-alpha * (delta) / temperature)

    def annealing_iteration(self, iteration=0):
        new_events = self.perturb(self.current_events)
        (
            new_error,
            new_fidelity_size_score,
            new_delta_fidelity,
            model_y,
            target_model_y,
            exp_y,
            target_exp_y,
            target_complement_exp_y,
        ) = self.objective_function(new_events)

        if self.expmode == "fidelity":
            new_score = new_error
        elif self.expmode == "fidelity+size":
            new_score = new_fidelity_size_score
        elif self.expmode == "delta_fidelity":
            new_score = new_delta_fidelity

        delta = new_score - self.score

        # Minimising objective function for error and exp size, Maximising for delta fidelity
        if self.expmode in ["fidelity", "fidelity+size"]:

            def delta_criteria(d):
                return d < 0

            def score_criteria(s, b_s):
                return s < b_s
        elif self.expmode in ["delta_fidelity"]:

            def delta_criteria(d):
                return d > 0

            def score_criteria(s, b_s):
                return s > b_s

        if delta_criteria(delta):
            # If the new_score is better than the current_score, accept it as the new current state
            self.score = copy.copy(new_score)
            self.current_events = copy.copy(new_events)

            # If it is better than the best score, update the best score and explanation events.
            if score_criteria(new_score, self.best_score):
                self.best_events = copy.copy(new_events)
                self.best_score = copy.copy(self.score)
                self.target_exp_y = target_exp_y
                #                self.best_delta_fidelity = delta_fidelity
                #                self.target_complement_exp_y = target_complement_exp_y
                self.target_exp_y = target_exp_y
                self.exp_y = exp_y
            self.acceptance_probabilities.append(None)
            self.actions.append(True)

        else:
            acceptance_probability = self.acceptance_probability(
                delta, self.temperature
            )
            self.acceptance_probabilities.append(acceptance_probability)
            rand_val = np.random.rand()
            accept = rand_val < acceptance_probability
            if accept:
                self.score = copy.copy(new_score)
                self.current_events = copy.copy(new_events)
                self.actions.append(True)
            else:
                self.actions.append(False)

        self.exp_sizes.append(len(new_events))
        self.scores.append(float(self.score))
        self.errors.append(abs(target_model_y - target_exp_y))

        if self.verbose == True:
            if iteration % 100 == 0:
                print("Exp Size: ", len(self.current_events))
                print("Score: ", self.score)
                print("Temperature: ", self.temperature)
                print(
                    "Model pred:",
                    self.target_model_y,
                    "Exp pred:",
                    self.target_exp_y,
                    "Unimportant pred:",
                    target_complement_exp_y,
                )
        #                print('Best score: ', self.best_score, 'Best pred:', self.target_exp_y, 'Best delta fidelity:', self.best_delta_fidelity, 'Best unimportant pred:', self.best_unimportant_pred)

        self.temperature *= self.cooling_rate

    #        if iteration%50==0:
    #        with open(f'{self.results_dir}best_result_{self.target_idx}_{self.exp_size}_{self.expmode}.pck.tmp', 'wb') as f:
    #            pck.dump((self.explainer, self), f)

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
        if self.expmode in ["delta_fidelity", "fidelity"]:
            return self.replace_event(current_events, num_events)
        elif self.expmode in ["fidelity+size"]:
            if len(current_events) <= 2:
                move = np.random.choice([self.add_event, self.replace_event])
            else:
                move = np.random.choice(
                    [self.add_event, self.remove_event, self.replace_event]
                )
            try:
                return move(current_events)
            except:
                return self.perturb(current_events)

    def visualisation_data(
        self,
    ):
        results = {}
        results["target_idx"] = self.target_idx
        results["x_data"] = self.data
        results["input_window"] = self.input_window
        results["adj_mx"] = self.adj_mx
        results["candidate_events"] = self.candidate_events
        results["graph_events"] = self.graph_events

        results["events"] = self.best_events

        results["scores"] = self.scores
        results["errors"] = self.errors
        results["exp_sizes"] = self.exp_sizes
        results["probabilities"] = self.acceptance_probabilities

        results["model_y"] = self.model_y
        results["target_model_y"] = self.target_model_y.item()
        results["exp_y"] = self.exp_y.detach().cpu().numpy()
        results["target_exp_y"] = self.exp_y[0, 0, self.target_idx, 0].item()

        return results

    def run(self, iterations=10000, expmode="fidelity", best_events=None):
        self.expmode = expmode
        # Initialize with random events if only optimising fidelity
        if self.exp_size >= len(self.candidate_events):
            self.exp_size = int(len(self.candidate_events))

        if self.expmode == "fidelity":
            initial_events = list(
                np.random.choice(self.candidate_events, self.exp_size, replace=False)
            )
            initial_events = [int(e) for e in initial_events]
        elif self.expmode in ["fidelity+size", "delta_fidelity"]:
            with open(
                f"results/{self.dataset_name}/stx_search/stx_search_{self.model_name}_{self.dataset_name}_{self.explaining_event_idx}_{self.exp_size}.pck",
                "rb",
            ) as f:
                best_result = pck.load(f)
            initial_events = best_result["events"]

        self.best_events = initial_events
        self.current_events = initial_events

        (
            self.error,
            self.fidelity_size_error,
            self.delta_fidelity,
            self.model_y,
            self.target_model_y,
            self.exp_y,
            self.target_exp_y,
            self.target_complement_exp_y,
        ) = self.objective_function(initial_events)

        self.model_y = self.model_y.detach().cpu().numpy()

        if self.expmode == "fidelity":
            self.best_score = self.error
        elif self.expmode == "fidelity+size":
            self.best_score = self.fidelity_size_error
        elif self.expmode == "delta_fidelity":
            self.best_score = self.delta_fidelity

        self.score = self.best_score

        #        print('Initial score: ', initial_score)
        #        print('Model pred:', self.model_pred)
        #        print('Best score: ', self.best_score, 'Best pred:', self.best_pred, 'Best delta fidelity:', self.best_delta_fidelity, 'Best unimportant pred:', self.best_unimportant_pred)

        if len(self.best_events) != int(len(self.candidate_events)):
            for i in range(iterations):
                self.annealing_iteration(iteration=i)
                if self.best_score < 0.00001:
                    print(
                        f"Early stopping at iteration {i} with score {self.best_score}"
                    )
                    with open(
                        f"results/{self.dataset_name}/stx_search/stx_search_{self.model_name}_{self.dataset_name}_{self.explaining_event_idx}_{self.exp_size}.pck",
                        "wb",
                    ) as f:
                        pck.dump(self.visualisation_data(), f)
                    break

                if i % 100 == 0:
                    with open(
                        f"results/{self.dataset_name}/stx_search/stx_search_{self.model_name}_{self.dataset_name}_{self.explaining_event_idx}_{self.exp_size}.pck",
                        "wb",
                    ) as f:
                        pck.dump(self.visualisation_data(), f)
        self.best_events = [int(e) for e in self.best_events]

        return self.best_score, self.best_events, self.target_model_y, self.target_exp_y
