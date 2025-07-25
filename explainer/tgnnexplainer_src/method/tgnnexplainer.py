import copy
import math
import os
import pickle as pck
import random
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from tqdm import tqdm

from tgnnexplainer_libcity import TGNNExplainer_LibCity
from tgnnexplainer_src.method.base_explainer_tg import BaseExplainerTG
from tgnnexplainer_src.method.other_baselines_tg import _create_explainer_input

sys.setrecursionlimit(2000)


def to_networkx_tg(events: DataFrame):
    base = events.iloc[:, 0].max() + 1
    g = nx.MultiGraph()
    g.add_nodes_from(events.iloc[:, 0])
    g.add_nodes_from(events.iloc[:, 1] + base)
    t_edges = []
    for i in range(len(events)):
        user, item, t, e_idx = (
            events.iloc[i, 0],
            events.iloc[i, 1],
            events.iloc[i, 2],
            events.index[i],
        )
        t_edges.append(
            (
                user,
                item,
                {"t": t, "e_idx": i},
            )
        )
    g.add_edges_from(t_edges)
    return g


def print_nodes(tree_nodes):
    print("\nSearched tree nodes (preserved edge idxs in candidates):")
    for i, node in enumerate(tree_nodes):
        # preserved_events = preserved_candidates(node.coalition, ori_event_idxs, candidates_idxs)
        # removed_idxs = obtain_removed_idxs(node.coalition, self.ori_subgraph_df.index.to_list())
        # preserved_events_gnn_score = self.tgnn_reward_wraper(preserved_events, event_idx)
        print(i, sorted(node.coalition), ": ", node.P)


def find_best_node_result(all_nodes, min_atoms=6, candidate_events=None, exp_size=40):
    """return the highest reward tree_node with its subgraph is smaller than max_nodes"""
    # if candidate_events is None:
    # filter using the min_atoms
    all_nodes = filter(lambda x: len(x.coalition) <= min_atoms, all_nodes)
    # else:
    #     if len(candidate_events) <= exp_size:
    #         exp_size = len(candidate_events)-1
    #     # filter using the min_atoms
    #     all_nodes = filter(lambda x: len(x.coalition) == exp_size, all_nodes)

    best_node = min(all_nodes, key=lambda x: x.P)
    return best_node

    # all_nodes = sorted(all_nodes, key=lambda x: len(x.coalition))
    # result_node = all_nodes[0]
    # for result_idx in range(len(all_nodes)):
    #     x = all_nodes[result_idx]
    #     # if len(x.coalition) <= max_nodes and x.P > result_node.P:
    #     if x.P > result_node.P:
    #         result_node = x
    # return result_node


class MCTSNode(object):
    def __init__(
        self,
        coalition: list = None,
        created_by_remove: int = None,
        c_puct: float = 10.0,
        W: float = 0,
        N: int = 0,
        P: float = 100,
        Sparsity: float = 1,
        Prediction=None,
    ):
        self.coalition = coalition  # in our case, the coalition should be edge indices?
        self.c_puct = c_puct
        self.children = []
        # created by remove which edge from its parents
        self.created_by_remove = created_by_remove
        self.W = W  # sum of node value
        self.N = N  # times of arrival
        self.P = P  # property score (reward)
        self.Sparsity = Sparsity  # len(self.coalition)/len(candidates)
        self.Prediction = Prediction

    def Q(self):
        return self.W / self.N if self.N > 0 else 0

    def U(self, n):
        # return self.c_puct * self.P * math.sqrt(n) / (1 + self.N)
        return self.c_puct * math.sqrt(n) / (1 + self.N)

    @property
    def info(self):
        info_dict = {
            "coalition": self.coalition,
            "created_by_remove": self.created_by_remove,
            "c_puct": self.c_puct,
            "W": self.W,
            "N": self.N,
            "P": self.P,
            "Sparsity": self.Sparsity,
            "Prediction": self.Prediction,
        }
        return info_dict

    def load_info(self, info_dict):
        self.coalition = info_dict["coalition"]
        self.created_by_remove = info_dict["created_by_remove"]
        self.c_puct = info_dict["c_puct"]
        self.W = info_dict["W"]
        self.N = info_dict["N"]
        self.P = info_dict["P"]
        self.Sparsity = info_dict["Sparsity"]
        self.Prediction = info_dict["Prediction"]

        self.children = []
        return self


def compute_scores(score_func, base_events, children, target_event_idx):
    results = []
    for child in children:
        if child.P == 100:
            # score = score_func(child.coalition, child.data)
            #            score = score_func( child.coalition, target_event_idx)
            score, exp_pred = score_func(child.coalition)
        else:
            score = child.P
        results.append(score)
    return results


def base_and_important_events(base_events, candidate_events, coalition):
    return base_events + coalition


def base_and_unimportant_events(base_events, candidate_events, coalition):
    important_ = set(coalition)
    unimportant_events = list(filter(lambda x: x not in important_, candidate_events))
    return base_events + unimportant_events


class MCTS(object):
    r"""
    Monte Carlo Tree Search Method.
    Args:
        n_rollout (:obj:`int`): The number of sequence to build the monte carlo tree.
        min_atoms (:obj:`int`): The number of atoms for the subgraph in the monte carlo tree leaf node. here is number of events preserved in the candidate events set.
        c_puct (:obj:`float`): The hyper-parameter to encourage exploration while searching.
        expand_atoms (:obj:`int`): The number of children to expand.
        high2low (:obj:`bool`): Whether to expand children tree node from high degree nodes to low degree nodes.
        node_idx (:obj:`int`): The target node index to extract the neighborhood.
        score_func (:obj:`Callable`): The reward function for tree node, such as mc_shapely and mc_l_shapely.
    """

    def __init__(
        self,
        events: DataFrame,
        candidate_events=None,
        computation_graph_events=None,
        base_events=None,
        candidate_initial_weights=None,
        node_idx: int = None,
        event_idx: int = None,
        n_rollout: int = 10,
        min_atoms: int = 5,
        c_puct: float = 10.0,
        score_func: Callable = None,
        libcity_base_explainer=None,
        #  device='cpu'
    ):
        self.events = events  # subgraph events or total events? subgraph events
        # self.num_users = num_users
        # self.subgraph_num_nodes = self.events.iloc[:, 0].nunique(
        # ) + self.events.iloc[:, 1].nunique()
        # self.graph = to_networkx_tg(events)
        # self.node_X = node_X # node features
        # self.event_X = event_X # event features
        self.node_idx = node_idx  # node index to explain
        self.event_idx = event_idx  # event index to explain
        # improve the strategy later
        # self.candidate_events = sorted(self.events.index.values.tolist())[-6:-1]
        # self.candidate_events = sorted(self.events.index.values.tolist())[-10:]
        # self.candidate_events = [10, 11, 12, 13, 14, 15, 19]
        self.candidate_events = candidate_events
        self.computation_graph_events = computation_graph_events
        self.base_events = base_events
        self.libcity_base_explainer = libcity_base_explainer

        """
        Set the base events to be all events not in the computation graph,             the explanation performance should then be measured only on the
        included events.
        """
        self.candidate_initial_weights = candidate_initial_weights

        # we only care these events, other events are preserved as is.
        # currently only take 10 temporal edges into consideration.

        # self.device = device
        # self.num_nodes = self.events.iloc[:, 0].nunique(
        # ) + self.events.iloc[:, 1].nunique()

        self.score_func = score_func

        self.n_rollout = n_rollout
        self.min_atoms = min_atoms
        self.c_puct = c_puct
        # self.expand_atoms = expand_atoms
        # self.high2low = high2low
        self.new_node_idx = None
        # self.data = None

        # self.MCTSNodeClass = partial(MCTSNode,
        #                              c_puct=self.c_puct,
        #                              )

        self._initialize_tree()
        self._initialize_recorder()

    def _initialize_recorder(self):
        self.recorder = {
            "rollout": [],
            "runtime": [],
            "best_reward": [],
            "num_states": [],
        }

    def mcts_rollout(self, tree_node):
        """
        The tree_node now is a set of events
        """
        # import ipdb; ipdb.set_trace()
        # print('mcts_rollout: ', len(tree_node.coalition))
        if len(tree_node.coalition) < self.min_atoms:
            # if len(tree_node.coalition) < 1:
            return tree_node.P  # its score

        # Expand if this node has never been visited
        # Expand if this node has un-expanded children
        if len(tree_node.children) != len(tree_node.coalition):
            # expand_events = tree_node.coalition

            exist_children = set(map(lambda x: x.created_by_remove, tree_node.children))
            not_exist_children = list(
                filter(lambda e_idx: e_idx not in exist_children, tree_node.coalition)
            )
            # print('not_exist_children:', not_exist_children)
            # print('exist_children:', exist_children)
            expand_events = self._select_expand_candidates(not_exist_children)
            # print('expand_events:', expand_events)

            # not_exist_children_score = {}
            # for event in not_exist_children:
            #     children_coalition = [e_idx for e_idx in treo_node.coalition if e_idx != event ]
            #     not_exist_children_score[event] = self.compute_action_score(children_coalition, expand_event=event)
            # # expand only one event
            # # expand_event = max( not_exist_children_score, key=not_exist_children_score.get )
            # expand_event = min( not_exist_children_score, key=not_exist_children_score.get ) # NOTE: min

            # expand_events = [expand_events[0], ]

            for event in expand_events:
                important_events = [
                    e_idx for e_idx in tree_node.coalition if e_idx != event
                ]

                # check the state map and merge the same sub-tg-graph (node in the tree)
                find_same = False
                subnode_coalition_key = self._node_key(important_events)
                for key in self.state_map.keys():
                    if key == subnode_coalition_key:
                        new_tree_node = self.state_map[key]
                        find_same = True
                        break

                if not find_same:
                    # new_tree_node = self.MCTSNodeClass(
                    #     coalition=important_events, created_by_remove=event)
                    exp_fidelity_inv, exp_pred = self.score_func(important_events)
                    new_tree_node = MCTSNode(
                        coalition=important_events,
                        created_by_remove=event,
                        c_puct=self.c_puct,
                        #                        Sparsity=len(important_events)/len(self.candidate_events),
                        Sparsity=len(important_events)
                        / len(self.computation_graph_events),
                        Prediction=exp_pred,
                    )

                    self.state_map[subnode_coalition_key] = new_tree_node

                # find same child ?
                find_same_child = False
                for child in tree_node.children:
                    if self._node_key(child.coalition) == self._node_key(
                        new_tree_node.coalition
                    ):
                        find_same_child = True
                        break
                if not find_same_child:
                    tree_node.children.append(new_tree_node)

                # coutinue until one valid child is expanded, otherewize this rollout will be wasted
                if not find_same:
                    break
                else:
                    continue

            # compute scores of all children
            #            print(len(self.base_events))
            #            self.base_events = []
            scores = compute_scores(
                self.score_func, self.base_events, tree_node.children, self.event_idx
            )
            # print(scores)
            # import ipdb; ipdb.set_trace()
            for child, score in zip(tree_node.children, scores):
                child.P = score

        # import ipdb; ipdb.set_trace()

        # If this node has children (it has been visited), then directly select one child
        sum_count = sum([c.N for c in tree_node.children])
        # import ipdb; ipdb.set_trace()
        # selected_node = max(tree_node.children, key=lambda x: x.Q() + x.U(sum_count))
        selected_node = min(
            tree_node.children, key=lambda x: self._compute_node_score(x, sum_count)
        )

        v = self.mcts_rollout(selected_node)
        selected_node.W += v
        selected_node.N += 1
        return v

    def _select_expand_candidates(self, not_exist_children):
        assert self.candidate_initial_weights is not None
        # return sorted(not_exist_children, key=self.candidate_initial_weights.get)

        # if self.candidate_initial_weights is not None:
        # return min(not_exist_children, key=self.candidate_initial_weights.get)

        # v1
        if np.random.random() > 0.5:
            return sorted(not_exist_children, key=self.candidate_initial_weights.get)
        else:
            return random.sample(not_exist_children, len(not_exist_children))

            # v2
            # return sorted(not_exist_children, key=self.candidate_initial_weights.get) # ascending

        # else:
        #     # return np.random.choice(not_exist_children)
        #     # return sorted(not_exist_children)[0]
        #     return shuffle(not_exist_children)

    def _compute_node_score(self, node, sum_count):
        """
        score for selecting a path
        """
        # import ipdb; ipdb.set_trace()
        # time score
        # tscore_eff = -10 # 0.1
        # tscore_coef = 0.1 # -100, -50, -10, -5, -1, 0, 0.5
        tscore_coef = 0
        beta = -3

        max_event_idx = min(self.root.coalition)
        curr_t = self.libcity_base_explainer.events[max_event_idx].timestamp
        ts = np.array(
            [
                self.libcity_base_explainer.events[e_idx].timestamp
                for e_idx in node.coalition
            ]
        )
        # ts = self.events['ts'][self.events.e_idx.isin(node.coalition)].values
        # np.array(node.coalition)-1].values # np array
        delta_ts = curr_t - ts
        t_score_exp = np.exp(beta * delta_ts)
        t_score_exp = np.sum(t_score_exp)

        # uct score
        uct_score = node.Q() + node.U(sum_count)

        # final score
        final_score = uct_score + tscore_coef * t_score_exp

        return final_score

    def mcts(self, verbose=False):
        start_time = time.time()
        pbar = tqdm(range(self.n_rollout), total=self.n_rollout, desc="mcts simulating")
        #        for rollout_idx in range(self.n_rollout):
        for rollout_idx in pbar:
            self.mcts_rollout(self.root)
            if verbose:
                elapsed_time = time.time() - start_time
            pbar.set_postfix({"states": len(self.state_map)})
            # print(f"At the {rollout_idx} rollout, {len(self.state_map)} states have been explored. Time: {elapsed_time:.2f} s")

            # record
            self.recorder["rollout"].append(rollout_idx)
            self.recorder["runtime"].append(elapsed_time)
            # self.recorder['best_reward'].append( np.max(list(map(lambda x: x.P, self.state_map.values()))) )
            curr_best_node = find_best_node_result(
                self.state_map.values(), self.min_atoms
            )
            self.recorder["best_reward"].append(curr_best_node.P)
            self.recorder["num_states"].append(len(self.state_map))

        end_time = time.time()
        self.run_time = end_time - start_time

        tree_nodes = list(self.state_map.values())

        return tree_nodes

    def _initialize_tree(self):
        # reset the search tree
        # self.root_coalition = self.events.index.values.tolist()
        print("Candidate events:", len(self.candidate_events))
        self.root_coalition = copy.copy(self.candidate_events)
        self.root = MCTSNode(
            self.root_coalition, created_by_remove=-1, c_puct=self.c_puct, Sparsity=1.0
        )
        self.root_key = self._node_key(self.root_coalition)
        self.state_map = {self.root_key: self.root}

        max_event_idx = min(self.root.coalition)
        # self.curr_t = self.events['ts'][self.events.e_idx ==
        #                                 max_event_idx].values[0]
        self.curr_t = self.libcity_base_explainer.events[max_event_idx].timestamp

    def _node_key(self, coalition):
        # NOTE: have sorted
        return "_".join(map(lambda x: str(x), sorted(coalition)))


class TGNNExplainer(BaseExplainerTG):
    """
    MCTS based temporal graph GNN explainer
    """

    def __init__(
        self,
        model_name: str,
        explainer_name: str,
        dataset_name: str,
        all_events: DataFrame,
        explanation_level: str,
        device,
        verbose: bool = True,
        results_dir="",
        debug_mode: bool = False,
        # specific params
        rollout: int = 20,
        min_atoms: int = 20,
        c_puct: float = 10.0,
        # expand_atoms=14,
        load_results=False,
        mcts_saved_dir: Optional[str] = "",
        save_results: bool = True,
        pg_explainer_model=None,
        pg_positive=True,
        edge_feat=None,
        candidate_events_num=100,
    ):
        super(TGNNExplainer, self).__init__(
            model_name=model_name,
            explainer_name=explainer_name,
            dataset_name=dataset_name,
            all_events=all_events,
            explanation_level=explanation_level,
            device=device,
            verbose=verbose,
            results_dir=results_dir,
            debug_mode=debug_mode,
            edge_feat=edge_feat,
        )

        print("initialised candidate num: ", candidate_events_num)
        # mcts hyper-parameters
        self.rollout = rollout
        self.min_atoms = min_atoms
        self.c_puct = c_puct
        self.all_events = all_events

        # saving and visualization
        self.load_results = load_results
        # dir for saving mcts nodes, not evaluation results ( e.g., fidelity )
        self.mcts_saved_dir = mcts_saved_dir
        # self.mcts_saved_filename = mcts_saved_filename
        self.save = save_results
        # to assign initial weights using a trained pg_explainer_tg
        self.pg_explainer_model = pg_explainer_model
        self.pg_positive = pg_positive
        self.suffix = self._path_suffix(pg_explainer_model, pg_positive)
        self.candidate_events_num = candidate_events_num

    @staticmethod
    def read_from_MCTSInfo_list(MCTSInfo_list):
        if isinstance(MCTSInfo_list[0], dict):
            ret_list = [MCTSNode().load_info(node_info) for node_info in MCTSInfo_list]
        else:
            raise NotImplementedError
        return ret_list

    def write_from_MCTSNode_list(self, MCTSNode_list):
        if isinstance(MCTSNode_list[0], MCTSNode):
            ret_list = [node.info for node in MCTSNode_list]
        else:
            raise NotImplementedError
        return ret_list

    def explain(
        self,
        node_idx: Optional[int] = None,
        time: Optional[float] = None,
        event_idx: Optional[int] = None,
        exp_sizes: list = [10],
    ):
        #        self.base_events, _ = self.find_candidates(event_idx, num_neighbors=600)
        #        self.base_events = list(set(self.base_events))
        #        print(type(self.base_events))
        # support event-level first
        if self.explanation_level == "node":
            raise NotImplementedError
            # node_idx + event_idx?

        # we now only care node/edge(event) level explanations, graph-level explanation is temporarily suspended
        elif self.explanation_level == "event":
            assert event_idx is not None
            # search
            self.mcts_state_map = MCTS(
                events=self.computation_graph_events,
                candidate_events=self.candidate_events,
                computation_graph_events=self.computation_graph_events,
                base_events=self.base_events,
                node_idx=node_idx,
                event_idx=event_idx,
                n_rollout=self.rollout,
                min_atoms=self.min_atoms,
                c_puct=self.c_puct,
                score_func=self.libcity_base_explainer.tgnne_score_func,
                #    device=self.device,
                # BUG: never pass through this parameter?????
                candidate_initial_weights=self.candidate_initial_weights,
                libcity_base_explainer=self.libcity_base_explainer,
            )

            if self.debug_mode:
                print("search graph:")
                print(self.ori_subgraph_df.to_string(max_rows=50))
                # print(f'{len(self.candidate_events)} candicate events:', self.mcts_state_map.candidate_events)
            tree_nodes = self.mcts_state_map.mcts(verbose=self.verbose)  # search
            print(
                f"Total number of nodes explored: {len(self.mcts_state_map.state_map)}"
            )

        else:
            raise NotImplementedError("Wrong explanaion level")

        explanation_results = {
            "important_events": [],
            "target_model_y": [
                # if len(self.candidate_events) > max(exp_sizes):
            ],
            "exp_pred": [],
        }
        # for exp_size in exp_sizes:
        tree_node_x = find_best_node_result(
            tree_nodes, self.min_atoms, self.computation_graph_events
        )

        print(
            f"Best Exp Score: {tree_node_x.P}, Exp Size: {len(tree_node_x.coalition)}"
        )
        # important_events = tree_node_x.coalition
        # exp_fidelity_inv, exp_pred = self.tgnn_reward_wraper(
        #     important_events, event_idx, final_result=True)
        # unimportant_events = [
        #     e_idx for e_idx in self.candidate_events if e_idx not in important_events]
        # _, unimportant_pred = self.tgnn_reward_wraper(
        #     unimportant_events, event_idx, final_result=True)
        #
        # target_model_y = self.tgnn_reward_wraper.original_scores
        #
        # if target_model_y == exp_pred:
        #     delta_fidelity = np.inf
        # else:
        #     delta_fidelity = abs(
        #         target_model_y - unimportant_pred)/abs(target_model_y - exp_pred)
        #
        # tree_nodes = sorted(tree_nodes, key=lambda x: x.P)

        if self.debug_mode:
            print_nodes(tree_nodes)
        target_model_y, exp_pred = self.libcity_base_explainer.tgnne_score_func(
            tree_node_x.coalition
        )
        explanation_results["important_events"] = tree_node_x.coalition
        explanation_results["target_model_y"] = target_model_y
        explanation_results["exp_pred"] = exp_pred
        with open(
            f"results/{self.dataset_name}/tgnnexplainer/tgnnexplainer_{self.model_name}_{self.dataset_name}_{event_idx}_{self.min_atoms}.pkl",
            "wb",
        ) as f:
            pck.dump(explanation_results, f)

        # print('Exp Len: ', len(important_events), 'Model Pred: ', target_model_y, 'Exp Pred: ',
        #       exp_pred, 'Unimportant Pred', unimportant_pred, 'Delta Fidelity: ', delta_fidelity)

        # return explanation_results

    @staticmethod
    def _path_suffix(pg_explainer_model, pg_positive):
        if pg_explainer_model is not None:
            suffix = "pg_true"
        else:
            suffix = "pg_false"

        if pg_explainer_model is not None:
            if pg_positive is True:
                suffix += "_pg_positive"
            else:
                suffix += "_pg_negative"

        return suffix

    @staticmethod
    def _mcts_recorder_path(result_dir, model_name, dataset_name, event_idx, suffix):
        if suffix is not None:
            record_filename = (
                result_dir
                / f"{model_name}_{dataset_name}_{event_idx}_mcts_recorder_{suffix}.csv"
            )
        else:
            record_filename = (
                result_dir
                / f"{model_name}_{dataset_name}_{event_idx}_mcts_recorder.csv"
            )

        return record_filename

    @staticmethod
    def _mcts_node_info_path(
        node_info_dir, model_name, dataset_name, event_idx, suffix
    ):
        if suffix is not None:
            nodeinfo_filename = (
                Path(node_info_dir)
                / f"{model_name}_{dataset_name}_{event_idx}_mcts_node_info_{suffix}.pt"
            )
        else:
            nodeinfo_filename = (
                Path(node_info_dir)
                / f"{model_name}_{dataset_name}_{event_idx}_mcts_node_info.pt"
            )

        return nodeinfo_filename

    def _save_mcts_recorder(self, event_idx):
        # save records
        recorder_df = pd.DataFrame(self.mcts_state_map.recorder)
        # ROOT_DIR.parent/'benchmarks'/'results'
        record_filename = self._mcts_recorder_path(
            self.results_dir,
            self.model_name,
            self.dataset_name,
            event_idx,
            suffix=self.suffix,
        )
        recorder_df.to_csv(record_filename, index=False)

        print(f"mcts recorder saved at {str(record_filename)}")

    def _save_mcts_nodes_info(self, tree_nodes, event_idx):
        saved_contents = {
            "saved_MCTSInfo_list": self.write_from_MCTSNode_list(tree_nodes),
        }
        path = self._mcts_node_info_path(
            self.mcts_saved_dir,
            self.model_name,
            self.dataset_name,
            event_idx,
            suffix=self.suffix,
        )
        torch.save(saved_contents, path)
        print(f"results saved at {path}")

    def _load_saved_nodes_info(self, event_idx):
        path = self._mcts_node_info_path(
            self.mcts_saved_dir,
            self.model_name,
            self.dataset_name,
            event_idx,
            suffix=self.suffix,
        )
        assert os.path.isfile(path)
        saved_contents = torch.load(path)

        saved_MCTSInfo_list = saved_contents["saved_MCTSInfo_list"]
        tree_nodes = self.read_from_MCTSInfo_list(saved_MCTSInfo_list)
        tree_node_x = find_best_node_result(tree_nodes, self.min_atoms)

        return tree_nodes, tree_node_x

    def _set_candidate_weights(self, event_idx):
        # save candidates' initial weights computed by the pg_explainer_tg
        # from src.method.tg_score import _set_tgat_data
        # from src.method.attn_explainer_tg import AttnExplainerTG

        self.pg_explainer_model.eval()  # mlp
        input_expl = _create_explainer_input(
            self.libcity_base_explainer.target_event,
            self.libcity_base_explainer.events,
            self.device,
            self.libcity_base_explainer,
        )

        edge_weights = self.pg_explainer_model(input_expl)  # compute importance scores
        edge_weights = edge_weights.cpu().detach().numpy().flatten()
        # event_idx_scores = event_idx_scores.cpu().detach().numpy().flatten()

        # added to original model attention scores
        # candidate_weights_dict = {'candidate_events': torch.tensor(
        #     self.candidate_events, dtype=torch.int64, device=self.device), 'edge_weights': edge_weights, }
        # src_idx_l, target_idx_l, cut_time_l = _set_tgat_data(
        #     self.all_events, event_idx)
        # output = self.model.get_prob(src_idx_l, target_idx_l, cut_time_l, logit=True,
        #                              candidate_weights_dict=candidate_weights_dict, num_neighbors=200)
        # e_idx_weight_dict = AttnExplainerTG._agg_attention(
        #     self.model, self.model_name)
        # edge_weights = np.array([e_idx_weight_dict[e_idx]
        #                         for e_idx in candidate_events])
        # # added to original model attention scores
        #
        # if not self.pg_positive:
        #     edge_weights = -1 * edge_weights

        # import ipdb; ipdb.set_trace()

        # event_idx_scores = np.random.random(size=(len(event_idx_scores,))) # ??
        candidate_initial_weights = {
            self.candidate_events[i]: edge_weights[i]
            for i in range(len(self.candidate_events))
        }
        self.candidate_initial_weights = candidate_initial_weights

    def _initialize(self, event_idx):
        super(TGNNExplainer, self)._initialize(
            event_idx,
            self.libcity_base_explainer,
            threshold_num=self.candidate_events_num,
        )
        if self.pg_explainer_model is not None:  # use pg model
            self._set_candidate_weights(event_idx)

    def __call__(
        self,
        node_idxs: Union[int, None] = None,
        event_idxs: Union[int, None] = None,
        return_dict=None,
        device=None,
        results_dir=None,
        results_batch=None,
    ):
        """
        Args:
            node_idxs: the target node index to explain for node prediction tasks
            event_idxs: the target event index to explain for edge prediction tasks
        """

        if device is not None:
            self._to_device(device)

        if isinstance(event_idxs, int):
            event_idxs = [event_idxs]

        for i, event_idx in enumerate(event_idxs):
            #            try:

            self.libcity_base_explainer = TGNNExplainer_LibCity(
                self.model_name, self.dataset_name
            )
            self._initialize(
                event_idx,
            )

            # if self.load_results:
            #     tree_nodes, tree_node_x = self._load_saved_nodes_info(
            #         event_idx)
            self.explain(event_idx=event_idx)

    def _to_device(self, device):
        pass
        # if torch.cuda.is_available():
        #     device = torch.device('cuda', index=device)
        # else:
        #     device = torch.device('cpu')

        # self.device = device
        # self.model.device = device
        # self.model.to(device)

        # if self.model_name == 'tgat':
        #     self.model.node_raw_embed = self.model.node_raw_embed.to(device)
        #     self.model.edge_raw_embed = self.model.edge_raw_embed.to(device)
        #     pass
        # elif self.model_name == 'tgn':
        #     self.model.node_raw_features = self.model.node_raw_features.to(device)
        #     self.model.edge_raw_features = self.model.edge_raw_features.to(device)

        # import ipdb; ipdb.set_trace()
