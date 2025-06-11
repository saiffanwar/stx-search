from typing import Union
from typing import List
import numpy as np
from pandas import DataFrame

from tgnnexplainer_src.evaluation.metrics_tg_utils import fidelity_inv_tg


def _set_tgat_data(all_events: DataFrame, target_event_idx: Union[int, List]):
    """ supporter for tgat """
    if isinstance(target_event_idx, (int, np.int64)):
        target_u = all_events.iloc[target_event_idx-1, 0]
        target_i = all_events.iloc[target_event_idx-1, 1]
        target_t = all_events.iloc[target_event_idx-1, 2]

        src_idx_l = np.array([target_u, ])
        target_idx_l = np.array([target_i, ])
        cut_time_l = np.array([target_t, ])
    elif isinstance(target_event_idx, list):
        # targets = all_events[all_events.e_idx.isin(target_event_idx)]
        targets = all_events.iloc[np.array(target_event_idx)-1] # faster?

        target_u = targets.u.values
        target_i = targets.i.values
        target_t = targets.ts.values

        src_idx_l = target_u
        target_idx_l = target_i
        cut_time_l = target_t
    else:
        raise ValueError

    input_data = [src_idx_l, target_idx_l, cut_time_l]
    return input_data


class TGNNRewardWraper(object):
    def __init__(self, model_name, all_events, explanation_level):
        """
        """
#        self.model = model
        self.model_name = model_name
        self.all_events = all_events
        self.n_users = all_events.iloc[:, 0].max() + 1
        self.explanation_level = explanation_level
        self.gamma = 0.05
        # if self.model_name == 'tgn':
            # self.tgn_memory_backup = self.model.memory.backup_memory()

    # def error(self, ori_pred, ptb_pred):

    #     pass


    def _get_model_prob(self, target_event_idx, exp_event_node_ids, exp_event_rltv_ts, num_neighbors=None):
        if self.model_name in ['tgat', 'tgn']:
            input_data = _set_tgat_data(self.all_events, target_event_idx)
            # seen_events_idxs = _set_tgat_events_idxs(seen_events_idxs) # NOTE: not important now
#            print('seen_events_idxs: ', seen_events_idxs)
            score = self.model.get_prob(*input_data, edge_idx_preserve_list=seen_events_idxs, logit=True, num_neighbors=num_neighbors)
            # import ipdb; ipdb.set_trace()
        else:
            raise NotImplementedError

        return score.item()



    def compute_original_score(self, events_idxs, events_rltv_ts, target_event_idx, num_neighbors=200):
        """
        events_idxs: could be seen by model
        """
        self.original_scores = self._get_model_prob(target_event_idx, events_idxs, events_rltv_ts)
        self.orininal_size = len(events_idxs)
        return self.original_scores

    def __call__(self, events_idxs, target_event_idx, final_result=False):
        """
        events_idxs the all the events' indices could be seen by the gnn model. from 1
        target_event_idx is the target edge that we want to compute a reward by the temporal GNN model. from 1
        """

        if self.model_name in ['tgat', 'tgn']:
            scores = self._get_model_prob(target_event_idx, events_idxs, num_neighbors=200)
            # import ipdb; ipdb.set_trace()
            reward = self._compute_reward(scores)
            if final_result:
                return reward, scores
            else:
                return reward
        else:
            raise NotImplementedError

        pred = self._get_model_prob(target_event_idx, events_idxs)


    def _compute_gnn_score(self, events_idxs, target_event_idx):
        """
        events_idxs the all the events' indices could be seen by the gnn model. idxs in the all_events space, not in the tgat space.
        target_event_idx is the target edge that we want to compute a gnn score by the temporal GNN model.
        """
        return self._get_model_prob(target_event_idx, events_idxs)


    def _compute_reward(self, exp_pred, remove_size=None):
        """
        Reward should be the larger the better.
        """
        fid_inv = abs(self.original_scores - exp_pred)
        return fid_inv

        # if self.original_scores >= 0:
        #     t1 = scores_petb - self.original_scores
        # else:
        #     t1 = self.original_scores - scores_petb

        # t2 = remove_size
        # # r = -1*t1 + -self.gamma * t2
        # # r = -t1
        # r = t1
        # return r


