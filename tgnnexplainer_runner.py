import numpy as np
import torch
import dill as pck
import os
import argparse
import copy
from matplotlib import pyplot as plt
import random
import torch
import pandas as pd

from libcity.data import get_dataset
from libcity.utils import get_model
from libcity.config import ConfigParser

import sys
sys.path.append(os.getcwd() + '/tgnnexplainer_src')

from tgnnexplainer_src.method.tgnnexplainer import TGNNExplainer
from tgnnexplainer_src.method.other_baselines_tg import PGExplainerExt

from libcity_explainer_runner import STX_Search_LibCity

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default='TGCN', help='Mode of operation: generate or visualise')
parser.add_argument('-t', '--target_idx', type=int, default=12, help='Target node index for explanation')
parser.add_argument('-s', '--subgraph_size', type=int, default=50, help='Size of the subgraph for explanation')
parser.add_argument('-d', '--dataset', type=str, default='METR_LA', help='Dataset name')
parser.add_argument('--mode', type=str, default='fidelity', help='Explanation Mode')
parser.add_argument('--train_pg_explainer', action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()

# Min Atoms is the number of events preserved from the candidate events
min_atoms= 50
# The hyperparameter to encourage exploration when calculating utility in MCTS
c_puct = 100


# This class includes all the base functions needed to load models and make predictions from LibCity
# models and datasets using all or some subset of input events.
#libcity_base_explainer = STX_Search_LibCity()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


events = pd.read_csv(f'raw_data/{args.dataset}/ml_{args.dataset}.csv')
events['e_idx'] = events.index
#libcity_base_explainer.all_events_df = events
#libcity_base_explainer.node_id_to_index = pd.unique(events['u'])
edge_feat = pd.read_csv(f'raw_data/{args.dataset}/ml_{args.dataset}_edge_feats.csv')

if args.train_pg_explainer:
     explainer = PGExplainerExt(
                model_name=args.model,
                explainer_name='tgnnexplainer',
                dataset_name=args.dataset,
                all_events=events,
                explanation_level='event',
                device=device,
                results_dir='',
                train_epochs=100,
                explainer_ckpt_dir='',
                reg_coefs=[0.5, 0.1],
                batch_size=16,
                lr=1e-4,
                debug_mode=False,
                exp_size=20,
                edge_feat=edge_feat,
                )

     explainer(event_idxs=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],)

#pg_explainer_model, explainer_ckpt_path = PGExplainerExt.expose_explainer_model(
#                        model_predict_func=libcity_base_explainer.exp_prediction, # load a trained mlp model
#                        model_name=args.model,
#                        explainer_name='pg_explainer_tg', # fixed
#                        dataset_name=args.dataset,
#                        ckpt_dir='',
#                        device=device,
#                        )

explainer = TGNNExplainer(model,
                        model_name=args.model,
                        explainer_name='tgnnexplainer',
                        dataset_name=args.dataset,
                        all_events=events,
                        explanation_level='event',
                        device=device,
                        rollout=500,
                        min_atoms=min_atoms,
                        c_puct=c_puct,
                        pg_explainer_model=pg_explainer_model,
                        )


