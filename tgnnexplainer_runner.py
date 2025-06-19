from tgnnexplainer_src.method.other_baselines_tg import PGExplainerExt
from tgnnexplainer_src.method.tgnnexplainer import TGNNExplainer
import torch
import argparse
import torch
import pandas as pd


import os
import sys
sys.path.append(os.getcwd() + '/tgnnexplainer_src')


# Min Atoms is the number of events preserved from the candidate events
min_atoms = 50
# The hyperparameter to encourage exploration when calculating utility in MCTS
c_puct = 10


# This class includes all the base functions needed to load models and make predictions from LibCity
# models and datasets using all or some subset of input events.
# libcity_base_explainer = STX_Search_LibCity()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


events = pd.read_csv(f'raw_data/{args.dataset}/ml_{args.dataset}.csv')
events['e_idx'] = events.index
# libcity_base_explainer.all_events_df = events
# libcity_base_explainer.node_id_to_index = pd.unique(events['u'])
edge_feat = pd.read_csv(
    f'raw_data/{args.dataset}/ml_{args.dataset}_edge_feats.csv')

if args.train_pg_explainer:
    explainer = PGExplainerExt(
        model_name=args.model,
        explainer_name='tgnnexplainer',
        dataset_name=args.dataset,
        all_events=events,
        explanation_level='event',
        device=device,
        results_dir='saved/models/PGE_models/',
        train_epochs=100,
        explainer_ckpt_dir='saved/models/PGE_models/',
        reg_coefs=[0.5, 0.1],
        batch_size=16,
        lr=1e-4,
        debug_mode=False,
        exp_size=20,
        edge_feat=edge_feat,
    )

    explainer(event_idxs=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],)

    pg_explainer_model, explainer_ckpt_path = PGExplainerExt.expose_explainer_model(
        model_name=args.model,
        explainer_name='tgnnexplainer',  # fixed
        dataset_name=args.dataset,
        ckpt_dir='saved/models/PGE_models/',
        device=device,
    )

    explainer = TGNNExplainer(model_name=args.model,
                              explainer_name='tgnnexplainer',
                              dataset_name=args.dataset,
                              all_events=events,
                              explanation_level='event',
                              device=device,
                              rollout=100,
                              min_atoms=min_atoms,
                              c_puct=c_puct,
                              pg_explainer_model=pg_explainer_model,
                              edge_feat=edge_feat,
                              candidate_events_num=200,
                              )

# explainer(event_idxs=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],)
