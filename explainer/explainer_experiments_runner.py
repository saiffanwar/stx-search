import argparse
import os
import random
import sys
import time
import pickle as pck

import pandas as pd
import torch

from stx_search_explainer import STX_Search_LibCity
from tgnnexplainer_src.method.other_baselines_tg import PGExplainerExt
from tgnnexplainer_src.method.tgnnexplainer import TGNNExplainer

sys.path.append(os.getcwd() + "/tgnnexplainer_src")


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="TGCN", help="Mode of operation: generate or visualise")
parser.add_argument("-t", "--target_idx", type=int, default=None, help="Target node index for explanation")
parser.add_argument("-s", "--exp_size", type=int, default=20, help="Size of the subgraph for explanation")
parser.add_argument("-d", "--dataset", type=str, default="METR_LA", help="Dataset name")
parser.add_argument("--stx_mode", type=str, default="fidelity", help="Explanation Mode")
parser.add_argument("--explainer", type=str, default="stx_search", help="Explainer to use: stx_search, tgnnexplainer or pg_explainer")
parser.add_argument("--num_exps", type=int, default=10, help="Number of events to explain")
args = parser.parse_args()



events = pd.read_csv(f"raw_data/{args.dataset}/ml_{args.dataset}.csv")
events["e_idx"] = events.index

events_to_explain = pck.load(open(f"raw_data/{args.dataset}/events_to_explain.pck", "rb"))

events_to_explain = events_to_explain[:1]

if args.explainer == "stx_search":
    with torch.no_grad():
        print("Initialising Explainer...")
        explainer = STX_Search_LibCity(args.model, args.dataset, all_events=events)
        for event_idx in events_to_explain:
            for exp_size in [args.exp_size]:
                print(
                    f"######################## Explaining event {event_idx} with exp size {exp_size} using STX Search"
                )

                num_iter = 100

                tic = time.time()
                score, exp_events, model_pred, exp_pred = explainer.explain(
                    explaining_event_idx=event_idx,
                    exp_size=exp_size,
                    mode=args.stx_mode,
                    num_iter=num_iter,
                )
                toc = time.time()
elif args.explainer in ["pg_explainer", "tgnnexplainer"]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    edge_feat = pd.read_csv(f"raw_data/{args.dataset}/ml_{args.dataset}_edge_feats.csv")

    explainer = PGExplainerExt(
        model_name=args.model,
        explainer_name="tgnnexplainer",
        dataset_name=args.dataset,
        all_events=events,
        explanation_level="event",
        device=device,
        results_dir="saved/models/PGE_models/",
        train_epochs=100,
        explainer_ckpt_dir="saved/models/PGE_models/",
        reg_coefs=[0.5, 0.1],
        batch_size=16,
        lr=1e-4,
        debug_mode=False,
        exp_size=args.exp_size,
        edge_feat=edge_feat,
        retrain_explainer=False,
    )

    if args.explainer == "pg_explainer":
        explainer(
            event_idxs=events_to_explain,
        )

    elif args.explainer == "tgnnexplainer":
        pg_explainer_model, explainer_ckpt_path = PGExplainerExt.expose_explainer_model(
            model_name=args.model,
            explainer_name="tgnnexplainer",  # fixed
            dataset_name=args.dataset,
            ckpt_dir="saved/models/PGE_models/",
            device=device,
        )

        explainer = TGNNExplainer(
            model_name=args.model,
            explainer_name="tgnnexplainer",
            dataset_name=args.dataset,
            all_events=events,
            explanation_level="event",
            device=device,
            rollout=200,
            min_atoms=args.exp_size,
            c_puct=10,
            pg_explainer_model=pg_explainer_model,
            edge_feat=edge_feat,
            candidate_events_num=20,
        )

        explainer(event_idxs=events_to_explain)
