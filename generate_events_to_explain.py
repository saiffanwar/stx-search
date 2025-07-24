import random
import pandas as pd
import argparse
import os
import sys
import pickle as pck

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default="METR_LA", help="Dataset name")
parser.add_argument("-n", "--num_exps", type=int, default=20, help="Number of events to explain")
args = parser.parse_args()


def generate_events_to_explain(num_events=100, num_features=10):
    events = pd.read_csv(f"raw_data/{args.dataset}/ml_{args.dataset}.csv")
    events["e_idx"] = events.index
    events_to_explain = random.sample(list(events["e_idx"]), num_events)
    random.seed(42)
    with open(f"raw_data/{args.dataset}/events_to_explain.pck", "wb") as f:
        pck.dump(events_to_explain, f)
    
if __name__ == "__main__":
    generate_events_to_explain(num_events=args.num_exps)
    print(f"Generated {args.num_exps} events to explain for dataset {args.dataset}.")

    

