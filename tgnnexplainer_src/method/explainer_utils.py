import logging
import os
import gc
import argparse
import math
import random
import warnings
import tqdm
import numpy as np
import pandas as pd
from sklearn import preprocessing

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils

from script import dataloader, utility, earlystopping, opt
from model import models

#import nni

class graphNode:
    def __init__(self,node_index, timestamp, speed):
        self.node_index = node_index
        self.timestamp = timestamp
        self.speed = speed

class Explainer():

    def __init__(self, args, device, blocks, target_index):
        self.n_his = args.n_his
        self.n_pred = args.n_pred
        self.args = args
        self.device = device
        self.blocks = blocks
        self.num_nodes = 25
        self.target_index = target_index

    def generate_graph_nodes(self, x):
        x = x.cpu()
        all_nodes = []
        for t in range(self.n_his):
            for n in range(self.num_nodes):
                node = graphNode(n, t, x[t][n])
                all_nodes.append(node)
        return all_nodes

    def create_masked_input(self, subgraph_nodes):
        masked_x = torch.zeros((self.n_his, self.num_nodes))
        def mask_gen(node):
            masked_x[node.timestamp][node.node_index] = self.input[node.timestamp][node.node_index]

        map(mask_gen, subgraph_nodes)

#        for node in subgraph_nodes:
#            masked_x[node.timestamp][node.node_index] = self.input[node.timestamp][node.node_index]
        return masked_x

    def exp_fidelity(self, exp_nodes, mode='fidelity'):

        masked_x = self.create_masked_input(exp_nodes)
        exp_pred = self.model(masked_x.unsqueeze(0).unsqueeze(0))

        exp_pred = self.scaler.inverse_transform(exp_pred.squeeze(0).squeeze(0).cpu().detach().numpy())[0]
        print(exp_pred)

        target_explanation_y = exp_pred[self.target_index]

        exp_absolute_error = abs(self.target_model_y - target_explanation_y)
        max_exp_size = 500
        exp_size_percentage = (100*len(exp_nodes)/max_exp_size)
        exp_percentage_error = 100*(exp_absolute_error/self.target_model_y)

#        gam = k*exp_error/(exp_error + (1-k)*exp_error)
#        gam = 1
#        lam = (exp_error - k*exp_error)/(exp_error + k*exp_error)


        if mode == 'fidelity':
            exp_score = exp_percentage_error
            return exp_score, exp_absolute_error

        elif mode == 'fidelity+size':
            lam = 0.1
            gam = 0.9
            exp_score = gam*exp_percentage_error + lam*exp_size_percentage
            return exp_score, gam*exp_percentage_error, lam*exp_size_percentage, exp_absolute_error

    def initialise_explainer(self):


        n_vertex, self.scaler, train_iter, val_iter, test_iter = data_preparate(self.args, self.device)
        for batch in train_iter:
            x, y = batch
            first_x = x[0].squeeze(0)
            first_y = y[0]
            break
        self.input = first_x
        loss, es, self.model, optimizer, scheduler = prepare_model(self.args, self.blocks, n_vertex, self.device)
        self.candidate_events = self.generate_graph_nodes(self.input)

        self.model_pred = self.model(self.input.unsqueeze(0).unsqueeze(0))
        self.model_pred = self.scaler.inverse_transform(self.model_pred.squeeze(0).squeeze(0).cpu().detach().numpy())[0]
        self.target_model_y = self.model_pred[self.target_index]
#        exp_nodes = random.sample(self.candidate_events, 10)
#
        exp_score, exp_absolute_error = self.exp_fidelity(self.candidate_events)
#        print(exp_score)
#        print(exp_absolute_error)


def run_explainer(target_index):
    args, device, blocks = get_parameters()
    explainer = Explainer(args, device, blocks, target_index)
    explainer.initialise_explainer()

    return explainer

