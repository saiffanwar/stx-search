"""Unified interface to all dynamic graph model experiments"""
import time
import pickle as pck
import math
import random
import sys
from tqdm import tqdm
import argparse
import os.path as osp
import h5py
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from src.method.temp_me_utils import RandEdgeSampler, load_subgraph, load_subgraph_margin, get_item, get_item_edge, NeighborFinder
import copy
from src.method.tempme_explainer import *
#from GraphM import GraphMixer
#from TGN.tgn import TGN


def norm_imp(imp):
    imp[imp < 0] = 0
    imp += 1e-16
    return imp / imp.sum()


### Load data and train val test split
def load_data(mode, args=None):
    g_df = pd.read_csv(osp.join(args.data_dir, 'ml_{}.csv'.format(args.data)))
    val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))
    src_l = g_df.u.values
    dst_l = g_df.i.values
    e_idx_l = g_df.index.values
    label_l = g_df.label.values
    ts_l = g_df.ts.values
    max_src_index = src_l.max()
    max_idx = max(src_l.max(), dst_l.max())
    random.seed(2023)
    total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))
    num_total_unique_nodes = len(total_node_set)
    mask_node_set = set(random.sample(sorted(set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time]))),
                                      int(0.1 * num_total_unique_nodes)))
    mask_src_flag = g_df.u.map(lambda x: x in mask_node_set).values
    mask_dst_flag = g_df.i.map(lambda x: x in mask_node_set).values
    none_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag)
    valid_train_flag = (ts_l <= val_time) * (none_node_flag > 0)
    train_src_l = src_l[valid_train_flag]
    train_dst_l = dst_l[valid_train_flag]
    train_ts_l = ts_l[valid_train_flag]
    train_e_idx_l = e_idx_l[valid_train_flag]
    train_label_l = label_l[valid_train_flag]
    valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
    valid_test_flag = ts_l > test_time
    val_src_l = src_l[valid_val_flag]
    val_dst_l = dst_l[valid_val_flag]
    test_src_l = src_l[valid_test_flag]
    test_dst_l = dst_l[valid_test_flag]
    test_ts_l = ts_l[valid_test_flag]
    test_e_idx_l = e_idx_l[valid_test_flag]
    test_label_l = label_l[valid_test_flag]
    adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
        adj_list[src].append((dst, eidx, ts))
        adj_list[dst].append((src, eidx, ts))
    train_ngh_finder = NeighborFinder(adj_list)
    full_adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
        full_adj_list[src].append((dst, eidx, ts))
        full_adj_list[dst].append((src, eidx, ts))
    full_ngh_finder = NeighborFinder(full_adj_list)
    train_rand_sampler = RandEdgeSampler((train_src_l,), (train_dst_l,))
    test_rand_sampler = RandEdgeSampler((train_src_l, val_src_l, test_src_l), (train_dst_l, val_dst_l, test_dst_l))
    if mode == "test":
        return test_rand_sampler, test_src_l, test_dst_l, test_ts_l, test_label_l, test_e_idx_l, full_ngh_finder
    else:
        return train_rand_sampler, train_src_l, train_dst_l, train_ts_l, train_label_l, train_e_idx_l, train_ngh_finder


def threshold_test(args, explanation, base_model, src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
                   pos_out_ori, neg_out_ori, y_ori, subgraph_src, subgraph_tgt, subgraph_bgd):
    '''
    calculate the AUC over ratios in [0~0.3]
    '''
    AUC_aps, AUC_acc, AUC_auc, AUC_fid_logit, AUC_fid_prob = [], [], [], [], []
    for ratio in args.ratios:
        if args.base_type == "tgn":
            num_edge = args.n_degree + args.n_degree * args.n_degree
            topk = min(max(math.ceil(ratio * num_edge), 1), num_edge)
            edge_imp_src = torch.cat([explanation[0][:args.bs], explanation[1][:args.bs]],
                                     dim=1)  # first: (batch, num_neighbors), second: (batch, num_neighbors * num_neighbors)
            edge_imp_tgt = torch.cat([explanation[0][args.bs:2 * args.bs], explanation[1][args.bs:2 * args.bs]],
                                     dim=1)
            edge_imp_bgd = torch.cat([explanation[0][2 * args.bs:], explanation[1][2 * args.bs:]], dim=1)
            selected_src = torch.topk(edge_imp_src, k=num_edge - topk, dim=-1, largest=False).indices
            selected_tgt = torch.topk(edge_imp_tgt, k=num_edge - topk, dim=-1, largest=False).indices
            selected_bgd = torch.topk(edge_imp_bgd, k=num_edge - topk, dim=-1, largest=False).indices

            node_records_src, eidx_records_src, t_records_src = subgraph_src
            node_records_src_cat = np.concatenate(node_records_src, axis=-1)  # [B, NUM + NUM**2]
            np.put_along_axis(node_records_src_cat, selected_src.cpu().numpy(), 0, axis=-1)
            node_records_src = np.split(node_records_src_cat, [args.n_degree], axis=1)
            subgraph_src_sub = node_records_src, eidx_records_src, t_records_src

            node_records_tgt, eidx_records_tgt, t_records_tgt = subgraph_tgt
            node_records_tgt_cat = np.concatenate(node_records_tgt, axis=-1)  # [B, NUM + NUM**2]
            np.put_along_axis(node_records_tgt_cat, selected_tgt.cpu().numpy(), 0, axis=-1)
            node_records_tgt = np.split(node_records_tgt_cat, [args.n_degree], axis=1)
            subgraph_tgt_sub = node_records_tgt, eidx_records_tgt, t_records_tgt

            node_records_bgd, eidx_records_bgd, t_records_bgd = subgraph_bgd
            node_records_bgd_cat = np.concatenate(node_records_bgd, axis=-1)  # [B, NUM + NUM**2]
            np.put_along_axis(node_records_bgd_cat, selected_bgd.cpu().numpy(), 0, axis=-1)
            node_records_bgd = np.split(node_records_bgd_cat, [args.n_degree], axis=1)
            subgraph_bgd_sub = node_records_bgd, eidx_records_bgd, t_records_bgd

        elif args.base_type == "graphmixer":
            num_edge = args.n_degree
            topk = min(max(math.ceil(ratio * num_edge), 1), num_edge)
            edge_imp_src, edge_imp_tgt, edge_imp_bgd = explanation[0][:args.bs], \
                                                       explanation[0][args.bs:2 * args.bs], \
                                                       explanation[0][2 * args.bs:]
            selected_src = torch.topk(edge_imp_src, k=num_edge - topk, dim=-1, largest=False).indices
            selected_tgt = torch.topk(edge_imp_tgt, k=num_edge - topk, dim=-1, largest=False).indices
            selected_bgd = torch.topk(edge_imp_bgd, k=num_edge - topk, dim=-1, largest=False).indices
            node_records_src, eidx_records_src, t_records_src = subgraph_src
            node_src_0 = node_records_src[0].copy()
            np.put_along_axis(node_src_0, selected_src.cpu().numpy(), 0, axis=-1)
            node_records_src_sub = [node_src_0, node_records_src[1]]
            subgraph_src_sub = node_records_src_sub, eidx_records_src, t_records_src

            node_records_tgt, eidx_records_tgt, t_records_tgt = subgraph_tgt
            node_tgt_0 = node_records_tgt[0].copy()
            np.put_along_axis(node_tgt_0, selected_tgt.cpu().numpy(), 0, axis=-1)
            node_records_tgt_sub = [node_tgt_0, node_records_tgt[1]]
            subgraph_tgt_sub = node_records_tgt_sub, eidx_records_tgt, t_records_tgt

            node_records_bgd, eidx_records_bgd, t_records_bgd = subgraph_bgd
            node_bgd_0 = node_records_bgd[0].copy()
            np.put_along_axis(node_bgd_0, selected_bgd.cpu().numpy(), 0, axis=-1)
            node_records_bgd_sub = [node_bgd_0, node_records_bgd[1]]
            subgraph_bgd_sub = node_records_bgd_sub, eidx_records_bgd, t_records_bgd
        elif args.base_type == "tgat":
            num_edge = args.n_degree + args.n_degree * args.n_degree
            topk = min(max(math.ceil(ratio * num_edge), 1), num_edge)
            edge_imp_src = torch.cat([explanation[0], explanation[1]], dim=1)  # first: (batch, num_neighbors), second: [batch, num_neighbors * num_neighbors]
            edge_imp_tgt = torch.cat([explanation[2], explanation[3]], dim=1)
            edge_imp_bgd = torch.cat([explanation[4], explanation[5]], dim=1)
            selected_src = torch.topk(edge_imp_src, k=num_edge - topk, dim=-1, largest=False).indices
            selected_tgt = torch.topk(edge_imp_tgt, k=num_edge - topk, dim=-1, largest=False).indices
            selected_bgd = torch.topk(edge_imp_bgd, k=num_edge - topk, dim=-1, largest=False).indices
            node_records_src, eidx_records_src, t_records_src = subgraph_src
            node_records_src_cat = np.concatenate(node_records_src, axis=-1)  # [B, NUM + NUM**2]
            np.put_along_axis(node_records_src_cat, selected_src.cpu().numpy(), 0, axis=-1)
            node_records_src = np.split(node_records_src_cat, [args.n_degree], axis=1)
            subgraph_src_sub = node_records_src, eidx_records_src, t_records_src

            node_records_tgt, eidx_records_tgt, t_records_tgt = subgraph_tgt
            node_records_tgt_cat = np.concatenate(node_records_tgt, axis=-1)  # [B, NUM + NUM**2]
            np.put_along_axis(node_records_tgt_cat, selected_tgt.cpu().numpy(), 0, axis=-1)
            node_records_tgt = np.split(node_records_tgt_cat, [args.n_degree], axis=1)
            subgraph_tgt_sub = node_records_tgt, eidx_records_tgt, t_records_tgt

            node_records_bgd, eidx_records_bgd, t_records_bgd = subgraph_bgd
            node_records_bgd_cat = np.concatenate(node_records_bgd, axis=-1)  # [B, NUM + NUM**2]
            np.put_along_axis(node_records_bgd_cat, selected_bgd.cpu().numpy(), 0, axis=-1)
            node_records_bgd = np.split(node_records_bgd_cat, [args.n_degree], axis=1)
            subgraph_bgd_sub = node_records_bgd, eidx_records_bgd, t_records_bgd

        else:
            raise ValueError(f"Wrong value for base_type {args.base_type}!")

        with torch.no_grad():
            if args.base_type == "tgat":
                pos_logit, neg_logit = base_model.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
                                                 subgraph_src_sub, subgraph_tgt_sub, subgraph_bgd_sub, test=True,
                                                 if_explain=False)
            else:
                pos_logit, neg_logit = base_model.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
                                                        subgraph_src_sub, subgraph_tgt_sub, subgraph_bgd_sub)
            y_pred = torch.cat([pos_logit, neg_logit], dim=0).sigmoid()
            pred_label = torch.where(y_pred > 0.5, 1., 0.).view(y_pred.size(0), 1)
            fid_prob_batch = torch.cat(
                [pos_logit.sigmoid() - pos_out_ori.sigmoid(), neg_out_ori.sigmoid() - neg_logit.sigmoid()],
                dim=0)
            fid_prob = torch.mean(fid_prob_batch, dim=0)
            fid_logit_batch = torch.cat([pos_logit - pos_out_ori, neg_out_ori - neg_logit], dim=0)
            fid_logit = torch.mean(fid_logit_batch, dim=0)
            AUC_fid_prob.append(fid_prob.item())
            AUC_fid_logit.append(fid_logit.item())
            AUC_aps.append(average_precision_score(y_ori.cpu(), y_pred.cpu()))
            AUC_auc.append(roc_auc_score(y_ori.cpu(), y_pred.cpu()))
            AUC_acc.append((pred_label.cpu() == y_ori.cpu()).float().mean())
    aps_AUC = np.mean(AUC_aps)
    auc_AUC = np.mean(AUC_auc)
    acc_AUC = np.mean(AUC_acc)
    fid_prob_AUC = np.mean(AUC_fid_prob)
    fid_logit_AUC = np.mean(AUC_fid_logit)
    return aps_AUC, auc_AUC, acc_AUC, fid_prob_AUC, fid_logit_AUC



def eval_one_epoch_tgat(args, base_model, explainer, full_ngh_finder, sampler, src, dst, ts, val_e_idx_l, epoch, best_accuracy, test_pack, test_edge):
    test_aps = []
    test_auc = []
    test_acc = []
    test_fid_prob = []
    test_fid_logit = []
    test_loss = []
    test_pred_loss = []
    test_kl_loss = []
    ratio_AUC_aps, ratio_AUC_auc, ratio_AUC_acc, ratio_AUC_prob, ratio_AUC_logit  = [],[],[],[],[]
    base_model = base_model.eval()
    num_test_instance = len(src) - 1
    num_test_batch = math.ceil(num_test_instance / args.test_bs)-1
    idx_list = np.arange(num_test_instance)
    criterion = torch.nn.BCEWithLogitsLoss()
    base_model.ngh_finder = full_ngh_finder
    for k in tqdm(range(num_test_batch)):
        s_idx = k * args.test_bs
        e_idx = min(num_test_instance - 1, s_idx + args.test_bs)
        if s_idx == e_idx:
            continue
        batch_idx = idx_list[s_idx:e_idx]
        src_l_cut = src[batch_idx]
        dst_l_cut = dst[batch_idx]
        ts_l_cut = ts[batch_idx]
        e_l_cut = val_e_idx_l[batch_idx] if (val_e_idx_l is not None) else None
        subgraph_src, subgraph_tgt, subgraph_bgd, walks_src, walks_tgt, walks_bgd, dst_l_fake = get_item(test_pack, batch_idx)
        edge_idfeature = get_item_edge(test_edge, batch_idx)
        src_edge, tgt_edge, bgd_edge = edge_idfeature
        with torch.no_grad():
            pos_out_ori, neg_out_ori = base_model.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
                                                     subgraph_src, subgraph_tgt, subgraph_bgd,
                                                     test=True, if_explain=False)  # [B, 1]
            y_pred = torch.cat([pos_out_ori, neg_out_ori], dim=0).sigmoid()  # [B*2, 1]
            y_ori = torch.where(y_pred > 0.5, 1., 0.).view(y_pred.size(0), 1)  # [2 * B, 1]

        explainer.eval()
        graphlet_imp_src = explainer(walks_src, ts_l_cut, src_edge)
        edge_imp_src = explainer.retrieve_edge_imp(subgraph_src, graphlet_imp_src, walks_src, training=args.if_bern)
        graphlet_imp_tgt = explainer(walks_tgt, ts_l_cut, tgt_edge)
        edge_imp_tgt = explainer.retrieve_edge_imp(subgraph_tgt, graphlet_imp_tgt, walks_tgt, training=args.if_bern)
        graphlet_imp_bgd = explainer(walks_bgd, ts_l_cut, bgd_edge)
        edge_imp_bgd = explainer.retrieve_edge_imp(subgraph_bgd, graphlet_imp_bgd, walks_bgd, training=args.if_bern)
        explain_weight = [[edge_imp_src, edge_imp_tgt], [edge_imp_src, edge_imp_bgd]]
        pos_logit, neg_logit = base_model.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
                                             subgraph_src, subgraph_tgt, subgraph_bgd, test=True,
                                             if_explain=True,
                                             exp_weights=explain_weight)
        pred = torch.cat([pos_logit, neg_logit], dim=0).to(args.device)
        pred_loss = criterion(pred, y_ori)
        kl_loss = explainer.kl_loss(graphlet_imp_src, walks_src, target=args.prior_p) + \
                    explainer.kl_loss(graphlet_imp_tgt, walks_tgt, target=args.prior_p) + \
                    explainer.kl_loss(graphlet_imp_bgd, walks_bgd, target=args.prior_p)
        loss = pred_loss + args.beta * kl_loss
        with torch.no_grad():
            y_pred = torch.cat([pos_logit, neg_logit], dim=0).sigmoid()
            pred_label = torch.where(y_pred > 0.5, 1., 0.).view(y_pred.size(0), 1)
            fid_prob_batch = torch.cat([pos_logit.sigmoid() - pos_out_ori.sigmoid(), neg_out_ori.sigmoid() - neg_logit.sigmoid()], dim=0)
            fid_prob = torch.mean(fid_prob_batch, dim=0)
            fid_logit_batch = torch.cat([pos_logit - pos_out_ori, neg_out_ori - neg_logit], dim=0)
            fid_logit = torch.mean(fid_logit_batch, dim=0)
            test_fid_prob.append(fid_prob.item())
            test_fid_logit.append(fid_logit.item())
            test_aps.append(average_precision_score(y_ori.cpu(), y_pred.cpu()))
            test_auc.append(roc_auc_score(y_ori.cpu(), y_pred.cpu()))
            test_acc.append((pred_label.cpu() == y_ori.cpu()).float().mean())
            test_loss.append(loss.item())
            test_pred_loss.append(pred_loss.item())
            test_kl_loss.append(kl_loss.item())
            if args.test_threshold:
                node_records, eidx_records, t_records = subgraph_src
                for i in range(len(node_records)):
                    batch_node_idx = torch.from_numpy(node_records[i]).long().to(args.device)
                    mask = batch_node_idx == 0
                    edge_imp_src[i] = edge_imp_src[i].masked_fill(mask, -1e10)
                node_records, eidx_records, t_records = subgraph_tgt
                for i in range(len(node_records)):
                    batch_node_idx = torch.from_numpy(node_records[i]).long().to(args.device)
                    mask = batch_node_idx == 0
                    edge_imp_tgt[i] = edge_imp_tgt[i].masked_fill(mask, -1e10)
                node_records, eidx_records, t_records = subgraph_bgd
                for i in range(len(node_records)):
                    batch_node_idx = torch.from_numpy(node_records[i]).long().to(args.device)
                    mask = batch_node_idx == 0
                    edge_imp_bgd[i] = edge_imp_bgd[i].masked_fill(mask, -1e10)
                edge_imps = edge_imp_src + edge_imp_tgt + edge_imp_bgd
                aps_AUC, auc_AUC, acc_AUC, fid_prob_AUC, fid_logit_AUC = threshold_test(args, edge_imps, base_model, src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
                                     pos_out_ori, neg_out_ori, y_ori, subgraph_src, subgraph_tgt, subgraph_bgd)
                ratio_AUC_aps.append(aps_AUC)
                ratio_AUC_auc.append(auc_AUC)
                ratio_AUC_acc.append(acc_AUC)
                ratio_AUC_prob.append(fid_prob_AUC)
                ratio_AUC_logit.append(fid_logit_AUC)
    aps_ratios_AUC = np.mean(ratio_AUC_aps) if len(ratio_AUC_aps) != 0 else 0
    auc_ratios_AUC = np.mean(ratio_AUC_auc) if len(ratio_AUC_auc) != 0 else 0
    acc_ratios_AUC = np.mean(ratio_AUC_acc) if len(ratio_AUC_acc) != 0 else 0
    prob_ratios_AUC = np.mean(ratio_AUC_prob) if len(ratio_AUC_prob) != 0 else 0
    logit_ratios_AUC = np.mean(ratio_AUC_logit) if len(ratio_AUC_logit) != 0 else 0
    aps_epoch = np.mean(test_aps)
    auc_epoch = np.mean(test_auc)
    acc_epoch = np.mean(test_acc)
    fid_prob_epoch = np.mean(test_fid_prob)
    fid_logit_epoch = np.mean(test_fid_logit)
    loss_epoch = np.mean(test_loss)
    pred_loss_epoch = np.mean(test_pred_loss)
    kl_loss_epoch = np.mean(test_kl_loss)

    print((f'Testing Epoch: {epoch} | '
           f'Testing loss: {loss_epoch} | '
           f'Testing Aps: {aps_epoch} | '
           f'Testing Auc: {auc_epoch} | '
           f'Testing Acc: {acc_epoch} | '
           f'Testing Fidelity Prob: {fid_prob_epoch} | '
           f'Testing Fidelity Logit: {fid_logit_epoch} | '
           f'Ratio APS: {aps_ratios_AUC} | '
           f'Ratio AUC: {auc_ratios_AUC} | '
           f'Ratio ACC: {acc_ratios_AUC} | '
           f'Ratio Prob: {prob_ratios_AUC} | '
           f'Ratio Logit: {logit_ratios_AUC} | '))

    if aps_ratios_AUC > best_accuracy:
        if args.save_model:
            model_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'params', 'explainer/tgat/')
            if not osp.exists(model_path):
                os.makedirs(model_path)
            save_path = f"{args.data}.pt"
            torch.save(explainer, osp.join(model_path, save_path))
            print(f"Save model to {osp.join(model_path, save_path)}")
        return aps_ratios_AUC
    else:
        return best_accuracy



def eval_one_epoch(args, base_model, explainer, full_ngh_finder, src, dst, ts, val_e_idx_l, epoch, best_accuracy,
                   test_pack, test_edge):
    test_aps = []
    test_auc = []
    test_acc = []
    test_fid_prob = []
    test_fid_logit = []
    test_loss = []
    test_pred_loss = []
    test_kl_loss = []
    ratio_AUC_aps, ratio_AUC_auc, ratio_AUC_acc, ratio_AUC_prob, ratio_AUC_logit = [], [], [], [], []
    base_model = base_model.eval()
    num_test_instance = len(src) - 1
    num_test_batch = math.ceil(num_test_instance / args.test_bs) - 1
    idx_list = np.arange(num_test_instance)
    criterion = torch.nn.BCEWithLogitsLoss()
    base_model.set_neighbor_sampler(full_ngh_finder)
    num_test_batch = 100
    for k in tqdm(range(num_test_batch)):
        s_idx = k * args.test_bs
        e_idx = min(num_test_instance - 1, s_idx + args.test_bs)
        if s_idx == e_idx:
            continue
        batch_idx = idx_list[s_idx:e_idx]
        src_l_cut = src[batch_idx]
        dst_l_cut = dst[batch_idx]
        ts_l_cut = ts[batch_idx]
        e_l_cut = val_e_idx_l[batch_idx] if (val_e_idx_l is not None) else None
        subgraph_src, subgraph_tgt, subgraph_bgd, walks_src, walks_tgt, walks_bgd, dst_l_fake = get_item(test_pack,
                                                                                                         batch_idx)
        src_edge, tgt_edge, bgd_edge  = get_item_edge(test_edge, batch_idx)
        with torch.no_grad():
            subgraph_src = base_model.grab_subgraph(src_l_cut, ts_l_cut)
            subgraph_tgt = base_model.grab_subgraph(dst_l_cut, ts_l_cut)
            subgraph_bgd = base_model.grab_subgraph(dst_l_fake, ts_l_cut)

            pos_out_ori, neg_out_ori = base_model.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
                                                           subgraph_src, subgraph_tgt, subgraph_bgd)  # [B, 1]
            y_pred = torch.cat([pos_out_ori, neg_out_ori], dim=0).sigmoid()  # [B*2, 1]
            y_ori = torch.where(y_pred > 0.5, 1., 0.).view(y_pred.size(0), 1)  # [2 * B, 1]

        explainer.eval()
        graphlet_imp_src = explainer(walks_src, ts_l_cut, src_edge)
        graphlet_imp_tgt = explainer(walks_tgt, ts_l_cut, tgt_edge)
        graphlet_imp_bgd = explainer(walks_bgd, ts_l_cut, bgd_edge)
        explanation = explainer.retrieve_explanation(subgraph_src, graphlet_imp_src, walks_src,
                                                     subgraph_tgt, graphlet_imp_tgt, walks_tgt,
                                                     subgraph_bgd, graphlet_imp_bgd, walks_bgd,
                                                     training=args.if_bern)
        pos_logit, neg_logit = base_model.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
                                                   subgraph_src, subgraph_tgt, subgraph_bgd,
                                                   explain_weights=explanation)
        pred = torch.cat([pos_logit, neg_logit], dim=0).to(args.device)
        pred_loss = criterion(pred, y_ori)
        kl_loss = explainer.kl_loss(graphlet_imp_src, walks_src, target=args.prior_p) + \
                    explainer.kl_loss(graphlet_imp_tgt, walks_tgt, target=args.prior_p) + \
                    explainer.kl_loss(graphlet_imp_bgd, walks_bgd, target=args.prior_p)
        loss = pred_loss + args.beta * kl_loss
        with torch.no_grad():
            y_pred = torch.cat([pos_logit, neg_logit], dim=0).sigmoid()
            pred_label = torch.where(y_pred > 0.5, 1., 0.).view(y_pred.size(0), 1)
            fid_prob_batch = torch.cat(
                [pos_logit.sigmoid() - pos_out_ori.sigmoid(), neg_out_ori.sigmoid() - neg_logit.sigmoid()], dim=0)
            fid_prob = torch.mean(fid_prob_batch, dim=0)
            fid_logit_batch = torch.cat([pos_logit - pos_out_ori, neg_out_ori - neg_logit], dim=0)
            fid_logit = torch.mean(fid_logit_batch, dim=0)
            test_fid_prob.append(fid_prob.item())
            test_fid_logit.append(fid_logit.item())
            test_aps.append(average_precision_score(y_ori.cpu(), y_pred.cpu()))
            test_auc.append(roc_auc_score(y_ori.cpu(), y_pred.cpu()))
            test_acc.append((pred_label.cpu() == y_ori.cpu()).float().mean())
            test_loss.append(loss.item())
            test_pred_loss.append(pred_loss.item())
            test_kl_loss.append(kl_loss.item())
            if args.test_threshold:
                explanation = explainer.retrieve_explanation(subgraph_src, graphlet_imp_src, walks_src,
                                                             subgraph_tgt, graphlet_imp_tgt, walks_tgt,
                                                             subgraph_bgd, graphlet_imp_bgd, walks_bgd,
                                                             training=False)
                aps_AUC, auc_AUC, acc_AUC, fid_prob_AUC, fid_logit_AUC = threshold_test(args, explanation, base_model,
                                                                                        src_l_cut, dst_l_cut,
                                                                                        dst_l_fake, ts_l_cut, e_l_cut,
                                                                                        pos_out_ori, neg_out_ori, y_ori,
                                                                                        subgraph_src, subgraph_tgt,
                                                                                        subgraph_bgd)
                ratio_AUC_aps.append(aps_AUC)
                ratio_AUC_auc.append(auc_AUC)
                ratio_AUC_acc.append(acc_AUC)
                ratio_AUC_prob.append(fid_prob_AUC)
                ratio_AUC_logit.append(fid_logit_AUC)
    aps_ratios_AUC = np.mean(ratio_AUC_aps) if len(ratio_AUC_aps) != 0 else 0
    auc_ratios_AUC = np.mean(ratio_AUC_auc) if len(ratio_AUC_auc) != 0 else 0
    acc_ratios_AUC = np.mean(ratio_AUC_acc) if len(ratio_AUC_acc) != 0 else 0
    prob_ratios_AUC = np.mean(ratio_AUC_prob) if len(ratio_AUC_prob) != 0 else 0
    logit_ratios_AUC = np.mean(ratio_AUC_logit) if len(ratio_AUC_logit) != 0 else 0
    aps_epoch = np.mean(test_aps)
    auc_epoch = np.mean(test_auc)
    acc_epoch = np.mean(test_acc)
    fid_prob_epoch = np.mean(test_fid_prob)
    fid_logit_epoch = np.mean(test_fid_logit)
    loss_epoch = np.mean(test_loss)
    pred_loss_epoch = np.mean(test_pred_loss)
    kl_loss_epoch = np.mean(test_kl_loss)
    print((f'Testing Epoch: {epoch} | '
           f'Testing loss: {loss_epoch} | '
           f'Testing Aps: {aps_epoch} | '
           f'Testing Auc: {auc_epoch} | '
           f'Testing Acc: {acc_epoch} | '
           f'Testing Fidelity Prob: {fid_prob_epoch} | '
           f'Testing Fidelity Logit: {fid_logit_epoch} | '
           f'Ratio APS: {aps_ratios_AUC} | '
           f'Ratio AUC: {auc_ratios_AUC} | '
           f'Ratio ACC: {acc_ratios_AUC} | '
           f'Ratio Prob: {prob_ratios_AUC} | '
           f'Ratio Logit: {logit_ratios_AUC} | '))

    if aps_ratios_AUC > best_accuracy:
        if args.save_model:
            model_path = osp.join(osp.dirname(osp.realpath(__file__)), 'params', f'explainer/{args.base_type}/')
            if not osp.exists(model_path):
                os.makedirs(model_path)
            save_path = f"{args.data}.pt"
            torch.save(explainer, osp.join(model_path, save_path))
            print(f"Save model to {osp.join(model_path, save_path)}")
        return aps_ratios_AUC
    else:
        if args.save_model:
            model_path = osp.join(osp.dirname(osp.realpath(__file__)), 'params', f'explainer/{args.base_type}/')
            if not osp.exists(model_path):
                os.makedirs(model_path)
            save_path = f"{args.data}.pt"
            torch.save(explainer, osp.join(model_path, save_path))
            print(f"Save model to {osp.join(model_path, save_path)}")
        return best_accuracy

def train(args, base_model, train_pack, test_pack, train_edge, test_edge):
#    if args.base_type == "tgat":
#        Explainer = TempME(base_model, base_model_type=args.base_type, data=args.data, out_dim=args.out_dim, hid_dim=args.hid_dim,
#                                temp=args.temp, if_cat_feature=True,
#                                dropout_p=args.drop_out, device=args.device, data_dir=args.data_dir)
##        Explainer = TempME_TGAT(base_model, data=args.data, out_dim=args.out_dim, hid_dim=args.hid_dim, temp=args.temp,
##                                dropout_p=args.drop_out, device=args.device, data_dir=args.data_dir)
#    else:
    Explainer = TempME(base_model, base_model_type=args.base_type, data=args.data, out_dim=args.out_dim, hid_dim=args.hid_dim,
                                temp=args.temp, if_cat_feature=True,
                                dropout_p=args.drop_out, device=args.device, data_dir=args.data_dir)
    Explainer = Explainer.to(args.device)
    optimizer = torch.optim.Adam(Explainer.parameters(),
                                 lr=args.lr,
                                 betas=(0.9, 0.999), eps=1e-8,
                                 weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
#    criterion = torch.nn.L1Loss()
    rand_sampler, src_l, dst_l, ts_l, label_l, e_idx_l, ngh_finder = load_data(mode="training", args=args)
    test_rand_sampler, test_src_l, test_dst_l, test_ts_l, test_label_l, test_e_idx_l, full_ngh_finder = load_data(
        mode="test", args=args)
#    num_instance = len(src_l) - 1
    num_instance = len(train_pack[0][0][0])
    num_batch = math.ceil(num_instance / args.bs)
    best_acc = 0
    print('num of training instances: {}'.format(num_instance))
    print('num of testing instances: {}'.format(len(test_src_l) - 1))
    print('num of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)
#    np.random.shuffle(idx_list)
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(args.n_epoch):
        base_model.set_neighbor_sampler(ngh_finder)
        train_aps = []
        train_auc = []
        train_acc = []
        train_fid_prob = []
        train_fid_logit = []
        train_loss = []
        train_pred_loss = []
        train_kl_loss = []
#        np.random.shuffle(idx_list)
        num_batch = 500
        Explainer.train()
        for k in tqdm(range(num_batch)):
            s_idx = k * args.bs
            e_idx = min(num_instance - 1, s_idx + args.bs)
            if s_idx == e_idx:
                continue
            batch_idx = idx_list[s_idx:e_idx]
#            print(batch_idx)
            src_l_cut, dst_l_cut = src_l[batch_idx], dst_l[batch_idx]
            ts_l_cut = ts_l[batch_idx]
            e_l_cut = e_idx_l[batch_idx]
            subgraph_src, subgraph_tgt, subgraph_bgd, walks_src, walks_tgt, walks_bgd, dst_l_fake = get_item(train_pack,
                                                                                                             batch_idx)
            src_edge, tgt_edge, bgd_edge = get_item_edge(train_edge, batch_idx)
            with torch.no_grad():
                subgraph_src = base_model.grab_subgraph(src_l_cut, ts_l_cut)
                subgraph_tgt = base_model.grab_subgraph(dst_l_cut, ts_l_cut)
                subgraph_bgd = base_model.grab_subgraph(dst_l_fake, ts_l_cut)
                if args.base_type == "tgat":
#                    pos_out_ori, neg_out_ori = base_model.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
#                                                         subgraph_src, subgraph_tgt, subgraph_bgd, test=True,
#                                                         if_explain=False)  #[B, 1]
                    pos_out_ori = base_model.get_temp_me_prob(src_l_cut, dst_l_cut, ts_l_cut,
                                                              subgraph_src, subgraph_tgt, explain_weight=None)
                    neg_out_ori = copy.copy(pos_out_ori)
                else:
                    pos_out_ori, neg_out_ori = base_model.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
                                                            subgraph_src, subgraph_tgt, subgraph_bgd)  # [B, 1]
                y_pred = torch.cat([pos_out_ori, neg_out_ori], dim=0)  # [B*2, 1]
#                y_pred = pos_out_ori
                y_ori = torch.where(y_pred > 0.5, 1., 0.).view(y_pred.size(0), 1)
#            optimizer.zero_grad()
            graphlet_imp_src = Explainer(walks_src, ts_l_cut, src_edge)
            graphlet_imp_tgt = Explainer(walks_tgt, ts_l_cut, tgt_edge)
            graphlet_imp_bgd = Explainer(walks_bgd, ts_l_cut, bgd_edge)
            if args.base_type == "tgat":
#
#                edge_imp_src = Explainer.retrieve_edge_imp(subgraph_src, graphlet_imp_src, walks_src, training=args.if_bern)
#                edge_imp_tgt = Explainer.retrieve_edge_imp(subgraph_tgt, graphlet_imp_tgt, walks_tgt, training=args.if_bern)
#                edge_imp_bgd = Explainer.retrieve_edge_imp(subgraph_bgd, graphlet_imp_bgd, walks_bgd, training=args.if_bern)
                explanation = Explainer.retrieve_explanation(subgraph_src, graphlet_imp_src, walks_src,
                                                            subgraph_tgt, graphlet_imp_tgt, walks_tgt,
                                                            subgraph_bgd, graphlet_imp_bgd, walks_bgd,
                                                            training=args.if_bern)
#                breakpoint()
#                print
#                explain_weight = [[edge_imp_src, edge_imp_tgt], [edge_imp_src, edge_imp_bgd]]
#                explain_weight = [[explanation[0], explanation[1]], [explanation[0], explanation[2]]]
#                pos_logit, neg_logit = base_model.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
#                                                    subgraph_src, subgraph_tgt, subgraph_bgd, test=True,
#                                                    if_explain=True, exp_weights=explain_weight)
                pos_logit = base_model.get_temp_me_prob(src_l_cut, dst_l_cut, ts_l_cut,
                                                              subgraph_src, subgraph_tgt, explain_weight=explanation)
                neg_logit = copy.copy(pos_logit)
            else:
                explanation = Explainer.retrieve_explanation(subgraph_src, graphlet_imp_src, walks_src,
                                                            subgraph_tgt, graphlet_imp_tgt, walks_tgt,
                                                            subgraph_bgd, graphlet_imp_bgd, walks_bgd,
                                                            training=args.if_bern)
                pos_logit, neg_logit = base_model.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
                                                    subgraph_src, subgraph_tgt, subgraph_bgd, explain_weights=explanation)
#            if k%100 == 0:
#                print(pos_logit, neg_logit)

            pred = torch.cat([pos_logit, neg_logit], dim=0).to(args.device)
#            pred = pos_logit.to(args.device)
            pred_loss = criterion(pred, y_pred)
#            pred_loss = criterion(pos_logit, pos_out_ori)
            kl_loss = Explainer.kl_loss(graphlet_imp_src, walks_src, target=args.prior_p) + \
                      Explainer.kl_loss(graphlet_imp_tgt, walks_tgt, target=args.prior_p) + \
                      Explainer.kl_loss(graphlet_imp_bgd, walks_bgd, target=args.prior_p)
            loss = pred_loss + args.beta * kl_loss
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                y_pred = torch.cat([pos_logit, neg_logit], dim=0).sigmoid()
                pred_label = torch.where(y_pred > 0.5, 1., 0.).view(y_pred.size(0), 1)
                fid_prob_batch = torch.cat(
                    [pos_logit.sigmoid() - pos_out_ori.sigmoid(), neg_out_ori.sigmoid() - neg_logit.sigmoid()], dim=0)
                fid_prob = torch.mean(fid_prob_batch, dim=0)
                fid_logit_batch = torch.cat([pos_logit - pos_out_ori, neg_out_ori - neg_logit], dim=0)
                fid_logit = torch.mean(fid_logit_batch, dim=0)
                train_fid_prob.append(fid_prob.item())
                train_fid_logit.append(fid_logit.item())
                train_aps.append(average_precision_score(y_ori.cpu(), y_pred.cpu()))
                train_auc.append(roc_auc_score(y_ori.cpu(), y_pred.cpu()))
                train_acc.append((pred_label.cpu() == y_ori.cpu()).float().mean())
                train_loss.append(loss.item())
                train_pred_loss.append(pred_loss.item())
                train_kl_loss.append(kl_loss.item())

        aps_epoch = np.mean(train_aps)
        auc_epoch = np.mean(train_auc)
        acc_epoch = np.mean(train_acc)
        fid_prob_epoch = np.mean(train_fid_prob)
        fid_logit_epoch = np.mean(train_fid_logit)
        loss_epoch = np.mean(train_loss)
        print((f'Training Epoch: {epoch} | '
               f'Training loss: {loss_epoch} | '
               f'Training Aps: {aps_epoch} | '
               f'Training Auc: {auc_epoch} | '
               f'Training Acc: {acc_epoch} | '
               f'Training Fidelity Prob: {fid_prob_epoch} | '
               f'Training Fidelity Logit: {fid_logit_epoch} | '))

        ### evaluation:
        if (epoch + 1) % args.verbose == 0:
            best_acc = eval_one_epoch(args, base_model, Explainer, full_ngh_finder, test_src_l,
                                      test_dst_l, test_ts_l, test_e_idx_l, epoch, best_acc, test_pack, test_edge)

class TempME_Executor():
    def __init__(self, args, base_model, train_pack, test_pack, train_edge, test_edge, results_dir):
        self.args = args
        self.base_model = base_model
        self.train_pack = train_pack
        self.test_pack = test_pack
        self.train_edge = train_edge
        self.test_edge = test_edge
        self.results_dir = results_dir

        if args.base_type == "tgaat":
            self.Explainer = TempME_TGAT(base_model, data=args.data, out_dim=args.out_dim, hid_dim=args.hid_dim, temp=args.temp,
                                dropout_p=args.drop_out, device=args.device, data_dir=args.data_dir)
        else:
            self.Explainer = TempME(base_model, base_model_type=args.base_type, data=args.data, out_dim=args.out_dim, hid_dim=args.hid_dim,
                               temp=args.temp, if_cat_feature=True,
                               dropout_p=args.drop_out, device=args.device, data_dir=args.data_dir)
        if args.preload:
            model_path = osp.join(osp.dirname(osp.realpath(__file__)), 'params', f'explainer/{args.base_type}/')
            save_path = f"{args.data}.pt"
            self.Explainer = torch.load(osp.join(model_path, save_path))
            print(f"Load model from {osp.join(model_path, save_path)}")

        self.Explainer = self.Explainer.to(args.device)

        self.rand_sampler, self.src_l, self.dst_l, self.ts_l, self.label_l, self.e_idx_l, self.ngh_finder = load_data(mode="training", args=self.args)
        self.test_rand_sampler, self.test_src_l, self.test_dst_l, self.test_ts_l, self.test_label_l, self.test_e_idx_l, self.full_ngh_finder = load_data(
                mode="test", args=args)

        self.base_model.set_neighbor_sampler(self.ngh_finder)

    def __call__(self, target_event_idxs, results_batch=None):

        rb = ['' if results_batch is None else f'_{results_batch}'][0]
        exp_sizes = [10,20,30,40,50,60,70,80,90,100]
#        results = {'target_event_idxs': [], 'explanations': [], 'explanation_predictions': [], 'model_predictions': []}
        results = {e:{'target_event_idxs': [], 'explanation_predictions': [], 'model_predictions': [], 'delta_fidelity': []} for e in exp_sizes}
        for target_idx in tqdm(target_event_idxs):
            curr_result = []

            print(f"Generating explanation for target index {target_idx}")
            batch_idx = [target_idx]

            src_l_cut, dst_l_cut = self.src_l[batch_idx], self.dst_l[batch_idx]
            ts_l_cut = self.ts_l[batch_idx]
            e_l_cut = self.e_idx_l[batch_idx]
            subgraph_src, subgraph_tgt, subgraph_bgd, walks_src, walks_tgt, walks_bgd, dst_l_fake = get_item(self.train_pack,
                                                                                                             batch_idx)
            src_edge, tgt_edge, bgd_edge = get_item_edge(self.train_edge, batch_idx)
            with torch.no_grad():
                subgraph_src = self.base_model.grab_subgraph(src_l_cut, ts_l_cut)
                subgraph_tgt = self.base_model.grab_subgraph(dst_l_cut, ts_l_cut)
                subgraph_bgd = self.base_model.grab_subgraph(dst_l_fake, ts_l_cut)
                if self.args.base_type == "tgat":
#                    pos_out_ori, neg_out_ori = self.base_model.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
#                                                         subgraph_src, subgraph_tgt, subgraph_bgd, test=True,
#                                                         if_explain=True)  #[B, 1]
                    pos_out_ori = self.base_model.get_temp_me_prob(src_l_cut, dst_l_cut, ts_l_cut,
                                                         subgraph_src, subgraph_tgt)
                else:
                    pos_out_ori, neg_out_ori = self.base_model.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
                                                            subgraph_src, subgraph_tgt, subgraph_bgd)  # [B, 1]
#                y_pred = torch.cat([pos_out_ori, neg_out_ori], dim=0).sigmoid()  # [B*2, 1]
#                y_ori = torch.where(y_pred > 0.5, 1., 0.).view(y_pred.size(0), 1)
                curr_result.append(pos_out_ori.detach().cpu().float().float())
                graphlet_imp_src = self.Explainer(walks_src, ts_l_cut, src_edge)
                graphlet_imp_tgt = self.Explainer(walks_tgt, ts_l_cut, tgt_edge)
                graphlet_imp_bgd = self.Explainer(walks_bgd, ts_l_cut, bgd_edge)


            if self.args.base_type == "tgat":
                explanation = self.Explainer.retrieve_explanation(subgraph_src, graphlet_imp_src, walks_src,
                                                            subgraph_tgt, graphlet_imp_tgt, walks_tgt,
                                                            subgraph_bgd, graphlet_imp_bgd, walks_bgd,
                                                            training=self.args.if_bern)
#                breakpoint()
#                print
#                explain_weight = [[edge_imp_src, edge_imp_tgt], [edge_imp_src, edge_imp_bgd]]
#                explain_weight = [[explanation[0], explanation[1]], [explanation[0], explanation[2]]]
#                pos_logit, neg_logit = base_model.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
#                                                    subgraph_src, subgraph_tgt, subgraph_bgd, test=True,
#                                                    if_explain=True, exp_weights=explain_weight)
                pos_logit = self.base_model.get_temp_me_prob(src_l_cut, dst_l_cut, ts_l_cut,
                                                              subgraph_src, subgraph_tgt, explain_weight=explanation)
                neg_logit = copy.copy(pos_logit)
            else:
                explanation = self.Explainer.retrieve_explanation(subgraph_src, graphlet_imp_src, walks_src,
                                                            subgraph_tgt, graphlet_imp_tgt, walks_tgt,
                                                            subgraph_bgd, graphlet_imp_bgd, walks_bgd,
                                                            training=self.args.if_bern)

                pos_logit, neg_logit = self.base_model.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
                                                                subgraph_src, subgraph_tgt, subgraph_bgd, explain_weights=explanation)
            for exp_size in exp_sizes:
                reconstructed_explanation, inverse_explanation = self.extract_important_events(explanation, exp_size=exp_size)
                if self.args.base_type == "tgn":
                    exp_pos_logit, exp_neg_logit = self.base_model.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut, subgraph_src, subgraph_tgt, subgraph_bgd, explain_weights=reconstructed_explanation)

                    inverse_pos_logit, inverse_neg_logit = self.base_model.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut, subgraph_src, subgraph_tgt, subgraph_bgd, explain_weights=inverse_explanation)
                else:
                    exp_pos_logit = self.base_model.get_temp_me_prob(src_l_cut, dst_l_cut, ts_l_cut,
                                          subgraph_src, subgraph_tgt, explain_weight=reconstructed_explanation)

                    inverse_pos_logit = self.base_model.get_temp_me_prob(src_l_cut, dst_l_cut, ts_l_cut,
                                            subgraph_src, subgraph_tgt, explain_weight=inverse_explanation)
                pos_out_ori = pos_out_ori.detach().cpu().float()
#                    neg_out_ori = neg_out_ori.detach().cpu().float()
                exp_pos_logit = exp_pos_logit.detach().cpu().float()
#                    exp_neg_logit = exp_neg_logit.detach().cpu().float()
                inverse_pos_logit = inverse_pos_logit.detach().cpu().float()
#                    inverse_neg_logit = inverse_neg_logit.detach().cpu().float()

                delta_fidelity = abs(pos_out_ori - inverse_pos_logit)/abs(pos_out_ori - exp_pos_logit)
#                    delta_fidelity = np.mean([abs(exp_pos_logit - inverse_pos_logit), abs(exp_neg_logit - inverse_neg_logit)])
                print(f"Exp Size: {exp_size}, Model Pred: {pos_out_ori}, Exp Pred: {exp_pos_logit}, Unimp Pred: {inverse_pos_logit}, Delta Fidelity: {delta_fidelity}")
                results[exp_size]['target_event_idxs'].append(target_idx)
#                    results[exp_size]['explanations'].append(exp_events)
                results[exp_size]['explanation_predictions'].append(exp_pos_logit.detach().cpu().float())
                results[exp_size]['model_predictions'].append(pos_out_ori.detach().cpu().float())
                results[exp_size]['delta_fidelity'].append(delta_fidelity)
#            print(curr_result)
#            for exp_size in exp_sizes:
#                exp_events, sorted_data = self.extract_important_events_(explanation, subgraph_src, subgraph_tgt, exp_size=exp_size)
#                exp_absolute_error, target_model_y, target_explanation_y = self.calculate_scores(exp_events, target_idx)
#
#                print(f"Exp Error: {exp_absolute_error}, Exp Size: {len(exp_events)}, Model Prediction: {target_model_y}, Explanation Prediction: {target_explanation_y}")
#
        with open(self.results_dir + f'/temp_me_results_{self.args.data}_{self.args.base_type}_exp_sizes{rb}.pkl', 'wb') as f:
            pck.dump(results, f)

    def extract_important_events_(self, explanation, subgraph_src, subgraph_tgt, exp_size=None):
        src_nodes_in_motifs = subgraph_src[0][1][0]
        src_nodes_event_indexs = subgraph_src[1][1][0]
        tgt_nodes_in_motifs = subgraph_tgt[0][1][0]
        tgt_nodes_event_indexs = subgraph_tgt[1][1][0]

        src_node_event_importances = explanation[1][0]
        tgt_node_event_importances = explanation[1][1]

        events, nodes, importances = [], [], []

        events.extend(src_nodes_event_indexs)
        nodes.extend(src_nodes_in_motifs)
        importances.extend(src_node_event_importances)

        events.extend(tgt_nodes_event_indexs)
        nodes.extend(tgt_nodes_in_motifs)
        importances.extend(tgt_node_event_importances)

        events = [int(e) for e in events]
        nodes = [int(n) for n in nodes]
        importances = [float(i) for i in importances]


        transposed = list(zip(*[events, nodes, importances]))
# Sort based on the desired sublist
        sorted_transposed = sorted(transposed, key=lambda x: x[2], reverse=True)

# Transpose back to the original structure
        sorted_data = [list(sublist) for sublist in zip(*sorted_transposed)]
        sorted_important_events = sorted_data[0]

        unique_events = []
        for event in sorted_important_events:
            if event not in unique_events:
                unique_events.append(event)


        if exp_size is not None:
            if exp_size > len(unique_events):
                exp_size = len(unique_events)
            else:
#                unique_events = unique_events[:exp_size]
                unique_events = np.random.choice(unique_events, exp_size, replace=False)


        exp_events = unique_events
        return exp_events, sorted_data

    def extract_important_events(self, explanation, exp_size):
        reconstructed_explanation = [torch.zeros(explanation[0].shape), torch.zeros(explanation[1].shape)]
        inverse_explanation = [torch.ones(explanation[0].shape), torch.zeros(explanation[1].shape)]

        for exp_part in range(len(explanation)):
            importances = []
            loc_indexs = []
            sub_locs = []
            for s, imps in enumerate(explanation[exp_part]):
                importances.extend(imps.tolist())
                loc_indexs.extend([i for i in range(len(imps))])
                sub_locs.extend([s]*len(imps))


#df = pd.DataFrame(columns=['loc index', 'importance', 'sub_loc'])
            df = pd.DataFrame({'loc index': loc_indexs, 'importance': importances, 'sub_loc': sub_locs})
            df = df[df['importance'] > 0.0]
            df = df.sort_values(by=['importance'], ascending=False)



#            if exp_part == 1:
#                exp_size = len(df)
            df = df[:exp_size]
#            df = df[:np.random.randint(1, len(df))]
            for i, row in df.iterrows():
                reconstructed_explanation[exp_part][int(row['sub_loc']), int(row['loc index'])] = explanation[exp_part][int(row['sub_loc']), int(row['loc index'])]
#                inverse_explanation[exp_part][int(row['sub_loc']), int(row['loc index'])] = 0.0


        reconstructed_explanation = [reconstructed_explanation[0].to(self.args.device), reconstructed_explanation[1].to(self.args.device)]
        inverse_explanation = [inverse_explanation[0].to(self.args.device), inverse_explanation[1].to(self.args.device)]

        return reconstructed_explanation, inverse_explanation

