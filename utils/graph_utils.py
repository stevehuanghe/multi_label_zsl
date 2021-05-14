"""
Adapted from Wei-Chung's code for Multi-label Zero-shot Learning with Structured Knowledge Graphs
"""
from itertools import combinations

from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer


import torch
import numpy as np
import pickle as pk

def readclass(path):
    with open(path, 'r') as f:
        ls = [i.strip() for i in f]
    return ls


def get_synsetlist(classlist, dataset_name, use_wnid=False):

    Lem = WordNetLemmatizer()

    synsetlist = []
    for i in classlist:
        if use_wnid:
            wtype = i[0]
            offset = int(i[1:])
            synset = wn.synset_from_pos_and_offset(wtype, offset)
            synsetlist.append((synset.name().split('.')[0], synset))
            continue
        word = Lem.lemmatize(i)
        try:    
            if dataset_name == 'coco':
                if len(word.split(' ')) == 1:
                    tmp = word+'.n.01'
                elif word == 'stop sign':
                    tmp = 'stop.n.05'
                elif word == 'sports ball':
                    tmp = 'ball.n.01'
                elif word == 'wine glass':
                    tmp = 'glass.n.02'
                elif word == 'hot dog':
                    tmp = 'hot_dog.n.02'
                elif word == 'potted plant':
                    tmp = 'plant.n.02'
                elif word == 'cell phone':
                    tmp = 'cellphone.n.01'
                else:
                    tmp = word.replace(' ', '_') + '.n.01'
            else:
                tmp = word+'.n.01'

            tmp = wn.synset(tmp)
            synsetlist.append((word, tmp))
        except:
            try:
                tmp = wn.synset(word + '.a.01')
                synsetlist.append((word, tmp))
            except:
                synsetlist.append((word, None))
    return synsetlist


def get_hyposets(synsetlist):
    hyposets = []
    for _, synset in synsetlist:
        if synset is not None:
            hypos = set([i for i in synset.closure(lambda s: s.hyponyms())])
        else:
            hypos = set([])
        hyposets.append(hypos)

    return hyposets


def get_cooc_edges(path, thres=0.1):
    targets = torch.load(path).numpy()

    cooc_mat = np.zeros((81, 81)).astype(int)

    for n in range(targets.shape[0]):
        target = targets[n]
        idx = np.argsort(target)[::-1]
        n_tar = target.sum()
        tars = idx[:n_tar].tolist()
        for a, b in combinations(tars, 2):
            cooc_mat[a,b] += 1
            cooc_mat[b,a] += 1

    ratio = cooc_mat.copy().astype(float)
    ratio /= np.sum(ratio, axis=0, keepdims=True)

    edges_mat = (ratio > thres).astype(int)
    return edges_mat


def get_edges(classes, dataset_name, unseen=0, neg=0.1, pos=0.8, use_wnid=False, node_dim=5):
    edge_types = 4
    edges = []
    for _ in range(edge_types):
        edges.append(([], []))

    synsets = get_synsetlist(classes, dataset_name, use_wnid)
    hyposets = get_hyposets(synsets)
    #cooc_mat = get_cooc_edges('data/nus/processed_data/train_targets.tensor',
    #                          thres=0.1)

    n_nodes = len(synsets) - unseen
    for i in range(n_nodes):
        for j in range(i+1,n_nodes):
            if i == j:
                continue
            si, sj = synsets[i][1], synsets[j][1]
            if (not si) or (not sj):
                continue

            if si in hyposets[j]:
                edges[2][0].append(i)
                edges[2][1].append(j)
                edges[3][0].append(j)
                edges[3][1].append(i)
            elif sj in hyposets[i]:
                edges[2][0].append(j)
                edges[2][1].append(i)
                edges[3][0].append(i)
                edges[3][1].append(j)
            #elif cooc_mat[i, j] == 1:
            #    edges[4][0].append(i)
            #    edges[4][1].append(j)
            else:
                sim = si.wup_similarity(sj)
                # if sim and sim > 0.5:
                if sim and sim > pos:
                    edges[0][0].append(i)
                    edges[0][1].append(j)
                    edges[0][0].append(j)
                    edges[0][1].append(i)
                elif not sim or 0 < sim < neg:
                # elif not sim or 0 < sim < 0.11:
                    edges[1][0].append(i)
                    edges[1][1].append(j)
                    edges[1][0].append(j)
                    edges[1][1].append(i)

    print('-' * 80)
    print('Number of edges:')
    print(f'Type 1 (semantically positive): {len(edges[0][0])}')
    print(f'Type 2 (semantically negative): {len(edges[1][0])}')
    print(f'Type 3 (Parent-Child): {len(edges[2][0])}')
    print(f'Type 4 (Child-Parent): {len(edges[3][0])}')
    # print(f'Type 5 (Co-ocurrence): {len(edges[4][0])}')

    edge_vars = []
    mat_vars_u = []
    mat_vars_v = []
    # node_dim = 5
    for idx, (u_ids, v_ids) in enumerate(edges):
        edge_vars.append((
            torch.LongTensor(u_ids),
            torch.LongTensor(v_ids)))

        total_len = len(u_ids) * (node_dim ** 2)
        u_arr = np.array(u_ids)
        v_arr = np.array(v_ids)
        v_arr = np.repeat(v_arr, node_dim)
        v_arr = np.hstack([node_dim * v_arr[:, np.newaxis] + i 
                           for i in range(node_dim)]).reshape(total_len)

        u_arr = np.repeat(u_arr[:, np.newaxis], node_dim, axis=1)
        u_arr = np.hstack([node_dim * u_arr + i 
                           for i in range(node_dim)]).reshape(total_len)
        mat_vars_u.append(u_arr)
        mat_vars_v.append(v_arr)

    mat_vars = [
        torch.from_numpy(np.concatenate(mat_vars_u, 0)),
        torch.from_numpy(np.concatenate(mat_vars_v, 0))]

    return edge_vars, mat_vars


def get_edges_by_synsets(synsets, unseen=0, neg=0.1, pos=0.8, device=torch.device("cpu"), node_dim=5, skg=False):
    edge_types = 4
    edges = []
    wup_pos = []
    wup_neg = []
    for _ in range(edge_types):
        edges.append(([], []))


#     synsets = get_synsetlist(classes, dataset_name, use_wnid)
    hyposets = get_hyposets(synsets)
    #cooc_mat = get_cooc_edges('data/nus/processed_data/train_targets.tensor',
    #                          thres=0.1)

    n_nodes = len(synsets) - unseen
    for i in range(n_nodes):
        for j in range(i+1,n_nodes):
            if i == j:
                continue
            si, sj = synsets[i][1], synsets[j][1]
            if (not si) or (not sj):
                continue

            if si in hyposets[j]:
                edges[2][0].append(i)
                edges[2][1].append(j)
                edges[3][0].append(j)
                edges[3][1].append(i)
            elif sj in hyposets[i]:
                edges[2][0].append(j)
                edges[2][1].append(i)
                edges[3][0].append(i)
                edges[3][1].append(j)
            #elif cooc_mat[i, j] == 1:
            #    edges[4][0].append(i)
            #    edges[4][1].append(j)
            # else:
            sim = si.wup_similarity(sj)
            if sim is None:
                sim = 0
            # if sim and sim > 0.5:
            if sim and sim >= pos:
                edges[0][0].append(i)
                edges[0][1].append(j)
                edges[0][0].append(j)
                edges[0][1].append(i)
                wup_pos.append(sim)
                wup_pos.append(sim)
            elif not sim or 0 < sim < neg:
            # elif not sim or 0 < sim < 0.11:
                edges[1][0].append(i)
                edges[1][1].append(j)
                edges[1][0].append(j)
                edges[1][1].append(i)
                wup_neg.append(sim)
                wup_neg.append(sim)

    print('-' * 80)
    print(f'Number of nodes: {n_nodes}')
    print('Number of edges:')
    print(f'Type 1 (semantically positive): {len(edges[0][0])}')
    print(f'Type 2 (semantically negative): {len(edges[1][0])}')
    print(f'Type 3 (Parent-Child): {len(edges[2][0])}')
    print(f'Type 4 (Child-Parent): {len(edges[3][0])}')
    # print(f'Type 5 (Co-ocurrence): {len(edges[4][0])}')

    edge_vars = []
    mat_vars_u = []
    mat_vars_v = []
    for idx, (u_ids, v_ids) in enumerate(edges):
        if skg:
            edge_vars.append((
                torch.LongTensor(u_ids),
                torch.LongTensor(v_ids)))
        else:
            edge_vars.append([u_ids, v_ids])

        total_len = len(u_ids) * (node_dim ** 2)
        u_arr = np.array(u_ids)
        v_arr = np.array(v_ids)
        v_arr = np.repeat(v_arr, node_dim)
        v_arr = np.hstack([node_dim * v_arr[:, np.newaxis] + i
                           for i in range(node_dim)]).reshape(total_len)

        u_arr = np.repeat(u_arr[:, np.newaxis], node_dim, axis=1)
        u_arr = np.hstack([node_dim * u_arr + i
                           for i in range(node_dim)]).reshape(total_len)
        mat_vars_u.append(u_arr)
        mat_vars_v.append(v_arr)

    mat_vars = [
        torch.from_numpy(np.concatenate(mat_vars_u, 0)).to(device),
        torch.from_numpy(np.concatenate(mat_vars_v, 0)).to(device)]

    wup_pos = torch.tensor(wup_pos).float().to(device)
    wup_neg = torch.tensor(wup_neg).float().to(device)

    return edge_vars, mat_vars, (wup_pos, wup_neg)


def merge_synset_list(list1, list2, data="coco"):
    result = []
    book = {}
    l1names = []
    for i, (w, s) in enumerate(list1):
        book[w] = 1
        result.append((w, s))
        l1names.append(w)

    common_idx2 = []
    common_idx1 = []
    unique_idx2 = []
    names = []
    if data == "coco":
        commons = [21, 340, 414, 620, 651, 673, 760, 795, 859, 879, 883, 937, 950, 954, 963, 968]
    else:
        commons = [288, 291, 292, 308, 340, 414, 425, 429, 430, 472, 483, 497, 536, 541, 562, 609, 624, 663, 668, 698,
                        718, 726, 733, 743, 762, 833, 837, 879, 908, 950, 972, 979, 980, 982]

    for i, (w, s) in enumerate(list2):
        if i not in commons:
            result.append((w, s))
            unique_idx2.append(i)
        else:
            common_idx2.append(i)
            names.append(w)

    unique_idx1 = [i for i in range(len(list1)) if i not in common_idx1]

    return {
        "merged_list": result,
        "common_idx1": np.array(common_idx1, dtype=np.long),
        "common_idx2": np.array(common_idx2, dtype=np.long),
        "unique_idx1": np.array(unique_idx1, dtype=np.long),
        "unique_idx2": np.array(unique_idx2, dtype=np.long),
        "common_cats": names
    }


def load_graph(coco_cats, imgnet_file, neg=0.1, pos=0.5, to_dense=True, binary=False,
               device=torch.device("cpu"), data_name="coco", node_dim=5, skg=False):
    if imgnet_file is not None:
        with open(imgnet_file, "rb") as fin:
            imgnet_meta = pk.load(fin)

        imgnet_cats = list(imgnet_meta["wnid_to_idx"].keys())

        synset_imgnet = get_synsetlist(imgnet_cats, "imgnet", True)
    else:
        synset_imgnet = []

    synset_coco = get_synsetlist(coco_cats, data_name)

    synset_results = merge_synset_list(synset_coco, synset_imgnet, data_name)

    synset_merged = synset_results["merged_list"]

    edge_vars, mat_vars, (wup_pos, wup_neg) = get_edges_by_synsets(synset_merged, 0, neg, pos, device, node_dim=node_dim, skg=skg)

    imgnet_idx = torch.from_numpy(synset_results["unique_idx2"]).long().to(device)

    if skg:
        return {
            "names": [x[0] for x in synset_merged],
            "edges": edge_vars,
            "mat": mat_vars,
            "imgnet_idx": imgnet_idx,
        }

    n_class = len(synset_merged)
    idx_pos = torch.LongTensor(edge_vars[0])
    idx_neg = torch.LongTensor(edge_vars[1])
    if binary:
        ones = torch.ones_like(wup_pos).float().to(device)
        edge_pos = torch.sparse_coo_tensor(idx_pos, ones, torch.Size([n_class, n_class]))
        edge_neg = torch.sparse_coo_tensor(idx_neg, ones, torch.Size([n_class, n_class]))
    else:
        edge_pos = torch.sparse_coo_tensor(idx_pos, wup_pos, torch.Size([n_class, n_class]))
        edge_neg = torch.sparse_coo_tensor(idx_neg, wup_neg, torch.Size([n_class, n_class]))

    if to_dense:
        edge_pos = edge_pos.to_dense()
        edge_neg = edge_neg.to_dense()

    data = {
        "names": [x[0] for x in synset_merged],
        "edges": [torch.LongTensor(x) for x in edge_vars],
        "mat": mat_vars,
        "imgnet_idx": imgnet_idx,
        "wup_pos": wup_pos,
        "wup_neg": wup_neg,
        "adj_wup_pos": edge_pos,
        "adj_wup_neg": edge_neg,
        "edge_vars":  edge_vars
    }

    return data