import networkx as nx
import matplotlib.pyplot as plt
import pymetis
import numpy as np
import json
import os
import scipy.sparse as spp
import dgl
import torch as th
import pandas as pd
from tqdm import tqdm
import time
from collections import defaultdict
from dgl.transform import partition_graph_with_halo
from dgl.data.utils import load_graphs, save_graphs, load_tensors, save_tensors
from dgl.base import NID, EID, NTYPE, ETYPE, dgl_warning
import new_sampling_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import backend as P
from dgl.convert import to_homogeneous
from dgl.utils.internal import toindex

n_graphs = {}

def findDegree(graph):
    return [val for (n, val) in graph.degree()]

def weight(node, sampling_length, degrees):
    probability_weight = 1 - pow((1 - 1 / degrees[node]), sampling_length)
    return int(probability_weight * 100)

# converts from adjacency matrix to adjacency list
def convert(a):
    adjList = defaultdict(list)
    for i in range(len(a)):
        for j in range(len(a[i])):
                       if a[i][j]== 1:
                           adjList[i].append(j)
    return adjList

edges = []
# Generate Edges from node and adjacent node
def generate_edges(graph):
    # edges = []

    # for each node in graph
    for node in graph:
        # for each neighbour node of a single node
        for neighbour in graph[node]:
            # if edge exists then append
            edges.append((node, neighbour))
    # print("Edges",edges)
    return edges

# function to get unique values
def unique(list1):
    x = np.array(list1)
    print(np.unique(x))

def _get_inner_node_mask(graph, ntype_id):
    if NTYPE in graph.ndata:
        dtype = P.dtype(graph.ndata['inner_node'])
        return graph.ndata['inner_node'] * P.astype(graph.ndata[NTYPE] == ntype_id, dtype) == 1
    else:
        return graph.ndata['inner_node'] == 1

def _get_inner_edge_mask(graph, etype_id):
    if ETYPE in graph.edata:
        dtype = P.dtype(graph.edata['inner_edge'])
        return graph.edata['inner_edge'] * P.astype(graph.edata[ETYPE] == etype_id, dtype) == 1
    else:
        return graph.edata['inner_edge'] == 1

def improved_partition_graph(g, num_parts, sample_length, out_path, reshuffle, graph_name='test', balance_ntypes=None, balance_edges=False):
    degrees = (g.in_degrees()).tolist()
    eweights = []
    embed = nn.Embedding(34, 5)  # 34 nodes with embedding dim equal to 5
    # ndata is a syntax sugar to access the feature data of all nodes
    g.ndata['feat'] = embed.weight

    adj = convert(g.adj())
    adjncy = []
    xadj = [0]
    size = 1
    partitions = []
    partition_sets = []
    Halo_node = []

    for node in adj:
        adjacency = adj[node]
        adjncy += adjacency

        for neighbor in adjacency:
            eweights.append(weight(neighbor, sample_length, degrees))
        xadj.append(xadj[size - 1] + len(adjacency))
        size += 1

    n_cuts, membership = pymetis.part_graph(num_parts, adjacency=None, xadj=xadj, adjncy=adjncy, vweights=None,
                                            eweights=eweights, recursive=False)
    
    
    def get_homogeneous(g, balance_ntypes):
        if len(g.etypes) == 1:
            sim_g = to_homogeneous(g)
            if isinstance(balance_ntypes, dict):
                assert len(balance_ntypes) == 1
                bal_ntypes = list(balance_ntypes.values())[0]
            else:
                bal_ntypes = balance_ntypes
        return sim_g, bal_ntypes

    for i in range(num_parts):
        partition_data = np.argwhere(np.array(membership) == i).ravel()
        partitions.append(partition_data)
        partition_sets.append(set(partition_data))

        destination = []
        src_ids = []
        dst_ids = []
        all_node = []

        for p in partition_data:
            destination.append(adj[p])
        
        g_edge = dict(zip(partition_data,destination))
         
        for x,y in generate_edges(g_edge):
            src_ids.append(x)
            dst_ids.append(y)
        
        flatten = [element for items in destination for element in items]   #Added on 26th Jan
        unique_nodes = set(flatten)
        Halo_node = unique_nodes - set(partition_data)

        src_ids_th = th.tensor(src_ids)
        dst_ids_th = th.tensor(dst_ids)
        number_node = Halo_node.union(set(partition_data))
        # print("source id", src_ids)
        # print("destination id", dst_ids)

        new_g = dgl.graph((src_ids_th, dst_ids_th))

        # print("graph nodes", new_g.nodes())
        # print("graph edges", new_g.edges())
        # print("new_g",new_g)
        # n_graphs[i] = new_g.to_networkx().to_undirected()
        # print("new_g",n_graphs[i])
        
        ng = new_g.to_networkx().to_undirected()
        pos = nx.kamada_kawai_layout(ng)
        nx.draw(ng, pos, with_labels=True, node_color=[[.7, .7, .7]])
        plt.savefig("graph_{}.png".format(i))

        node_feats = {}
        edge_feats = {}
        sg1 = g.subgraph(partition_data)      
        node_feats['item'] = sg1.ndata['feat']
        
        node_parts = P.zeros((sg1.number_of_nodes(),), th.int64, th.device('cpu'))
        print("number of parts",node_parts) #########################
        parts = {0: sg1.clone()}
        # part = parts[i]

        # # Let's calculate edge assignment.
        # if not reshuffle:
        #     start = time.time()
        #     # We only optimize for reshuffled case. So it's fine to use int64 here.
        #     edge_parts = np.zeros((g.number_of_edges(),), dtype=np.int64) - 1
        #     for part_id in parts:
        #         part = parts[part_id]
        #         # To get the edges in the input graph, we should use original node IDs.
        #         local_edges = P.boolean_mask(part.edata[EID], part.edata['inner_edge'])
        #         edge_parts[P.asnumpy(local_edges)] = part_id
        #     print('Calculate edge assignment: {:.3f} seconds'.format(time.time() - start))
        orig_nids = parts[0].ndata[NID] = P.arange(0, sg1.number_of_nodes())
        orig_eids = parts[0].edata[EID] = P.arange(0, sg1.number_of_edges())
        
        node_map_val = {}
        edge_map_val = {}
        if not reshuffle:
            node_part_file = os.path.join(out_path, "node_map")
            edge_part_file = os.path.join(out_path, "edge_map")
            np.save(node_part_file, P.asnumpy(node_parts), allow_pickle=False)
            # np.save(edge_part_file, edge_parts, allow_pickle=False)
            node_map_val = node_part_file + ".npy"
            edge_map_val = edge_part_file + ".npy"
        # else:
        #     print("HERE")
        #     if num_parts > 1:
        #         node_map_val = {}
        #         edge_map_val = {}
        #         for ntype in g.ntypes:
        #             ntype_id = g.get_ntype_id(ntype)
        #             val = []
        #             node_map_val[ntype] = []
        #             # for i in parts:
                    #     inner_node_mask = _get_inner_node_mask(parts[i], ntype_id)
                    #     val.append(P.as_scalar(F.sum(F.astype(inner_node_mask, F.int64), 0)))
                    #     inner_nids = P.boolean_mask(parts[i].ndata[NID], inner_node_mask)
                    #     node_map_val[ntype].append([int(P.as_scalar(inner_nids[0])),
                    #                                 int(P.as_scalar(inner_nids[-1])) + 1])
                    # val = np.cumsum(val).tolist()
                    # assert val[-1] == g.number_of_nodes(ntype)
                # for etype in g.etypes:
                #     etype_id = g.get_etype_id(etype)
                #     val = []
                #     edge_map_val[etype] = []
                    # for i in parts:
                    #     inner_edge_mask = _get_inner_edge_mask(parts[i], etype_id)
                    #     val.append(P.as_scalar(P.sum(P.astype(inner_edge_mask, F.int64), 0)))
                    #     inner_eids = np.sort(P.asnumpy(P.boolean_mask(parts[i].edata[EID],
                    #                                                   inner_edge_mask)))
                    #     edge_map_val[etype].append([int(inner_eids[0]), int(inner_eids[-1]) + 1])
                    # val = np.cumsum(val).tolist()
                    # assert val[-1] == g.number_of_edges(etype)

            # Double check that the node IDs in the global ID space are sorted.
            # for ntype in node_map_val:
            #     val = np.concatenate([np.array(l) for l in node_map_val[ntype]])
            #     assert np.all(val[:-1] <= val[1:])
            # for etype in edge_map_val:
            #     val = np.concatenate([np.array(l) for l in edge_map_val[etype]])
            #     assert np.all(val[:-1] <= val[1:])
    
        
        os.makedirs(out_path, mode=0o775, exist_ok=True)
        out_path = os.path.abspath(out_path)
        start = time.time()
        ntypes = {ntype:g.get_ntype_id(ntype) for ntype in g.ntypes}
        etypes = {etype:g.get_etype_id(etype) for etype in g.etypes}
        part_metadata = {'graph_name': graph_name,
                 'num_nodes': g.number_of_nodes(),
                 'num_edges': g.number_of_edges(),
                 # 'part_method': part_method,
                 'num_parts': num_parts,
                 # 'halo_hops': num_hops,
                 'node_map': node_map_val,
                 'edge_map': edge_map_val,
                 'ntypes': ntypes,
                 'etypes': etypes
                 }       
        
        orig_nids = parts[0].ndata[NID] = P.arange(0, sg1.number_of_nodes())
        orig_eids = parts[0].edata[EID] = P.arange(0, sg1.number_of_edges())

        # # Get the node/edge features of each partition.
        if num_parts > 1:
            node_feats = {}
            edge_feats = {}
            for ntype in g.ntypes:
                ntype_id = g.get_ntype_id(ntype)

                # To get the edges in the input graph, we should use original node IDs.
                # Both orig_id and NID stores the per-node-type IDs.
                ndata_name = 'orig_id' if reshuffle else NID
                # inner_node_mask = _get_inner_node_mask(part, ntype_id)
                # This is global node IDs.
                # local_nodes = P.boolean_mask(part.ndata[ndata_name], inner_node_mask)
                local_nodes = partition_sets
                if len(g.ntypes) > 1:
                    # If the input is a heterogeneous graph.
                    local_nodes = P.gather_row(sim_g.ndata[NID], local_nodes)
                    print('part {} has {} nodes of type {} and {} are inside the partition'.format(
                        i, P.as_scalar(F.sum(part.ndata[NTYPE] == ntype_id, 0)),
                        ntype, len(local_nodes)))
                else:
                    # print('part {} has {} nodes and {} are inside the partition'.format(
                    #     part_id, part.number_of_nodes(), len(local_nodes)))
                    print('part {} has {} nodes and {} are inside the partition'.format(
                        i, len(number_node), len(list(partition_data))))

                for name in g.nodes[ntype].data:
                    if name in [NID, 'inner_node']:
                        continue
                    # node_feats[ntype + '/' + name] = P.gather_row(g.nodes[ntype].data[name],
                    #                                               local_nodes)

            for etype in g.etypes:
                etype_id = g.get_etype_id(etype)
                edata_name = 'orig_id' if reshuffle else EID
                # inner_edge_mask = _get_inner_edge_mask(part, etype_id)
                # This is global edge IDs.
                local_edges = edges
                # local_edges = P.boolean_mask(part.edata[edata_name], inner_edge_mask)
                if len(g.etypes) > 1:
                    local_edges = P.gather_row(sim_g.edata[EID], local_edges)
                    print('part {} has {} edges of type {} and {} are inside the partition'.format(
                        part_id, P.as_scalar(F.sum(part.edata[ETYPE] == etype_id, 0)),
                        etype, len(local_edges)))
                else:
                    print('part {} has {} edges and {} are inside the partition'.format(
                        i, len(local_edges), len(local_edges)))
                # tot_num_inner_edges += len(local_edges)

                for name in g.edges[etype].data:
                    if name in [EID, 'inner_edge']:
                        continue
                    # edge_feats[etype + '/' + name] = P.gather_row(g.edges[etype].data[name],
                    #                                               local_edges)
        # else:
        #     for ntype in g.ntypes:
        #         if reshuffle and len(g.ntypes) > 1:
        #             ndata_name = 'orig_id'
        #             ntype_id = g.get_ntype_id(ntype)
        #             inner_node_mask = _get_inner_node_mask(part, ntype_id)
        #             # This is global node IDs.
        #             local_nodes = P.boolean_mask(part.ndata[ndata_name], inner_node_mask)
        #             local_nodes = P.gather_row(sim_g.ndata[NID], local_nodes)
        #         elif reshuffle:
        #             local_nodes = sim_g.ndata[NID]
        #         for name in g.nodes[ntype].data:
        #             if name in [NID, 'inner_node']:
        #                 continue
        #             if reshuffle:
        #                 node_feats[ntype + '/' + name] = P.gather_row(g.nodes[ntype].data[name],
        #                                                               local_nodes)
        #             else:
        #                 node_feats[ntype + '/' + name] = g.nodes[ntype].data[name]
        #     for etype in g.etypes:
        #         if reshuffle and len(g.etypes) > 1:
        #             edata_name = 'orig_id'
        #             etype_id = g.get_etype_id(etype)
        #             inner_edge_mask = _get_inner_edge_mask(part, etype_id)
        #             # This is global edge IDs.
        #             local_edges = P.boolean_mask(part.edata[edata_name], inner_edge_mask)
        #             local_edges = P.gather_row(sim_g.edata[EID], local_edges)
        #         elif reshuffle:
        #             local_edges = sim_g.edata[EID]
        #         for name in g.edges[etype].data:
        #             if name in [EID, 'inner_edge']:
        #                 continue
        #             if reshuffle:
        #                 edge_feats[etype + '/' + name] = P.gather_row(g.edges[etype].data[name],
        #                                                               local_edges)
        #             else:
        #                 edge_feats[etype + '/' + name] = g.edges[etype].data[name]

        part_dir = os.path.join(out_path, "part" + str(i))
        node_feat_file = os.path.join(part_dir, "node_feat.dgl")
        edge_feat_file = os.path.join(part_dir, "edge_feat.dgl")
        part_graph_file = os.path.join(part_dir, "graph.dgl")
        part_metadata['part-{}'.format(i)] = {
            'node_feats': os.path.relpath(node_feat_file, out_path),
            'edge_feats': os.path.relpath(edge_feat_file, out_path),
            'part_graph': os.path.relpath(part_graph_file, out_path)}
        os.makedirs(part_dir, mode=0o775, exist_ok=True)
        save_tensors(node_feat_file, node_feats)
        save_tensors(edge_feat_file, edge_feats)

        save_graphs(part_graph_file, [new_g])
        
        # np.save(node_part_file, (sg1.ndata['feat']).detach().numpy(), allow_pickle=False)

    with open('{}/{}.json'.format(out_path, graph_name), 'w') as outfile:
        json.dump(part_metadata, outfile, sort_keys=True, indent=4)
    print('Save partitions: {:.3f} seconds'.format(time.time() - start))
    
    # for k in n_graphs:
    #     ng = n_graphs[k]
    #     print("node of graph",ng)
    #     pos = nx.kamada_kawai_layout(ng)
    #     nx.draw(ng, pos, with_labels=True, node_color=[[.7, .7, .7]])
    #     plt.savefig("graph_{}.png".format(k))
        # print(ng.number_of_nodes())
        # print(ng.number_of_edges())
        # print(ng.nodes())
        # print(ng.edges())