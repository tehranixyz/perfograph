import programl as pg
import json
import dgl
from os import listdir
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import IntTensor
import re
import numpy as np
import logging
from tqdm import tqdm


# Digit extraction starts here
def is_numeric(txt):
    if re.match(r"^[-+]?[0-9]*\.?[0-9]+(e[-+]?[0-9]+)?$", txt):
        return True
    else:
        return False

def get_digit_and_pos(txt):
    digits_array = []
    digits_pos_array = []
    if is_numeric(txt) == True:
        point_position = txt.find('.')
        if point_position > -1:
            for i in range(0, point_position):
                digits_array.append(txt[i])
            for i in reversed(range(point_position)):
                digits_pos_array.append(str(i))
            j = -1
            for i in range(point_position+1, len(txt)):
                digits_array.append(txt[i])
                digits_pos_array.append(str(j))
                j = j - 1
        else:
            for i in range(0, len(txt)):
                digits_array.append(txt[i])
            for i in reversed(range(len(txt))):
                digits_pos_array.append(str(i))
    # print(digits_array)
    # print(digits_pos_array)
    return digits_array, digits_pos_array

def get_hetero_graph(file: str, source_file_path: str, label_count: int,  llvm_version: str = '10'):
    def get_node_attr(node):
        text = node.get('text')
        full_text = node.get('features')
        if full_text is not None:
            full_text = full_text.get('full_text')
        return [text, full_text]

    try:
        with open(file) as f:
            file_content_str = f.read()
            if file[-2:] == 'll':
                # this is a llvm IR file
                graph = pg.from_llvm_ir(file_content_str, version=llvm_version)
            elif file[-3:] == 'cpp':
                graph = pg.from_cpp(file_content_str)
            else:
                raise 'Only LLVM IR (*.ll) or CPP files are accepted.'
    except:
        raise "An error Happened, Perhaps file is not found"

    graph_json = pg.to_json(graph)

    # PerfoGraph Here
    # UPDATING THE EDGES
    # We want to have one node as the child of `alloca`
    for node in graph_json.get('nodes'):
        node_text = node.get('text')
        if node_text == 'alloca':
            # find the first child
            first_child = None
            other_children = []
            # Finding the first child and other children of this node
            for edge_idx, edge in enumerate(graph_json.get('links')):
                # skip any edge that is not a DATA edge
                if edge.get('flow') != 1:
                    continue

                # if source of the edge is `alloca` node, this could be the edge to the first child
                if (first_child is None) and (edge.get('source') == node.get('id')):
                    first_child = edge.get('target')
                # finding other children of `alloca`
                elif edge.get('source') == node.get('id'):
                    other_children.append(edge.get('target'))
                    # remove the edges pointing to other children
                    del graph_json['links'][edge_idx]

            # remove the node of other_children
            for node_idx, node in enumerate(graph_json['nodes']):
                node_id = node.get('id')
                if node_id in other_children:
                    del graph_json['nodes'][node_idx]

            # Update the edges and point to the first_child instead of other_children
            for edge_idx, edge in enumerate(graph_json['links']):
                if edge['source'] in other_children:
                    graph_json['links'][edge_idx]['source'] = first_child

    # Update the `store` nodes to point to the nodes where the new data is stored
    for node in graph_json.get('nodes'):
        if node.get('text') == 'store':
            # TODO: For later experiment: remove edges with position 1 from store nodes
            for edge_idx, edge in enumerate(graph_json.get('links')):
                if (edge.get('target') == node.get('id')) and (edge.get('flow') == 1) and (edge.get('position') == 1):
                    new_edge = {
                        'flow': 1,
                        'key': 0,
                        'position': 0,
                        'source': node.get('id'),
                        'target': edge.get('source')
                    }
                    graph_json['links'].append(new_edge)

    # Convert to Pytorch Geometric Heterogeneous Format
    all_nodes_mapping = {}
    control_nodes = []
    variable_nodes = []
    constant_nodes = []

    edge_type = {
        0: 'control',
        1: 'data',
        2: 'call'
    }
    node_type = {
        0: 'control',
        1: 'variable',
        2: 'constant'
    }

    for node in graph_json.get('nodes'):
        if node['type'] == 0:
            all_nodes_mapping[node['id']] = (len(control_nodes), node['type'])
            node_attr = get_node_attr(node)
            control_nodes.append(node_attr)
        elif node['type'] == 1:
            all_nodes_mapping[node['id']] = (len(variable_nodes), node['type'])
            node_attr = get_node_attr(node)
            node_attr[1] = node_attr[1][0].split(' ')[1]
            variable_nodes.append(node_attr)

        elif node['type'] == 2:
            all_nodes_mapping[node['id']] = (len(constant_nodes), node['type'])
            node_attr = get_node_attr(node)
            node_attr[1] = node_attr[1][0].split(' ')[1]
            constant_nodes.append(node_attr)

    source_code_lines = ""
    print('file to open ', source_file_path)
    with open(source_file_path, 'r') as loop_file:
        loop_lines = loop_file.readlines()
        for line in loop_lines:
            # loop_lines_data.append(line)
            source_code_lines += line + " "


    node_features = {
        'control': control_nodes,
        'variable': variable_nodes,
        'constant': constant_nodes
    }

    edges = {
        'control_control_control': [],
        'control_call_control': [],
        'control_data_variable': [],
        'variable_data_control': [],
        'constant_data_control': [],
        'control_data_constant': []
    }
    edges_attr = {
        'control_control_control': [],
        'control_call_control': [],
        'control_data_variable': [],
        'variable_data_control': [],
        'constant_data_control': [],
        'control_data_constant': []

    }

    file_info = {
        'code_text': source_code_lines,
        'label': label_count
    }

    for edge in graph_json.get('links'):
        flow_type = edge_type[edge.get('flow')]
        edge_pos = edge.get('position')
        source_node_idx = edge.get('source')
        source_node_type = node_type[all_nodes_mapping[source_node_idx][1]]
        source_node_idx = all_nodes_mapping[source_node_idx][0]
        target_node_idx = edge.get('target')
        target_node_type = node_type[all_nodes_mapping[target_node_idx][1]]
        target_node_idx = all_nodes_mapping[target_node_idx][0]
        edge_triple = "_".join([source_node_type, flow_type, target_node_type])
        edges[edge_triple].append([source_node_idx, target_node_idx])
        edges_attr[edge_triple].append(edge_pos)

    return node_features, edges, edges_attr, file_info

# feature file read starts here

feat_count = 0
feature_file = open("../FeatureMap/feature_map_file_pg_plus_text_all_dev_map_with_nvidia.txt", 'r')
feature_lines = feature_file.readlines()
feature_map = {}
for feature_line in feature_lines:
    feature_key = feature_line.split(",")[0]
    if feature_key not in feature_map:
        feature_key = feature_key
        feature_map[feature_key] = feat_count
        feat_count = feat_count + 1
embeds = nn.Embedding(feat_count, 3)  # feat_count words in vocab, 3 dimensional embeddings

# ends here

import os

def find_subdirectories(directory):
    subdirectories = []
    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)
        if os.path.isdir(full_path):
            subdirectories.append(full_path)

    return subdirectories

# Usage example
label_count = 0
directory_paths = ["/Users/quaziishtiaquemahmud/Desktop/Start/RA/Darmstadt/Simd-Target-Combined/SIMD-Target-Combined/CPU/", "/Users/quaziishtiaquemahmud/Desktop/Start/RA/Darmstadt/Simd-Target-Combined/SIMD-Target-Combined/GPU/"]  # Replace with your directory path

graph_list = []
graph_count = 0


for directory in tqdm(directory_paths):
    sub_directories = find_subdirectories(directory)
    for sub_directory in sub_directories:
        files = listdir(sub_directory)
        consider = 0
        for file in tqdm(files):
            if file.endswith('.ll'):
                full_path = sub_directory + "/" + file
                graph = get_hetero_graph(full_path, full_path, label_count)
                graph_list.append(graph)
                graph_count += 1
    label_count += 1
print(graph_count)
torch.save(graph_list, './project_darmstadt_graph_list_simd_target_combined_base_perfograph.pt')
