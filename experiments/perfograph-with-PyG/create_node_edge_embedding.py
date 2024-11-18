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
from torch_geometric.data import HeteroData
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import re
from torch_geometric.nn import RGCNConv
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear
from torch.nn import LayerNorm
from tqdm import tqdm

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

def get_digit_emb_of_number(token, feature_map):
    digits = []
    digits_pos = []
    digit_embedding_vector = []
    digit_pos_vector = []
    reduced_final_embedding = []
    if is_numeric(token) == True:
        digits, digits_pos = get_digit_and_pos(token)
        for digit in digits:
            if digit in feature_map:
                lookup_tensor = torch.tensor([feature_map[digit]], dtype=torch.long)
                node_embed = embeds(lookup_tensor)
                node_embed_real_numpy = node_embed.detach().numpy()
                node_embed_list = []
                for value in node_embed_real_numpy:
                    node_embed_list = value.tolist()
                digit_embedding_vector.append(node_embed_list)
            else:
                digit_embedding_vector.append([0.0, 0.0, 0.0])
        for digit_pos in digits_pos:
            if digit_pos in feature_map:
                lookup_tensor = torch.tensor([feature_map[digit_pos]], dtype=torch.long)
                node_embed = embeds(lookup_tensor)
                node_embed_real_numpy = node_embed.detach().numpy()
                node_embed_list = []
                for value in node_embed_real_numpy:
                    node_embed_list = value.tolist()
                digit_pos_vector.append(node_embed_list)
            else:
                digit_pos_vector.append([0.0, 0.0, 0.0])

        final_embedding_vector = []
        final_embedding_list_of_np_arrays = list(
            (np.array(digit_embedding_vector) + np.array(digit_pos_vector)))

        for embeddin in final_embedding_list_of_np_arrays:
            final_embedding_vector.append(list(embeddin))

        final_embedding_vector_np_array = np.array(final_embedding_vector)
        final_embedding_vector_np_array_sum = np.sum(final_embedding_vector_np_array, axis=0)
        reduced_final_embedding = list(final_embedding_vector_np_array_sum)

        max_of_reduced_final_embedding = max(reduced_final_embedding, key=abs)

        for i3 in range(len(reduced_final_embedding)):
            reduced_final_embedding[i3] = reduced_final_embedding[i3] / (abs(max_of_reduced_final_embedding) + 1)
    else:
        reduced_final_embedding = [0.0, 0.0, 0.0]
    return reduced_final_embedding

#feature file read starts here

# nodes_0_file = open("dgl-csv-XSBench/nodes_0.csv", "r+")
# nodes_1_file = open("dgl-csv-XSBench/nodes_1.csv", "r+")
# nodes_2_file = open("dgl-csv-XSBench/nodes_2.csv", "r+")


feat_count = 0
feature_file = open("./feature_map_file_pg_plus_text_all_dev_map_with_nvidia.txt", 'r')
feature_lines = feature_file.readlines()
feature_map = {}
for feature_line in feature_lines:
    feature_key = feature_line.split(",")[0]
    if feature_key not in feature_map:
        feature_key = feature_key
        feature_map[feature_key] = feat_count
        feat_count = feat_count + 1
embeds = nn.Embedding(feat_count, 3)  # feat_count words in vocab, 3 dimensional embeddings

graphs = torch.load('base_perfograph.pt')
# print(graphs[0])

data_list = []

cnt = 0

for graph in tqdm(graphs):
    control_nodes = graph[0].get('control')
    variable_nodes = graph[0].get('variable')
    constant_nodes = graph[0].get('constant')

    len_control_nodes = len(control_nodes)
    len_variable_nodes = len(variable_nodes)
    len_constant_nodes = len(constant_nodes)

    data = HeteroData()
    data['control'].x = []
    data['variable'].x = []
    data['constant'].x = []

    control_node_counter = 0
    variable_node_counter = 0
    constant_node_counter = 0
    # get embedding for control nodes
    all_control_node_embedding_array = []
    for control_node in control_nodes:
        full_text_of_control_node = str(control_node[1])
        # print('full text : ',full_text_of_control_node)
        node_byte = str(full_text_of_control_node)
        node_str = ""
        for i2 in range(2, len(node_byte) - 2):
            node_str += node_byte.__getitem__(i2)
        # if len(node_str) > max_control:
        #     max_control = len(node_str)
        # print('node_str_control', node_str)
        # Digit embedding of full_text
        tokens = []
        tokens = node_str.split(' ')
        node_embed_str = []
        node_embed_str_flag = 0
        prev_token_length = len(tokens)
        if len(tokens) < 40:
            for j in range(40 - prev_token_length):
                tokens.append('QQ')
        elif len(tokens) > 40:
            for j in range(len(tokens) - 40):
                tokens.pop()
        for j in range(len(tokens)):
            # digit embedding starts here
            digits = []
            digits_pos = []
            digit_embedding_vector = []
            digit_pos_vector = []
            if is_numeric(tokens[j]) == True:

                reduced_final_embedding = get_digit_emb_of_number(tokens[j], feature_map)

                if j == 0:
                    node_embed_str.append(reduced_final_embedding[0])
                    node_embed_str.append(reduced_final_embedding[1])
                    node_embed_str.append(reduced_final_embedding[2])
                elif j == len(tokens) - 1:
                    node_embed_str.append(reduced_final_embedding[0])
                    node_embed_str.append(reduced_final_embedding[1])
                    node_embed_str.append(reduced_final_embedding[2])
                else:
                    node_embed_str.append(reduced_final_embedding[0])
                    node_embed_str.append(reduced_final_embedding[1])
                    node_embed_str.append(reduced_final_embedding[2])
            elif tokens[j] in feature_map:
                lookup_tensor = torch.tensor([feature_map[tokens[j]]], dtype=torch.long)
                node_embed = embeds(lookup_tensor)
                node_embed_real_numpy = node_embed.detach().numpy()
                node_embed_list = []
                for value in node_embed_real_numpy:
                    node_embed_list = value.tolist()
                if j == 0:
                    node_embed_str.append(node_embed_list[0])
                    node_embed_str.append(node_embed_list[1])
                    node_embed_str.append(node_embed_list[2])
                elif j == len(tokens) - 1:
                    node_embed_str.append(node_embed_list[0])
                    node_embed_str.append(node_embed_list[1])
                    node_embed_str.append(node_embed_list[2])
                else:
                    node_embed_str.append(node_embed_list[0])
                    node_embed_str.append(node_embed_list[1])
                    node_embed_str.append(node_embed_list[2])
            else:
                if j == 0:
                    node_embed_str.append(0.0)
                    node_embed_str.append(0.0)
                    node_embed_str.append(0.0)
                elif j == len(tokens) - 1:
                    node_embed_str.append(0.0)
                    node_embed_str.append(0.0)
                    node_embed_str.append(0.0)
                else:
                    node_embed_str.append(0.0)
                    node_embed_str.append(0.0)
                    node_embed_str.append(0.0)

        # node_file_string = str(cnt) + "," + str(control_node_counter) + "," + node_embed_str + "\n"
        all_control_node_embedding_array.append(node_embed_str)
        control_node_counter = control_node_counter + 1
    data['control'].x = torch.tensor(all_control_node_embedding_array, dtype=torch.float)

    # ends here
    # get embedding for variable nodes
    all_variable_node_embedding_array = []
    for variable_node in variable_nodes:
        text_of_variable_node = str(variable_node[0])
        digits = []
        digits_pos = []
        digit_embedding_vector = []
        digit_pos_vector = []
        node_embed_str = []
        if is_numeric(text_of_variable_node) == True:

            reduced_final_embedding = get_digit_emb_of_number(text_of_variable_node, feature_map)

            node_embed_str.append(reduced_final_embedding[0])
            node_embed_str.append(reduced_final_embedding[1])
            node_embed_str.append(reduced_final_embedding[2])

            #   padding
            for padd in range(0,116):
                node_embed_str.append(0.0)
            node_embed_str.append(0.0)
            #   ends here


        elif text_of_variable_node in feature_map:
            lookup_tensor = torch.tensor([feature_map[text_of_variable_node]], dtype=torch.long)
            node_embed = embeds(lookup_tensor)
            node_embed_real_numpy = node_embed.detach().numpy()
            node_embed_list = []
            for value in node_embed_real_numpy:
                node_embed_list = value.tolist()

            node_embed_str.append(node_embed_list[0])
            node_embed_str.append(node_embed_list[1])
            node_embed_str.append(node_embed_list[2])

            #   padding
            for padd in range(0, 116):
                node_embed_str.append(0.0)
            node_embed_str.append(0.0)
            #   ends here

        else:
            node_embed_str.append(0.0)
            node_embed_str.append(0.0)
            node_embed_str.append(0.0)
            #   padding
            for padd in range(0, 116):
                node_embed_str.append(0.0)
            node_embed_str.append(0.0)
            #   ends here

        # data['variable'].x.append([torch.tensor(node_embed_str)])
        # data['variable'].x.append([torch.tensor([node_embed_str])])
        all_variable_node_embedding_array.append(node_embed_str)
        variable_node_counter = variable_node_counter + 1
    data['variable'].x = torch.tensor(all_variable_node_embedding_array, dtype=torch.float)

    all_constant_node_embedding_array = []
    # get embedding for constant nodes
    for constant_node in constant_nodes:
        node_embed_str = []
        text_embed_list = []
        text_type_of_constant_node = str(constant_node[0])
        if text_type_of_constant_node in feature_map:
            lookup_tensor = torch.tensor([feature_map[text_type_of_constant_node]], dtype=torch.long)
            text_embed = embeds(lookup_tensor)
            text_embed_real_numpy = text_embed.detach().numpy()
            for value in text_embed_real_numpy:
                text_embed_list = value.tolist()
        else:
            text_embed_list = [0.0, 0.0, 0.0]
        text_value_of_constant_node = str(constant_node[1])
        digit_emb_vec_of_text_value = get_digit_emb_of_number(text_value_of_constant_node, feature_map)
        for component in digit_emb_vec_of_text_value:
            text_embed_list.append(component)
        for i in range(0, 6-len(text_embed_list)):
            text_embed_list.append(0.0)

        node_embed_str.append(text_embed_list[0])
        node_embed_str.append(text_embed_list[1])
        node_embed_str.append(text_embed_list[2])
        node_embed_str.append(text_embed_list[3])
        node_embed_str.append(text_embed_list[4])
        node_embed_str.append(text_embed_list[5])


        #   padding
        for padd in range(0, 113):
            node_embed_str.append(0.0)
        node_embed_str.append(0.0)
        #   ends here
        # data['constant'].x.append([torch.tensor(node_embed_str)])
        # data['constant'].x.append([torch.tensor([node_embed_str])])
        all_constant_node_embedding_array.append(node_embed_str)
        constant_node_counter = constant_node_counter + 1

    data['constant'].x = torch.tensor(all_constant_node_embedding_array, dtype=torch.float)


    control_control_control_edges = graph[1].get('control_control_control')
    control_call_control_edges = graph[1].get('control_call_control')
    control_data_variable_edges = graph[1].get('control_data_variable')
    variable_data_control_edges = graph[1].get('variable_data_control')
    constant_data_control_edges = graph[1].get('constant_data_control')
    control_data_constant_edges = graph[1].get('control_data_constant')

    # get embedding for control_control_control edges
    control_control_control_edges_index = []
    control_control_control_edges_attr = []
    control_control_control_indv_edge_attr = []

    # create the individual one-hot-encoding vector
    for i in range(120):
        if i == 0:
            control_control_control_indv_edge_attr.append(1)
        else:
            control_control_control_indv_edge_attr.append(0)

    source_node = []
    des_node = []
    for control_control_control_edge in control_control_control_edges:
        if control_control_control_edge[0] < len_control_nodes and control_control_control_edge[1] < len_control_nodes:
            source_node.append(control_control_control_edge[0])
            des_node.append(control_control_control_edge[1])
            control_control_control_edges_attr.append(control_control_control_indv_edge_attr)

    control_control_control_edges_index.append(source_node)
    control_control_control_edges_index.append(des_node)
    data['control', 'control', 'control'].edge_index = torch.tensor(control_control_control_edges_index, dtype=torch.int64)
    data['control', 'control', 'control'].edge_attr = torch.tensor(control_control_control_edges_attr,
                                                                    dtype=torch.int64)

    # get embedding for control_call_control edges
    control_call_control_edges_index = []
    control_call_control_edges_attr = []
    control_call_control_indv_edge_attr = []

    # create the individual one-hot-encoding vector
    for i in range(120):
        if i == 1:
            control_call_control_indv_edge_attr.append(1)
        else:
            control_call_control_indv_edge_attr.append(0)

    source_node = []
    des_node = []
    for control_call_control_edge in control_call_control_edges:
        if control_call_control_edge[0] < len_control_nodes and control_call_control_edge[1] < len_control_nodes:
            source_node.append(control_call_control_edge[0])
            des_node.append(control_call_control_edge[1])
            control_call_control_edges_attr.append(control_call_control_indv_edge_attr)

    control_call_control_edges_index.append(source_node)
    control_call_control_edges_index.append(des_node)
    data['control', 'call', 'control'].edge_index = torch.tensor(control_call_control_edges_index, dtype=torch.int64)
    data['control', 'call', 'control'].edge_attr = torch.tensor(control_call_control_edges_attr, dtype=torch.int64)

    # get embedding for control_data_variable edges
    control_data_variable_edges_index = []
    control_data_variable_edges_attr = []
    control_data_variable_indv_edge_attr = []

    # create the individual one-hot-encoding vector
    for i in range(120):
        if i == 2:
            control_data_variable_indv_edge_attr.append(1)
        else:
            control_data_variable_indv_edge_attr.append(0)

    source_node = []
    des_node = []
    for control_data_variable_edge in control_data_variable_edges:
        if control_data_variable_edge[0] < len_control_nodes and control_data_variable_edge[1] < len_variable_nodes:
            source_node.append(control_data_variable_edge[0])
            des_node.append(control_data_variable_edge[1])
            control_data_variable_edges_attr.append(control_data_variable_indv_edge_attr)

    control_data_variable_edges_index.append(source_node)
    control_data_variable_edges_index.append(des_node)
    data['control', 'data', 'variable'].edge_index = torch.tensor(control_data_variable_edges_index, dtype=torch.int64)
    data['control', 'data', 'variable'].edge_attr = torch.tensor(control_data_variable_edges_attr, dtype=torch.int64)


    # get embedding for variable_data_control edges
    variable_data_control_edges_index = []
    variable_data_control_edges_attr = []
    variable_data_control_indv_edge_attr = []

    # create the individual one-hot-encoding vector
    for i in range(120):
        if i == 3:
            variable_data_control_indv_edge_attr.append(1)
        else:
            variable_data_control_indv_edge_attr.append(0)

    source_node = []
    des_node = []
    for variable_data_control_edge in variable_data_control_edges:
        if variable_data_control_edge[0] < len_variable_nodes and variable_data_control_edge[1] < len_control_nodes:
            source_node.append(variable_data_control_edge[0])
            des_node.append(variable_data_control_edge[1])
            variable_data_control_edges_attr.append(variable_data_control_indv_edge_attr)

    variable_data_control_edges_index.append(source_node)
    variable_data_control_edges_index.append(des_node)
    data['variable', 'data', 'control'].edge_index = torch.tensor(variable_data_control_edges_index, dtype=torch.int64)
    data['variable', 'data', 'control'].edge_attr = torch.tensor(variable_data_control_edges_attr, dtype=torch.int64)



    # get embedding for constant_data_control edges
    constant_data_control_edges_index = []
    constant_data_control_edges_attr = []
    constant_data_control_indv_edge_attr = []

    # create the individual one-hot-encoding vector
    for i in range(120):
        if i == 4:
            constant_data_control_indv_edge_attr.append(1)
        else:
            constant_data_control_indv_edge_attr.append(0)

    source_node = []
    des_node = []
    for constant_data_control_edge in constant_data_control_edges:
        if constant_data_control_edge[0] < len_constant_nodes and constant_data_control_edge[1] < len_control_nodes:
            source_node.append(constant_data_control_edge[0])
            des_node.append(constant_data_control_edge[1])
            constant_data_control_edges_attr.append(constant_data_control_indv_edge_attr)

    constant_data_control_edges_index.append(source_node)
    constant_data_control_edges_index.append(des_node)
    data['constant', 'data', 'control'].edge_index = torch.tensor(constant_data_control_edges_index, dtype=torch.int64)
    data['constant', 'data', 'control'].edge_attr = torch.tensor(constant_data_control_edges_attr, dtype=torch.int64)

    # get embedding for control_data_constant edges
    control_data_constant_edges_index = []
    control_data_constant_edges_attr = []
    control_data_constant_indv_edge_attr = []

    # create the individual one-hot-encoding vector
    for i in range(120):
        if i == 5:
            control_data_constant_indv_edge_attr.append(1)
        else:
            control_data_constant_indv_edge_attr.append(0)

    source_node = []
    des_node = []
    for control_data_constant_edge in control_data_constant_edges:
        if control_data_constant_edge[0] < len_control_nodes and control_data_constant_edge[1] < len_constant_nodes:
            source_node.append(control_data_constant_edge[0])
            des_node.append(control_data_constant_edge[1])
            control_data_constant_edges_attr.append(control_data_constant_indv_edge_attr)

    control_data_constant_edges_index.append(source_node)
    control_data_constant_edges_index.append(des_node)
    data['control', 'data', 'constant'].edge_index = torch.tensor(control_data_constant_edges_index, dtype=torch.int64)
    data['control', 'data', 'constant'].edge_attr = torch.tensor(control_data_constant_edges_attr, dtype=torch.int64)


    data.code_text = graph[3].get('code_text')
    data.label = graph[3].get('label')

    cnt += 1
    data_list.append(data)
print(cnt)

torch.save(data_list, './datalist-base-perfograph.pt')
