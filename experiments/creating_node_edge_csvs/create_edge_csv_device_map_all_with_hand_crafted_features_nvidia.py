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
    return reduced_final_embedding

#feature file read starts here

edges_0_file = open("dgl-csv-dev-map-all-with-hand-crafted-features-nvidia/edges_0.csv", "r+")
edges_1_file = open("dgl-csv-dev-map-all-with-hand-crafted-features-nvidia/edges_1.csv", "r+")
edges_2_file = open("dgl-csv-dev-map-all-with-hand-crafted-features-nvidia/edges_2.csv", "r+")
edges_3_file = open("dgl-csv-dev-map-all-with-hand-crafted-features-nvidia/edges_3.csv", "r+")
edges_4_file = open("dgl-csv-dev-map-all-with-hand-crafted-features-nvidia/edges_4.csv", "r+")
edges_5_file = open("dgl-csv-dev-map-all-with-hand-crafted-features-nvidia/edges_5.csv", "r+")
graph_file = open("dgl-csv-dev-map-all-with-hand-crafted-features-nvidia/graphs.csv", "r+")

feat_count = 0
feature_file = open("feature_map_file_pg_plus_text_all_dev_map_with_nvidia.txt", 'r')
feature_lines = feature_file.readlines()
feature_map = {}
for feature_line in feature_lines:
    feature_key = feature_line.split(",")[0]
    if feature_key not in feature_map:
        feature_key = feature_key
        feature_map[feature_key] = feat_count
        feat_count = feat_count + 1
embeds = nn.Embedding(feat_count, 3)  # feat_count words in vocab, 3 dimensional embeddings

graphs = torch.load('graph_list_with_hand_crafted_features_nvidia.pt')

data_list = []

cnt = 0

for graph in graphs:
    data = HeteroData()
    feature_vector_control_nodes = []
    feature_vector_variable_nodes = []
    feature_vector_constant_nodes = []
    control_nodes = graph[0].get('control')
    variable_nodes = graph[0].get('variable')
    constant_nodes = graph[0].get('constant')
    # get embedding for control nodes
    for control_node in control_nodes:
        text_of_control_node = str(control_node[0])
        if text_of_control_node in feature_map:
            lookup_tensor = torch.tensor([feature_map[text_of_control_node]], dtype=torch.long)
            text_embed = embeds(lookup_tensor)
            text_embed_real_numpy = text_embed.detach().numpy()
            text_embed_list = []
            for value in text_embed_real_numpy:
                text_embed_list = value.tolist()
            feature_vector_control_nodes.append(text_embed_list)
        else:
            feature_vector_control_nodes.append([0.0, 0.0, 0.0])
    data['control'].x = torch.tensor(feature_vector_control_nodes)
    # data['control'].num_nodes = len(control_nodes)
    # data_list.append(data)
    # get embedding for variable nodes
    for variable_node in variable_nodes:
        text_of_variable_node = str(variable_node[0])
        if text_of_variable_node in feature_map:
            lookup_tensor = torch.tensor([feature_map[text_of_variable_node]], dtype=torch.long)
            text_embed = embeds(lookup_tensor)
            text_embed_real_numpy = text_embed.detach().numpy()
            text_embed_list = []
            for value in text_embed_real_numpy:
                text_embed_list = value.tolist()
            feature_vector_variable_nodes.append(text_embed_list)
        else:
            feature_vector_variable_nodes.append([0.0, 0.0, 0.0])
    data['variable'].x = torch.tensor(feature_vector_variable_nodes)
    # data['variable'].num_nodes = len(variable_nodes)
    # # data_list.append(data)
    # get embedding for constant nodes
    for constant_node in constant_nodes:
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
        feature_vector_constant_nodes.append(text_embed_list)

    data['constant'].x = torch.tensor(feature_vector_constant_nodes)
    # data['constant'].num_nodes = len(constant_nodes)
    # embedding edges
    control_control_control_edges = []
    control_call_control_edges = []
    control_data_variable_edges = []
    variable_data_control_edges = []
    constant_data_control_edges = []
    control_data_constant_edges = []
    control_control_control_edges = graph[1].get('control_control_control')
    control_call_control_edges = graph[1].get('control_call_control')
    control_data_variable_edges = graph[1].get('control_data_variable')
    variable_data_control_edges = graph[1].get('variable_data_control')
    constant_data_control_edges = graph[1].get('constant_data_control')
    control_data_constant_edges = graph[1].get('control_data_constant')
    # get embedding for control_control_control edges
    control_control_control_edges_index = []
    source_node = []
    des_node = []
    for control_control_control_edge in control_control_control_edges:
        source_node.append(control_control_control_edge[0])
        des_node.append(control_control_control_edge[1])
        edge_strin = ""
        edge_string = str(cnt) + "," + str(control_control_control_edge[0]) + "," + str(control_control_control_edge[1]) + "\n"
        edges_0_file.writelines(edge_string)
    control_control_control_edges_index.append(source_node)
    control_control_control_edges_index.append(des_node)
    data['control', 'control', 'control'].edge_index = control_control_control_edges_index
    # get embedding for control_call_control edges
    control_call_control_edges_index = []
    source_node = []
    des_node = []
    for control_call_control_edge in control_call_control_edges:
        source_node.append(control_call_control_edge[0])
        des_node.append(control_call_control_edge[1])
        edge_strin = ""
        edge_string = str(cnt) + "," + str(control_call_control_edge[0]) + "," + str(
            control_call_control_edge[1]) + "\n"
        edges_1_file.writelines(edge_string)
    control_call_control_edges_index.append(source_node)
    control_call_control_edges_index.append(des_node)
    data['control', 'call', 'control'].edge_index = control_call_control_edges_index
    # get embedding for control_data_variable edges
    control_data_variable_edges_index = []
    source_node = []
    des_node = []
    for control_data_variable_edge in control_data_variable_edges:
        source_node.append(control_data_variable_edge[0])
        des_node.append(control_data_variable_edge[1])
        edge_strin = ""
        edge_string = str(cnt) + "," + str(control_data_variable_edge[0]) + "," + str(
            control_data_variable_edge[1]) + "\n"
        edges_2_file.writelines(edge_string)
    control_data_variable_edges_index.append(source_node)
    control_data_variable_edges_index.append(des_node)
    data['control', 'data', 'variable'].edge_index = control_data_variable_edges_index
    # get embedding for variable_data_control edges
    variable_data_control_edges_index = []
    source_node = []
    des_node = []
    for variable_data_control_edge in variable_data_control_edges:
        source_node.append(variable_data_control_edge[0])
        des_node.append(variable_data_control_edge[1])
        edge_strin = ""
        edge_string = str(cnt) + "," + str(variable_data_control_edge[0]) + "," + str(
            variable_data_control_edge[1]) + "\n"
        edges_3_file.writelines(edge_string)
    variable_data_control_edges_index.append(source_node)
    variable_data_control_edges_index.append(des_node)
    data['variable', 'data', 'control'].edge_index = variable_data_control_edges_index
    # get embedding for constant_data_control edges
    constant_data_control_edges_index = []
    source_node = []
    des_node = []
    for constant_data_control_edge in constant_data_control_edges:
        source_node.append(constant_data_control_edge[0])
        des_node.append(constant_data_control_edge[1])
        edge_strin = ""
        edge_string = str(cnt) + "," + str(constant_data_control_edge[0]) + "," + str(
            constant_data_control_edge[1]) + "\n"
        edges_4_file.writelines(edge_string)
    constant_data_control_edges_index.append(source_node)
    constant_data_control_edges_index.append(des_node)
    data['constant', 'data', 'control'].edge_index = constant_data_control_edges_index
    # get embedding for control_data_constant edges
    control_data_constant_edges_index = []
    source_node = []
    des_node = []
    for control_data_constant_edge in control_data_constant_edges:
        source_node.append(control_data_constant_edge[0])
        des_node.append(control_data_constant_edge[1])
        edge_strin = ""
        edge_string = str(cnt) + "," + str(control_data_constant_edge[0]) + "," + str(
            control_data_constant_edge[1]) + "\n"
        edges_5_file.writelines(edge_string)
    control_data_constant_edges_index.append(source_node)
    control_data_constant_edges_index.append(des_node)
    data['control', 'data', 'constant'].edge_index = control_data_constant_edges_index
    # get embedding for edge_attr
    control_control_control_edges_attr = []
    control_call_control_edges_attr = []
    control_data_variable_edges_attr = []
    variable_data_control_edges_attr = []
    constant_data_control_edges_attr = []
    control_data_constant_edges_attr = []
    control_control_control_edges_attr = graph[2].get('control_control_control')
    control_call_control_edges_attr = graph[2].get('control_call_control')
    control_data_variable_edges_attr = graph[2].get('control_data_variable')
    variable_data_control_edges_attr = graph[2].get('variable_data_control')
    constant_data_control_edges_attr = graph[2].get('constant_data_control')
    control_data_constant_edges_attr = graph[2].get('control_data_constant')
    # get embedding for control_control_control edges attr
    control_control_control_edges_attr_all = []
    for control_control_control_feature in control_control_control_edges_attr:
        control_control_control_edges_attr_all.append(control_control_control_feature)
    data['control', 'control', 'control'].edge_attr = torch.tensor(control_control_control_edges_attr_all)
    # get embedding for control_call_control edges attr
    control_call_control_edges_attr_all = []
    for control_call_control_feature in control_call_control_edges_attr:
        control_call_control_edges_attr_all.append(control_call_control_feature)
    data['control', 'call', 'control'].edge_attr = torch.tensor(control_call_control_edges_attr_all)
    # get embedding for control_data_variable edges attr
    control_data_variable_edges_attr_all = []
    for control_data_variable_feature in control_data_variable_edges_attr:
        control_data_variable_edges_attr_all.append(control_data_variable_feature)
    data['control', 'data', 'variable'].edge_attr = torch.tensor(control_data_variable_edges_attr_all)
    # get embedding for variable_data_control edges attr
    variable_data_control_edges_attr_all = []
    for variable_data_control_feature in variable_data_control_edges_attr:
        variable_data_control_edges_attr_all.append(variable_data_control_feature)
    data['variable', 'data', 'control'].edge_attr = torch.tensor(variable_data_control_edges_attr_all)
    # get embedding for constant_data_control edges attr
    constant_data_control_edges_attr_all = []
    for constant_data_control_feature in constant_data_control_edges_attr:
        constant_data_control_edges_attr_all.append(constant_data_control_feature)
    data['constant', 'data', 'control'].edge_attr = torch.tensor(constant_data_control_edges_attr_all)
    # get embedding for control_data_constant edges attr
    control_data_constant_edges_attr_all = []
    for control_data_constant_feature in control_data_constant_edges_attr:
        control_data_constant_edges_attr_all.append(control_data_constant_feature)
    data['control', 'data', 'constant'].edge_attr = torch.tensor(control_data_constant_edges_attr_all)
    if cnt < 395:
        data['y'] = torch.tensor([0])
    else:
        data['y'] = torch.tensor([1])
    data_list.append(data)

    cnt += 1
    # if cnt == 3:
    #     break
print(cnt)

i = 0

for graph in graphs:
    graph_string = ""
    if (i < 395):
        # feature_string = graph[4].get('hand_crafted_features')
        graph_string = str(i) + "," + "0" + "\n"
    elif (i >= 395):
        # feature_string = graph[4].get('hand_crafted_features')
        graph_string = str(i) + "," + "1" + "\n"
    graph_file.writelines(graph_string)
    i = i + 1
#
# for i in range(0, cnt):
#     if (i < 395):
#         graph_string = str(i) + "," + "0" + "\n"
#     elif (i >= 395):
#         graph_string = str(i) + "," + "1" + "\n"
#     graph_file.writelines(graph_string)
# torch.save(data_list,'data-list.pt')

# for data in data_list:
#     print(data)

# mydata2 = torch.load('data-list.pt')
#
# print('from file')
#
# for data in mydata2:
#     print(data)


# train_loader = DataLoader(data_list, batch_size=32, shuffle=True)

# for batched_graph in train_loader:
#     print(batched_graph['control','data','variable'].edge_index)
#     print('len(batched_graph[edge_index])', len(batched_graph['control','data','variable'].edge_index))
#     break
# it = iter(train_loader)
# batch = next(it)
# print(batch)
# len1 = len(data_list[0]['constant'].x[0])
# print(len1)
# print(data_list[0]['constant'].x)
# len2 = len(data_list[0]['control','control','control'].edge_index)
# print(len2)
# print(data_list[0]['control','control','control'])
# len2 = len(data_list[0]['control','call','control'].edge_index)
# print(len2)
# print(data_list[0]['control','call','control'])
# len2 = len(data_list[0]['control','data','variable'].edge_index)
# print(len2)
# print(data_list[0]['control','data','variable'])
# len2 = len(data_list[0]['variable','data','control'].edge_index)
# print(len2)
# print(data_list[0]['variable','data','control'])
# len2 = len(data_list[0]['constant','data','control'].edge_index)
# print(len2)
# print(data_list[0]['constant','data','control'])
# len2 = len(data_list[0]['control','data','constant'].edge_index)
# print(len2)
# print(data_list[0]['control','data','constant'])
