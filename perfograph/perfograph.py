import programl as pg
import re
import tqdm
import traceback
from copy import copy
import pygraphviz as pgv
import json

# Node type 3 => data arrays
# Node type 4 => data vectors
# Node type 5 => constant arrays
# Node type 6 => constant vectors

class Perfograph:
    def __init__(self, file: str, llvm_version: str = '10', with_vectors=True, disable_progress_bar=False):
        try:
            with open(file) as f:
                file_content_str = f.read()
                if file[-3:] == '.ll':
                    # this is a llvm IR file
                    self.programl_graph = pg.from_llvm_ir(file_content_str, version=llvm_version)
                elif file[-4:] == '.cpp':
                    self.programl_graph = pg.from_cpp(file_content_str)
                else:
                    raise ValueError('Only LLVM IR (*.ll) or CPP (*.cpp) files are accepted.')
        except Exception as e:
            print(e)
            traceback.print_exc()
            raise
        self.graph_json = _get_hetero_graph(self.programl_graph, with_vectors, disable_progress_bar)

def from_file(file: str, llvm_version: str = '10', with_vectors=True, disable_progress_bar=False):
    return Perfograph(file, llvm_version, with_vectors, disable_progress_bar)

def to_dot(G, file_name: str):
    _visualize(G.graph_json, file_name)
        
def to_json(G, file_name: str= ''):
    if file_name[-5:] == '.json':
        with open(file_name, 'w') as f:
            json.dump(G.graph_json, f, indent=4)
    elif file_name == '':
        return G.graph_json
    else:
        raise ValueError('Only JSON (*.json) files are accepted.')
        
def _get_hetero_graph(programl_graph, with_vectors, disable_progress_bar):
    def get_node_attr(node):
        text = node.get('text')
        full_text = node.get('features')
        if full_text is not None:
            full_text = full_text.get('full_text')
        return [text, full_text]


    graph_json = pg.to_json(programl_graph)

    # UPDATING THE EDGES
    for node in graph_json.get('nodes'):
        node_text = node.get('text')
        # We want to have one node as the child of `alloca`
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

    if with_vectors:
        # new nodes and edges to support vector and arrays
        new_nodes = []
        new_edges = []
        last_node_id = graph_json['nodes'][-1]['id']
        array_vector_first_last_node_mapping = {}
        idx_of_nodes_to_delete = set()
        for idx, node in tqdm.tqdm(enumerate(graph_json.get('nodes')), disable=disable_progress_bar):
            # This is an array node with type data
            if ' x ' in node["text"] and '[' in node['text'] and node['type'] == 1:
                text = node['text']
                full_text = node['features']['full_text'][0]
                all_opening_brackets = [x.start() for x in re.finditer('\[', text)]
                all_closing_brackets = [x.start() for x in re.finditer(']', text)]
                all_spaces = [x.start() for x in re.finditer('\ ', text)]
                array_type = text[all_spaces[-1]+1:all_closing_brackets[0]]
                # Keep track of first node (inner array) and last node (outer array)
                # All the edges pointing to ProGraML array node will point to the first (inner array) child
                # All edge with ProGraML array node as source node, will have the last node (outer array) as the source node
                # '[3 x [4 x i32]]* %3' ===> ([4 x i32]) -> ([3 x [4 x i32]]* %3)
                # loops starts from the most inner arry
                # Keep track of first and last node for mapping the edge later
                array_vector_first_last_node_mapping[node['id']] = {}

                # connecting childs together
                last_child_id = None
                for i in range(len(all_opening_brackets)):
                    # increment the last node_id by 1 to have a new id for the new node
                    last_node_id += 1
                    # We keep the id of the first child as the same id of the node in ProGraML
                    new_text = text[all_opening_brackets[-(i + 1)]:all_closing_brackets[0] + 1]
                    #last node will have everything as the node in ProGraML
                    if i == len(all_opening_brackets) - 1:
                        new_full_text = full_text
                        new_text = text
                        # Keep track of last node for mapping the edge later
                        array_vector_first_last_node_mapping[node['id']]['last'] = last_node_id
                        idx_of_nodes_to_delete.add(idx)
                    else:
                        new_full_text = new_text
                    array_length = new_text.split(' x ')[0].replace('[', '')
                    #Keep track of first and last node for mapping the edge later

                    new_node = {
                        'block': node['block'],
                        'features': {'full_text': new_full_text},
                        'id': last_node_id,
                        'text': new_text,
                        'array_length': array_length,
                        'array_type': array_type,
                        'type': 3
                    }
                    new_nodes.append(new_node)

                    if last_child_id is not None:
                        new_edge = {
                            'flow': 1,
                            'key': 0,
                            'position': 0,
                            'source': last_child_id,
                            'target': last_node_id

                        }
                        new_edges.append(new_edge)
                    last_child_id = last_node_id

                    if i == 0:
                        # Keep track of first node for mapping the edge later
                        array_vector_first_last_node_mapping[node['id']]['first'] = last_node_id
                        idx_of_nodes_to_delete.add(idx)

            if ' x ' in node["text"] and '<' in node['text'] and node['type'] == 1:
                # This is an array node
                text = node['text']
                full_text = node['features']['full_text'][0]
                all_opening_brackets = [x.start() for x in re.finditer('<', text)]
                all_closing_brackets = [x.start() for x in re.finditer('>', text)]
                all_spaces = [x.start() for x in re.finditer('\ ', text)]
                array_type = text[all_spaces[-1] + 1:all_closing_brackets[0]]
                # Keep track of first node (inner array) and last node (outer array)
                # All the edges pointing to ProGraML array node will point to the first (inner array) child
                # All edge with ProGraML array node as source node, will have the last node (outer array) as the source node
                # '[3 x [4 x i32]]* %3' ===> ([4 x i32]) -> ([3 x [4 x i32]]* %3)
                # loops starts from the most inner arry
                # Keep track of first and last node for mapping the edge later
                array_vector_first_last_node_mapping[node['id']] = {}

                # connecting childs together
                last_child_id = None
                for i in range(len(all_opening_brackets)):
                    # increment the last node_id by 1 to have a new id for the new node
                    last_node_id += 1
                    # We keep the id of the first child as the same id of the node in ProGraML
                    new_text = text[all_opening_brackets[-(i + 1)]:all_closing_brackets[0] + 1]
                    # last node will have everything as the node in ProGraML
                    if i == len(all_opening_brackets) - 1:
                        new_full_text = full_text
                        new_text = text
                        # Keep track of last node for mapping the edge later
                        array_vector_first_last_node_mapping[node['id']]['last'] = last_node_id
                    else:
                        new_full_text = new_text
                    array_length = new_text.split(' x ')[0].replace('<', '')
                    # Keep track of first and last node for mapping the edge later

                    new_node = {
                        'block': node['block'],
                        'features': {'full_text': new_full_text},
                        'id': last_node_id,
                        'text': new_text,
                        'array_length': array_length,
                        'array_type': array_type,
                        'type': 4
                    }
                    new_nodes.append(new_node)

                    if last_child_id is not None:
                        new_edge = {
                            'flow': 1,
                            'key': 0,
                            'position': 0,
                            'source': last_child_id,
                            'target': last_node_id

                        }
                        new_edges.append(new_edge)
                    last_child_id = last_node_id

                    if i == 0:
                        # Keep track of first node for mapping the edge later
                        array_vector_first_last_node_mapping[node['id']]['first'] = last_node_id

            if ' x ' in node["text"] and '[' in node['text'] and node['type'] == 2:
                # This is an array node
                text = node['text']
                full_text = node['features']['full_text'][0]
                all_opening_brackets = [x.start() for x in re.finditer('\[', text)]
                all_closing_brackets = [x.start() for x in re.finditer(']', text)]
                all_spaces = [x.start() for x in re.finditer('\ ', text)]
                array_type = text[all_spaces[-1]+1:all_closing_brackets[0]]
                node_chain = []
                # Keep track of first node (inner array) and last node (outer array)
                # All the edges pointing to ProGraML array node will point to the first (inner array) child
                # All edge with ProGraML array node as source node, will have the last node (outer array) as the source node
                # '[3 x [4 x i32]]* %3' ===> ([4 x i32]) -> ([3 x [4 x i32]]* %3)
                # loops starts from the most inner arry
                # Keep track of first and last node for mapping the edge later
                array_vector_first_last_node_mapping[node['id']] = {}

                # connecting childs together
                last_child_id = None
                for i in range(len(all_opening_brackets)):
                    # increment the last node_id by 1 to have a new id for the new node
                    last_node_id += 1
                    # We keep the id of the first child as the same id of the node in ProGraML
                    new_text = text[all_opening_brackets[-(i + 1)]:all_closing_brackets[0] + 1]
                    #last node will have everything as the node in ProGraML
                    if i == len(all_opening_brackets) - 1:
                        new_full_text = full_text
                        new_text = text
                        # Keep track of last node for mapping the edge later
                        array_vector_first_last_node_mapping[node['id']]['last'] = last_node_id
                    else:
                        new_full_text = new_text
                    array_length = new_text.split(' x ')[0].replace('[', '')
                    #Keep track of first and last node for mapping the edge later

                    new_node = {
                        'block': node['block'],
                        'features': {'full_text': new_full_text},
                        'id': last_node_id,
                        'text': new_text,
                        'array_length': array_length,
                        'array_type': array_type,
                        'type': 5
                    }
                    new_nodes.append(new_node)

                    if last_child_id is not None:
                        new_edge = {
                            'flow': 1,
                            'key': 0,
                            'position': 0,
                            'source': last_child_id,
                            'target': last_node_id

                        }
                        new_edges.append(new_edge)
                    last_child_id = last_node_id

                    if i == 0:
                        # Keep track of first node for mapping the edge later
                        array_vector_first_last_node_mapping[node['id']]['first'] = last_node_id

            if ' x ' in node["text"] and '<' in node['text'] and node['type'] == 2:
                # This is an array node
                text = node['text']
                full_text = node['features']['full_text'][0]
                all_opening_brackets = [x.start() for x in re.finditer('<', text)]
                all_closing_brackets = [x.start() for x in re.finditer('>', text)]
                all_spaces = [x.start() for x in re.finditer('\ ', text)]
                array_type = text[all_spaces[-1] + 1:all_closing_brackets[0]]
                # Keep track of first node (inner array) and last node (outer array)
                # All the edges pointing to ProGraML array node will point to the first (inner array) child
                # All edge with ProGraML array node as source node, will have the last node (outer array) as the source node
                # '[3 x [4 x i32]]* %3' ===> ([4 x i32]) -> ([3 x [4 x i32]]* %3)
                # loops starts from the most inner arry
                # Keep track of first and last node for mapping the edge later
                array_vector_first_last_node_mapping[node['id']] = {}

                # connecting childs together
                last_child_id = None
                for i in range(len(all_opening_brackets)):
                    # increment the last node_id by 1 to have a new id for the new node
                    last_node_id += 1
                    # We keep the id of the first child as the same id of the node in ProGraML
                    new_text = text[all_opening_brackets[-(i + 1)]:all_closing_brackets[0] + 1]
                    # last node will have everything as the node in ProGraML
                    if i == len(all_opening_brackets) - 1:
                        new_full_text = full_text
                        new_text = text
                        # Keep track of last node for mapping the edge later
                        array_vector_first_last_node_mapping[node['id']]['last'] = last_node_id
                    else:
                        new_full_text = new_text
                    array_length = new_text.split(' x ')[0].replace('<', '')
                    # Keep track of first and last node for mapping the edge later

                    new_node = {
                        'block': node['block'],
                        'features': {'full_text': new_full_text},
                        'id': last_node_id,
                        'text': new_text,
                        'array_length': array_length,
                        'array_type': array_type,
                        'type': 6
                    }
                    new_nodes.append(new_node)

                    if last_child_id is not None:
                        new_edge = {
                            'flow': 1,
                            'key': 0,
                            'position': 0,
                            'source': last_child_id,
                            'target': last_node_id

                        }
                        new_edges.append(new_edge)
                    last_child_id = last_node_id

                    if i == 0:
                        # Keep track of first node for mapping the edge later
                        array_vector_first_last_node_mapping[node['id']]['first'] = last_node_id

            # Update the `store` nodes to point to the nodes where the new data is stored
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

        # Updating edges to support Vectors and Arrays
        for idx, link in tqdm.tqdm(enumerate(graph_json['links']), disable=False):
            if link['source'] in array_vector_first_last_node_mapping:
                graph_json['links'][idx]['source'] = array_vector_first_last_node_mapping[link['source']]['last']
            elif link['target'] in array_vector_first_last_node_mapping:
                graph_json['links'][idx]['target'] = array_vector_first_last_node_mapping[link['target']]['first']

        # Adding new nodes
        graph_json['nodes'].extend(new_nodes)
        # Adding new edges
        graph_json['links'].extend(new_edges)

        # Reconstructing nodes, ignoring the old array/vector nodes which are now replaced by chain of nodes
        graph_json['nodes'] = [n for i, n in enumerate(graph_json['nodes']) if i not in idx_of_nodes_to_delete]

    else: # without vectors
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
    #0
    instruction_nodes = []
    #1
    variable_nodes = []
    # 2
    constant_nodes = []
    #3 Variable Array
    varray_nodes = []
    #4 Variabel Vector
    vvector_nodes = []
    #5 Constant Array
    carray_nodes = []
    #6 Constant Vector
    cvector_nodes = []

    edge_type = {
        0: 'control',
        1: 'data',
        2: 'call'
    }
    node_type = {
        0: 'instruction',
        1: 'variable',
        2: 'constant',
        3: 'varray',
        4: 'vvector',
        5: 'carray',
        6: 'cvector'
    }

    for node in graph_json.get('nodes'):
        if node['type'] == 0:
            all_nodes_mapping[node['id']] = (len(instruction_nodes), node['type'])
            node_attr = get_node_attr(node)
            instruction_nodes.append(node_attr)
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
        elif node['type'] == 3:
            all_nodes_mapping[node['id']] = (len(varray_nodes), node['type'])
            node_attr = get_node_attr(node)
            node_attr.append(node['array_length'])
            node_attr.append(node['array_type'])
            varray_nodes.append(node_attr)
        elif node['type'] == 4:
            all_nodes_mapping[node['id']] = (len(vvector_nodes), node['type'])
            node_attr = get_node_attr(node)
            node_attr.append(node['array_length'])
            node_attr.append(node['array_type'])
            vvector_nodes.append(node_attr)
        elif node['type'] == 5:
            all_nodes_mapping[node['id']] = (len(carray_nodes), node['type'])
            node_attr = get_node_attr(node)
            node_attr.append(node['array_length'])
            node_attr.append(node['array_type'])
            carray_nodes.append(node_attr)
        elif node['type'] == 6:
            all_nodes_mapping[node['id']] = (len(cvector_nodes), node['type'])
            node_attr = get_node_attr(node)
            node_attr.append(node['array_length'])
            node_attr.append(node['array_type'])
            cvector_nodes.append(node_attr)

    if with_vectors:
        node_features = {
            'instruction': instruction_nodes,
            'variable': variable_nodes,
            'constant': constant_nodes,
            'varray': varray_nodes,
            'vvector': vvector_nodes,
            'carray': carray_nodes,
            'cvector': cvector_nodes,
            }

        edges = {
            'instruction_control_instruction': [],
            'instruction_call_instruction': [],
            'instruction_data_variable': [],
            'instruction_data_varray': [],
            'instruction_data_vvector': [],
            'variable_data_instruction': [],
            'varray_data_instruction': [],
            'varray_data_varray': [],
            'vvector_data_vvector': [],
            'vvector_data_instruction': [],
            'constant_data_instruction': [],
            'carray_data_instruction': [],
            'carray_data_carray': [],
            'cvector_data_instruction': [],
            'cvector_data_cvector': [],
            'instruction_data_constant': [],
            'instruction_data_carray': [],
            'instruction_data_cconstant': []

        }
        edges_attr = {
            'instruction_control_instruction': [],
            'instruction_call_instruction': [],
            'instruction_data_variable': [],
            'instruction_data_varray': [],
            'instruction_data_vvector': [],
            'variable_data_instruction': [],
            'varray_data_instruction': [],
            'varray_data_varray': [],
            'vvector_data_vvector': [],
            'vvector_data_instruction': [],
            'constant_data_instruction': [],
            'carray_data_instruction': [],
            'carray_data_carray': [],
            'cvector_data_instruction': [],
            'cvector_data_cvector': [],
            'instruction_data_constant': [],
            'instruction_data_carray': [],
            'instruction_data_cconstant': []
        }

    else: # without vectors
        node_features = {
            'instruction': instruction_nodes,
            'variable': variable_nodes,
            'constant': constant_nodes
        }

        edges = {
            'instruction_control_instruction': [],
            'instruction_call_instruction': [],
            'instruction_data_variable': [],
            'variable_data_instruction': [],
            'constant_data_instruction': [],
            'instruction_data_constant': []
        }
        edges_attr = {
            'instruction_control_instruction': [],
            'instruction_call_instruction': [],
            'instruction_data_variable': [],
            'variable_data_instruction': [],
            'constant_data_instruction': [],
            'instruction_data_constant': []
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

    return node_features, edges, edges_attr

def _visualize(graph_json, file_name):
    graph_json = copy(graph_json)
    G = pgv.AGraph(directed=True, strict=False)
    G.graph_attr.update(fontname='Inconsolata', fontsize=20, margin=0, ranksep=0.4, nodesep=0.4)
    G.node_attr.update(fontname='Inconsolata', penwidth=2, margin=0, fontsize=20, width=1)
    G.edge_attr.update(fontname='Inconsolata', penwidth=3, fontsize=20, arrowsize=.8)

    for i in range(len(graph_json[0]['instruction'])):
        instruction = graph_json[0]['instruction'][i]
        G.add_node('instruction_' + str(i), label=instruction[0], shape='box', fontcolor="#ffffff", fillcolor="#3c78d8", style='filled')

    for i in range(len(graph_json[0]['variable'])):
        variable = graph_json[0]['variable'][i]
        G.add_node('variable_' + str(i), label=variable[0], fillcolor="#f4cccc", fontcolor="#990000", style='filled', color="#990000", shape='ellipse')

    for i in range(len(graph_json[0]['constant'])):
        constant = graph_json[0]['constant'][i]
        G.add_node('constant_' + str(i), label=constant[0], fillcolor="#e99c9c", fontcolor="#990000", style='filled', color="#990000", shape='diamond')

    for i in range(len(graph_json[1]['instruction_control_instruction'])):
        instruction_control_instruction = graph_json[1]['instruction_control_instruction'][i]
        if graph_json[2]['instruction_control_instruction'][i] == 0:
            head_label = ''
        else: head_label = graph_json[2]['instruction_control_instruction'][i]
        G.add_edge('instruction_' + str(instruction_control_instruction[0]), 'instruction_' + str(instruction_control_instruction[1]), color="#345393", headlabel=head_label)

    for i in range(len(graph_json[1]['instruction_call_instruction'])):
        instruction_call_instruction = graph_json[1]['instruction_call_instruction'][i]
        if graph_json[2]['instruction_call_instruction'][i] == 0:
            head_label = ''
        else: head_label = graph_json[2]['instruction_call_instruction'][i]
        G.add_edge('instruction_' + str(instruction_call_instruction[0]), 'instruction_' + str(instruction_call_instruction[1]), color="#65ae4d", headlabel=head_label)

    for i in range(len(graph_json[1]['instruction_data_variable'])):
        instruction_data_variable = graph_json[1]['instruction_data_variable'][i]
        if graph_json[2]['instruction_data_variable'][i] == 0:
            head_label = ''
        else: head_label = graph_json[2]['instruction_data_variable'][i]
        G.add_edge('instruction_' + str(instruction_data_variable[0]), 'variable_' + str(instruction_data_variable[1]), color="#990000", headlabel=head_label)

    for i in range(len(graph_json[1]['variable_data_instruction'])):
        variable_data_instruction = graph_json[1]['variable_data_instruction'][i]
        if graph_json[2]['variable_data_instruction'][i] == 0:
            head_label = ''
        else: head_label = graph_json[2]['variable_data_instruction'][i]
        G.add_edge('variable_' + str(variable_data_instruction[0]), 'instruction_' + str(variable_data_instruction[1]), color="#990000", headlabel=head_label)

    for i in range(len(graph_json[1]['constant_data_instruction'])):
        constant_data_instruction = graph_json[1]['constant_data_instruction'][i]
        if graph_json[2]['constant_data_instruction'][i] == 0:
            head_label = ''
        else: head_label = graph_json[2]['constant_data_instruction'][i]
        G.add_edge('constant_' + str(constant_data_instruction[0]), 'instruction_' + str(constant_data_instruction[1]), color="#990000", headlabel=head_label)

    for i in range(len(graph_json[1]['instruction_data_constant'])):
        instruction_data_constant = graph_json[1]['instruction_data_constant'][i]
        if graph_json[2]['instruction_data_constant'][i] == 0:
            head_label = ''
        else: head_label = graph_json[2]['instruction_data_constant'][i]
        G.add_edge('instruction_' + str(instruction_data_constant[0]), 'constant_' + str(instruction_data_constant[1]), color="#990000", headlabel=head_label)

    try:
        for i in range(len(graph_json[0]['varray'])):
            varray = graph_json[0]['varray'][i]
            G.add_node('varray_' + str(i), label=varray[0], shape='hexagon', color="#990000", fontcolor="#990000", fillcolor="#f4cccc", style='filled')

        for i in range(len(graph_json[0]['vvector'])):
            vvector = graph_json[0]['vvector'][i]
            G.add_node('vvector_' + str(i), label=vvector[0], shape='octagon', color="#990000", fontcolor="#990000", fillcolor="#f4cccc", style='filled')

        for i in range(len(graph_json[0]['carray'])):
            carray = graph_json[0]['carray'][i]
            G.add_node('carray_' + str(i), label=carray[0], shape='box', color="#990000", fontcolor="#990000", fillcolor="#e99c9c", style='rounded,filled')

        for i in range(len(graph_json[0]['cvector'])):
            cvector = graph_json[0]['cvector'][i]
            G.add_node('cvector_' + str(i), label=cvector[0], shape='parallelogram', color="#990000", fontcolor="#990000", fillcolor="#e99c9c", style='dotted,filled')

        for i in range(len(graph_json[1]['instruction_data_varray'])):
            instruction_data_varray = graph_json[1]['instruction_data_varray'][i]
            if graph_json[2]['instruction_data_varray'][i] == 0:
                head_label = ''
            else: head_label = graph_json[2]['instruction_data_varray'][i]
            G.add_edge('instruction_' + str(instruction_data_varray[0]), 'varray_' + str(instruction_data_varray[1]), color="#990000", headlabel=head_label)

        for i in range(len(graph_json[1]['instruction_data_vvector'])):
            instruction_data_vvector = graph_json[1]['instruction_data_vvector'][i]
            if graph_json[2]['instruction_data_vvector'][i] == 0:
                head_label = ''
            else: head_label = graph_json[2]['instruction_data_vvector'][i]
            G.add_edge('instruction_' + str(instruction_data_vvector[0]), 'vvector_' + str(instruction_data_vvector[1]), color="#990000", headlabel=head_label)

        for i in range(len(graph_json[1]['varray_data_instruction'])):
            varray_data_instruction = graph_json[1]['varray_data_instruction'][i]
            if graph_json[2]['varray_data_instruction'][i] == 0:
                head_label = ''
            else: head_label = graph_json[2]['varray_data_instruction'][i]
            G.add_edge('varray_' + str(varray_data_instruction[0]), 'instruction_' + str(varray_data_instruction[1]), color="#990000", headlabel=head_label)

        for i in range(len(graph_json[1]['varray_data_varray'])):
            varray_data_varray = graph_json[1]['varray_data_varray'][i]
            if graph_json[2]['varray_data_varray'][i] == 0:
                head_label = ''
            else: head_label = graph_json[2]['varray_data_varray'][i]
            G.add_edge('varray_' + str(varray_data_varray[0]), 'varray_' + str(varray_data_varray[1]), color="#990000", headlabel=head_label)

        for i in range(len(graph_json[1]['vvector_data_vvector'])):
            vvector_data_vvector = graph_json[1]['vvector_data_vvector'][i]
            if graph_json[2]['vvector_data_vvector'][i] == 0:
                head_label = ''
            else: head_label = graph_json[2]['vvector_data_vvector'][i]
            G.add_edge('vvector_' + str(vvector_data_vvector[0]), 'vvector_' + str(vvector_data_vvector[1]), color="#990000", headlabel=head_label)

        for i in range(len(graph_json['vvector_data_instruction'])):
            vvector_data_instruction = graph_json[1]['vvector_data_instruction'][i]
            if graph_json[2]['vvector_data_instruction'][i] == 0:
                head_label = ''
            else: head_label = graph_json[2]['vvector_data_instruction'][i]
            G.add_edge('vvector_' + str(vvector_data_instruction[0]), 'instruction_' + str(vvector_data_instruction[1]), color="#990000", headlabel=head_label)

        for i in range(len(graph_json[1]['carray_data_instruction'])):
            carray_data_instruction = graph_json[1]['carray_data_instruction'][i]
            if graph_json[2]['carray_data_instruction'][i] == 0:
                head_label = ''
            else: head_label = graph_json[2]['carray_data_instruction'][i]
            G.add_edge('carray_' + str(carray_data_instruction[0]), 'instruction_' + str(carray_data_instruction[1]), color="#990000", headlabel=head_label)

        for i in range(len(graph_json[1]['carray_data_carray'])):
            carray_data_carray = graph_json[1]['carray_data_carray'][i]
            if graph_json[2]['carray_data_carray'][i] == 0:
                head_label = ''
            else: head_label = graph_json[2]['carray_data_carray'][i]
            G.add_edge('carray_' + str(carray_data_carray[0]), 'carray_' + str(carray_data_carray[1]), color="#990000", headlabel=head_label)

        for i in range(len(graph_json[1]['cvector_data_instruction'])):
            cvector_data_instruction = graph_json[1]['cvector_data_instruction'][i]
            if graph_json[2]['cvector_data_instruction'][i] == 0:
                head_label = ''
            else: head_label = graph_json[2]['cvector_data_instruction'][i]
            G.add_edge('cvector_' + str(cvector_data_instruction[0]), 'instruction_' + str(cvector_data_instruction[1]), color="#990000", headlabel=head_label)

        for i in range(len(graph_json[1]['cvector_data_cvector'])):
            cvector_data_cvector = graph_json[1]['cvector_data_cvector'][i]
            if graph_json[2]['cvector_data_cvector'][i] == 0:
                head_label = ''
            else: head_label = graph_json[2]['cvector_data_cvector'][i]
            G.add_edge('cvector_' + str(cvector_data_cvector[0]), 'cvector_' + str(cvector_data_cvector[1]), color="#990000", headlabel=head_label)

        for i in range(len(graph_json[1]['instruction_data_constant'])):
            instruction_data_constant = graph_json[1]['instruction_data_constant'][i]
            if graph_json[2]['instruction_data_constant'][i] == 0:
                head_label = ''
            else: head_label = graph_json[2]['instruction_data_constant'][i]
            G.add_edge('instruction_' + str(instruction_data_constant[0]), 'constant_' + str(instruction_data_constant[1]), color="#990000", headlabel=head_label)

        for i in range(len(graph_json[1]['instruction_data_carray'])):
            instruction_data_carray = graph_json[1]['instruction_data_carray'][i]
            if graph_json[2]['instruction_data_carray'][i] == 0:
                head_label = ''
            else: head_label = graph_json[2]['instruction_data_carray'][i]
            G.add_edge('instruction_' + str(instruction_data_carray[0]), 'carray_' + str(instruction_data_carray[1]), color="#990000", headlabel=head_label)

        for i in range(len(graph_json[1]['instruction_data_cconstant'])):
            instruction_data_cconstant = graph_json[1]['instruction_data_cconstant'][i]
            if graph_json[2]['instruction_data_cconstant'][i] == 0:
                head_label = ''
            else: head_label = graph_json[2]['instruction_data_cconstant'][i]
            G.add_edge('instruction_' + str(instruction_data_cconstant[0]), 'cconstant_' + str(instruction_data_cconstant[1]), color="#990000", headlabel=head_label)
    except: pass # Graph is in no_vectors form

    G.draw(file_name, prog='dot')