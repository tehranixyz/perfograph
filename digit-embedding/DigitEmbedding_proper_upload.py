import re
import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
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
                # new added
                node_embed_list_mult = [ii * 10 for ii in node_embed_list]
                # add ends
                # this was actual
                # digit_pos_vector.append(node_embed_list)
                digit_pos_vector.append(node_embed_list_mult)
            else:
                digit_pos_vector.append([0.0, 0.0, 0.0])

        final_embedding_vector = []


        #newly added
        # ve = digit_pos_vector
        # for ar in digit_pos_vector:
        #     for i in range(len(ar)):
        #         b = ar[i]
        #         ar[i]*= 100
        # ve2 = digit_pos_vector
        #add ends

        # mult instead of add
        final_embedding_list_of_np_arrays = list(
            (np.array(digit_embedding_vector) * np.array(digit_pos_vector)))

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
feat_count = 0
feature_file = open("./FeatureMap/feature_map_file_pg_plus_text_all_digits.txt", 'r')
feature_lines = feature_file.readlines()
feature_map = {}
for feature_line in feature_lines:
    feature_key = feature_line.split(",")[0]
    if feature_key not in feature_map:
        feature_key = feature_key
        feature_map[feature_key] = feat_count
        feat_count = feat_count + 1
embeds = nn.Embedding(feat_count, 2)

# arr_num = [10, 12, 1000, 1020, 1000000, 10000020]
arr_num = []

# for i in range(20):
#     arr_num.append(random.uniform(1.10, 10.50))

for i in range(50):
    arr_num.append(random.randint(1, 51))

# for i in range(20):
#     arr_num.append(random.uniform(20.02, 30.90))


for i in range(50):
    arr_num.append(random.randint(200050, 200101))

# emb_file = open('emb-plot.csv', 'w')
emb_file = open('emb-plot-dec-proper-1-100-200051-502001-updated.csv', 'w')

arr_num_emb = []

for num in tqdm(arr_num):
    num_emb = get_digit_emb_of_number(str(num), feature_map)
    # if num_emb[0] < 0:
    #     num_emb[0] *= -1
    # if num_emb[1] < 0:
    #     num_emb[1] *= -1
    num_emb.append(str(num))
    arr_num_emb.append(num_emb)

count = 0
for emb in tqdm(arr_num_emb):
    if count == 0:
        emb_file.write('Number,x,y\n')
        count += 1
    write_string = str(emb[2]) + ',' + str(emb[0]) + ',' + str(emb[1]) + '\n'
    emb_file.write(write_string)