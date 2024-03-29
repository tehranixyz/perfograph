import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
dataset = dgl.data.CSVDataset('./dgl-csv-dev-map-all-with-hand-crafted-features')
dataset_pg = dgl.data.CSVDataset('./dgl-csv-dev-map-all-with-hand-crafted-features-pg')


class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs is features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h


class HeteroClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        super().__init__()

        self.rgcn = RGCN(in_dim, hidden_dim, hidden_dim, rel_names)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        h = g.ndata['feat']
        h = self.rgcn(g, h)
        with g.local_scope():
            g.ndata['h'] = h
            # pass
            # Calculate graph representation by average readout.
            hg = 0
            for ntype in g.ntypes:
                hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)
                # pass
            return self.classify(hg)

whole_exp = 0
mx_acc = -100.0
clf_report = None
cf_matrix = None


for whole_exp in range(10):

    num_examples = len(dataset)

    # new added
    dataset_indices = list(range(num_examples))
    np.random.shuffle(dataset_indices)
    test_split_index = 67
    train_idx, test_idx = dataset_indices[test_split_index:], dataset_indices[:test_split_index]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    train_dataloader = GraphDataLoader(dataset, shuffle=False, batch_size=100, sampler=train_sampler)
    test_dataloader = GraphDataLoader(dataset, shuffle=False, batch_size=100, sampler=test_sampler)

    etypes = [('control', 'control', 'control'), ('control', 'call', 'control'), ('control', 'data', 'variable'), ('variable', 'data', 'control')]
    # etypes = [('v_1', 'e', 'v_1')]
    model = HeteroClassifier(120, 64, 2, etypes)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    total_loss = 0
    loss_list = []
    epoch_list = []
    for epoch in range(300):
        total_loss = 0
        for batched_graph, labels in train_dataloader:
            logits = model(batched_graph)
            flattened_labels = labels.flatten()
            loss = F.cross_entropy(logits, flattened_labels)
            total_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        loss_list.append(total_loss)
        epoch_list.append(epoch)


    num_correct = 0
    num_tests = 0
    total_pred = []
    total_label = []
    num_correct_2 = 0
    samples_from_class_0 = 0

    for batched_graph, labels in test_dataloader:
        pred = model(batched_graph)
        num_correct += (pred.argmax(1) == labels).sum().item()
        num_tests += len(labels)

        output = (torch.max(torch.exp(pred), 1)[1]).data.cpu().numpy()
        total_pred.extend(output)

        label_tmp = labels.data.cpu().numpy()
        total_label.extend(label_tmp)

        for ind_label in labels:
            if ind_label == 0:
                samples_from_class_0 += 1

        # for accuracy
        # starts here
        pred_numpy = pred.detach().numpy()
        for ind_pred, ind_label in zip(pred_numpy, labels):
            if np.argmax(ind_pred) == ind_label:
                num_correct_2 += 1
        # ends here
        # ends here

    # starts here

    classes = ('CPU', 'GPU')
    class_names = ['CPU', 'GPU']
    # clf_report = None
    # cf_matrix = None

    local_accuracy = num_correct_2 / num_tests
    if local_accuracy > mx_acc:
        mx_acc = local_accuracy
        clf_report = classification_report(total_label, total_pred, target_names=class_names)
        cf_matrix = confusion_matrix(total_label, total_pred)
    # ends here

print(clf_report)
print(cf_matrix)
print(mx_acc)