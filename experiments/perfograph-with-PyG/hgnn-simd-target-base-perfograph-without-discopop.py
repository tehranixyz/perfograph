import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, Linear, HeteroConv, global_mean_pool
from torch_geometric.data import HeteroData, DataLoader, Batch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import time

start_time = time.time()

# Create a batch of graphs with manual labels
graphs = torch.load('./datalist-base-perfograph.pt')
# 42 gave best
np.random.seed(873)

dataset_indices = list(range(len(graphs)))

cpu_indices = list(range(126))
gpu_indices = list(range(126, 199))

np.random.shuffle(cpu_indices)
np.random.shuffle(gpu_indices)

updated_cpu_indices = cpu_indices[:73]


train_idx = []
val_idx = []
test_idx = []

for i in range(73):
    if i < 59:
        train_idx.append(updated_cpu_indices[i])
    elif i >= 59 and i < 66:
        val_idx.append(updated_cpu_indices[i])
    else:
        test_idx.append(updated_cpu_indices[i])

for i in range(73):
    if i < 59:
        train_idx.append(gpu_indices[i])
    elif i >= 59 and i < 66:
        val_idx.append(gpu_indices[i])
    else:
        test_idx.append(gpu_indices[i])

# np.random.shuffle(dataset_indices)

# train_idx, val_idx, test_idx = dataset_indices[:114], dataset_indices[114:124], dataset_indices[124:134]
np.random.shuffle(train_idx)
np.random.shuffle(val_idx)
np.random.shuffle(test_idx)


train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)
test_sampler = SubsetRandomSampler(test_idx)
print(train_idx, val_idx, test_idx)



# Define the GAT model with 6 GATConv layers for heterogeneous data
class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(GAT, self).__init__()
        self.convs = torch.nn.ModuleList()

        self.convs.append(HeteroConv({
            ('control', 'control', 'control'): GATConv((-1, -1), hidden_channels, heads=8, concat=False, add_self_loops=False),
            ('control', 'call', 'control'): GATConv((-1, -1), hidden_channels, heads=8, concat=False, add_self_loops=False),
            ('control', 'data', 'variable'): GATConv((-1, -1), hidden_channels, heads=8, concat=False, add_self_loops=False),
            ('variable', 'data', 'control'): GATConv((-1, -1), hidden_channels, heads=8, concat=False, add_self_loops=False),
            ('constant', 'data', 'control'): GATConv((-1, -1), hidden_channels, heads=8, concat=False, add_self_loops=False),
            ('control', 'data', 'constant'): GATConv((-1, -1), hidden_channels, heads=8, concat=False, add_self_loops=False)
        }))

        for _ in range(1):
            self.convs.append(HeteroConv({
                ('control', 'control', 'control'): GATConv((-1, -1), hidden_channels, heads=8, concat=False, add_self_loops=False),
                ('control', 'call', 'control'): GATConv((-1, -1), hidden_channels, heads=8, concat=False, add_self_loops=False),
                ('control', 'data', 'variable'): GATConv((-1, -1), hidden_channels, heads=8, concat=False, add_self_loops=False),
                ('variable', 'data', 'control'): GATConv((-1, -1), hidden_channels, heads=8, concat=False, add_self_loops=False),
                ('constant', 'data', 'control'): GATConv((-1, -1), hidden_channels, heads=8, concat=False, add_self_loops=False),
                ('control', 'data', 'constant'): GATConv((-1, -1), hidden_channels, heads=8, concat=False, add_self_loops=False)
            }))

        self.lin = Linear(64*3, out_channels)

    def forward(self, x_dict, edge_index_dict, batch_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.elu(x) for key, x in x_dict.items()}

        # Apply global mean pooling for each node type
        pooled_dict = {key: global_mean_pool(x, batch_dict[key]) for key, x in x_dict.items()}

        # Concatenate the pooled outputs
        x = torch.cat([pooled_dict[key] for key in pooled_dict.keys()], dim=1)

        x = self.lin(x)
        return F.log_softmax(x, dim=1)

def load_model_with_size_filter(model_best, state_dict):
    # Create a new state dictionary to store compatible weights

    filtered_state_dict = {}

    for key, param in state_dict.items():

        # Check if the parameter shape matches the model's parameter

        if model_best.state_dict()[key].size() == param.size():
            filtered_state_dict[key] = param

    # Load the filtered state dictionary with strict=False

    model_best.load_state_dict(filtered_state_dict, strict=False)

    return model_best


# Initialize model, optimizer, and loss function
model = GAT(hidden_channels=64, out_channels=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()


# Training loop
def train():
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
        loss = criterion(out, batch.label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


# Testing loop
def val(loader):
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        out = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
        # print('this is out: ', out)
        pred = out.argmax(dim=1)
        correct += int((pred == batch.label).sum())
        total += batch.label.size(0)
    return correct / total

def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        out = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
        # print('this is out: ', out)
        pred = out.argmax(dim=1)
        print('this is pred: ', pred)
        print('this is label: ', batch.label)
        correct += int((pred == batch.label).sum())
        total += batch.label.size(0)
    return correct / total


# Split data into train and test sets
train_loader = DataLoader(graphs, batch_size=32, sampler=train_sampler)
val_loader = DataLoader(graphs, batch_size=32, sampler=val_sampler)
test_loader = DataLoader(graphs, batch_size=32, sampler=test_sampler)

# Run the training and testing loops
training_loss = []
validation_loss = []
global_accuracy = -100.0
global_validation_accuracy = -100.0

for epoch in range(1, 20):
    loss = train()
    training_loss.append(loss)
    val_acc = val(val_loader)
    validation_loss.append(1.0-val_acc)
    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Test Accuracy: {val_acc:.4f}')

torch.save(model.state_dict(), './model-simd-target-validtn-model-20-epoch.pt')


print('highest validation accuracy: ', global_validation_accuracy)

base_model = GAT(hidden_channels=64, out_channels=2)

loaded_state_dict_all_epoch_model = torch.load("model-simd-target-validtn-model-20-epoch.pt")
model_all_epoch  = load_model_with_size_filter(base_model, loaded_state_dict_all_epoch_model)
acc_with_all_epoch_model = test(model_all_epoch, test_loader)
print('final accuracy with all epoch model: ', acc_with_all_epoch_model)
