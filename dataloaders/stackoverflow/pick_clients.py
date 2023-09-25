import os
import sys

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)

from configs.hyperparameters import *
import datasets.stackoverflow.stackoverflow_nwp_dataloader as stackoverflow_loader

vocab_size = stackoverflow['vocab_size']
sequence_length = stackoverflow['sequence_length']
batch_size = stackoverflow['batch_size']

train_data, test_data = stackoverflow_loader.get_federated_datasets(vocab_size, sequence_length, train_client_batch_size=batch_size)

clients = list(set(train_data.client_ids) & set(test_data.client_ids))

print(clients)

with open('./datasets/stackoverflow/available_clients.txt', 'w') as f:
    for c in clients:
        f.write(c+"\n")