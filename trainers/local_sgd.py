import os
import sys

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)
# print(sys.path)

import glob
import json
import math
import copy
import random
import argparse
from colorama import Fore

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from configs.hyperparameters import *

import dataloaders.stackoverflow.stackoverflow_nwp_dataloader as stackoverflow_nwp_dataloader
import dataloaders.stackoverflow.stackoverflow_lr_dataloader as stackoverflow_lr_dataloader
import dataloaders.emnist.emnist_dataloader as emnist_dataloader
import dataloaders.cifar10.cifar10_dataloader as cifar10_dataloader
import dataloaders.cifar100.cifar100_dataloader as cifar100_dataloader
import dataloaders.shakespeare.shakespeare_dataloader as shakespeare_dataloader

from models.stackoverflow_nwp_rnn import StackoverflowNet as StackoverflowNWPNet
from models.synthetic_fc import SyntheticNet
from models.emnist_cnn import EMNISTNet
from models.cifar100_resnet import resnet18 as CifarResNet
from models.shakespeare_rnn import ShakespeareNet
from models.stackoverflow_tag_lr import StackoverflowLogisticRegression as StackoverflowLRNet

AVAILABLE_GPUS = 1 #torch.cuda.device_count()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def stackoverflow_nwp_train(net, train_data, test_data):
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(DEVICE)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    
    net.train()
    for epoch in range(epochs):
        
        accumulated_loss = 0
        sample_count = 0
        
        for tokens, next_tokens in train_data:
            tokens_, next_tokens_ = torch.from_numpy(tokens.numpy()).to(DEVICE), torch.from_numpy(next_tokens.numpy()).to(DEVICE)
            optimizer.zero_grad()
            logits = net(tokens_)
            loss = criterion(logits, next_tokens_)
            accumulated_loss += loss.item()
            sample_count += tokens.shape[0]
            loss.backward()
            optimizer.step()
    
    _, accuracy, _, _ = stackoverflow_nwp_test(net, test_data)

    with open(checkpoint_dir + 'intermediate_'+METHOD+ '_' + str(num_clients) + '_' + str(total_clients) + '_' + str(total_rounds) + '_' + str(epochs) + '_' + str(lr).replace('.', '_') + '_test_accuracies.txt', 'a+') as f:
        f.write(str(accuracy)+"\n")

    return accumulated_loss / sample_count, sample_count

def stackoverflow_nwp_test(net, test_data):
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(DEVICE)
    correct, total, loss = 0, 0, 0.0
    net.eval()

    with torch.no_grad():
        for tokens, next_tokens in test_data:
            tokens_, next_tokens_ = torch.from_numpy(tokens.numpy()).to(DEVICE), torch.from_numpy(next_tokens.numpy()).to(DEVICE)
            logits = net(tokens_)
            loss += criterion(logits, next_tokens_).item()
            _, predicted = torch.max(logits, 1)
            paddings = ~(next_tokens_ == 0)
            total += torch.count_nonzero(paddings).item()
            correct += ((predicted == next_tokens_) * paddings).sum().item()

    accuracy = correct / total
    # print("Accuracy: ", correct, "/", total, " = ", accuracy)
    return loss, accuracy, correct, total

def stackoverflow_nwp_generalized_test(net, train_data, test_data):
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(DEVICE)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    before_loss, before_accuracy, before_correct, before_total = stackoverflow_nwp_test(net, test_data)

    net.train()
    for epoch in range(epochs):
        
        for tokens, next_tokens in train_data:
            tokens_, next_tokens_ = torch.from_numpy(tokens.numpy()).to(DEVICE), torch.from_numpy(next_tokens.numpy()).to(DEVICE)
            optimizer.zero_grad()
            logits = net(tokens_)
            loss = criterion(logits, next_tokens_)
            loss.backward()
            optimizer.step()

    after_loss, after_accuracy, after_correct, after_total = stackoverflow_nwp_test(net, test_data)
    
    # print("Accuracy: ", correct, "/", total, " = ", accuracy)
    return before_loss, before_accuracy, after_loss, after_accuracy, after_correct, after_total

def synthetic_train(net, train_data, test_data):
    criterion = nn.NLLLoss().to(DEVICE)
    optimizer = optim.SGD(net.parameters(), lr=lr)

    net.train()
    for epoch in range(epochs):

        accumulated_loss = 0
        sample_count = 0

        for i in range(0, len(train_data['x']), batch_size):
            true_output = np.asarray(train_data['y'][i:i+batch_size], dtype=np.int32)
            input_, true_output = torch.from_numpy(np.asarray(train_data['x'][i:i+batch_size], dtype=np.float32)).to(DEVICE), torch.from_numpy(true_output).to(DEVICE)
            true_output = true_output.type(torch.LongTensor).to(DEVICE)

            optimizer.zero_grad()
            pred_output = net(input_)
            loss = criterion(pred_output, true_output)
            accumulated_loss += loss.item()
            sample_count += input_.shape[0]
            loss.backward()
            optimizer.step()

    _, accuracy, _, _ = synthetic_test(net, test_data)
    with open(checkpoint_dir + 'intermediate_'+METHOD+ '_' + str(num_clients) + '_' + str(total_clients) + '_' + str(total_rounds) + '_' + str(epochs) + '_' + str(lr).replace('.', '_') + '_test_accuracies.txt', 'a+') as f:
        f.write(str(accuracy)+"\n")

    return accumulated_loss / sample_count, sample_count

def synthetic_test(net, test_data):
    criterion = nn.NLLLoss().to(DEVICE)
    correct, total, loss = 0, 0, 0.0
    net.eval()

    with torch.no_grad():
        for i in range(0, len(test_data['x']), batch_size):
            true_output = np.asarray(test_data['y'][i:i+batch_size], dtype=np.int32)

            input_, true_output = torch.from_numpy(np.asarray(test_data['x'][i:i+batch_size], dtype=np.float32)).to(DEVICE), torch.from_numpy(true_output).to(DEVICE)
            true_output = true_output.type(torch.LongTensor).to(DEVICE)

            pred_output = net(input_)
            loss += criterion(pred_output, true_output)
            _, predicted = torch.max(pred_output, 1)
            # print(predicted)
            paddings = ~(true_output == 0)
            total += torch.count_nonzero(paddings).item()
            # total += pred_output.shape[0]
            correct += ((predicted == true_output) * paddings).sum().item()
            # correct += ((predicted == true_output)).sum().item()
            
    accuracy = correct / total
    # print("Accuracy: ", correct, "/", total, " = ", accuracy)
    return loss, accuracy, correct, total

def synthetic_generalized_test(net, train_data, test_data):
    criterion = nn.NLLLoss().to(DEVICE)
    optimizer = optim.SGD(net.parameters(), lr=lr)

    before_loss, before_accuracy, before_correct, before_total = synthetic_test(net, test_data)
    
    for epoch in range(epochs):
        net.train()
        
        for i in range(0, len(train_data['x']), batch_size):
            true_output = np.asarray(train_data['y'][i:i+batch_size], dtype=np.int32)

            input_, true_output = torch.from_numpy(np.asarray(train_data['x'][i:i+batch_size], dtype=np.float32)).to(DEVICE), torch.from_numpy(true_output).to(DEVICE)
            true_output = true_output.type(torch.LongTensor).to(DEVICE)

            optimizer.zero_grad()
            pred_output = net(input_)
            loss = criterion(pred_output, true_output)
            loss.backward()
            optimizer.step()

    after_loss, after_accuracy, after_correct, after_total = synthetic_test(net, test_data)
    
    # print("Accuracy: ", correct, "/", total, " = ", accuracy)
    return before_loss, before_accuracy, after_loss, after_accuracy, after_correct, after_total

def emnist_train(net, train_data, test_data):
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    
    net.train()
    for epoch in range(epochs):
        
        accumulated_loss = 0
        sample_count = 0

        for inputs, labels in train_data:
            labels = labels.numpy()
            inputs, labels = torch.from_numpy(inputs.numpy()).to(DEVICE), torch.from_numpy(labels).type(torch.LongTensor).to(DEVICE)
            if inputs.shape[0] == 1:
                break

            optimizer.zero_grad()
            logits = net(inputs)
            loss = criterion(logits, labels)
            accumulated_loss += loss.item()
            sample_count += inputs.shape[0]
            loss.backward()
            optimizer.step()
    
    _, accuracy, _, _ = emnist_test(net, test_data)

    with open(checkpoint_dir + 'intermediate_'+METHOD+ '_' + str(num_clients) + '_' + str(total_clients) + '_' + str(total_rounds) + '_' + str(epochs) + '_' + str(lr).replace('.', '_') + '_test_accuracies.txt', 'a+') as f:
        f.write(str(accuracy)+"\n")

    return accumulated_loss / sample_count, sample_count

def emnist_test(net, test_data):
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    correct, total, loss = 0, 0, 0.0
    net.eval()

    with torch.no_grad():
        for inputs, labels in test_data:
            labels = labels.numpy()
            inputs, labels = torch.from_numpy(inputs.numpy()).to(DEVICE), torch.from_numpy(labels).type(torch.LongTensor).to(DEVICE)

            logits = net(inputs)
            loss += criterion(logits, labels).item()
            _, predicted = torch.max(logits, 1)
            total += labels.shape[0]
            correct += ((predicted == labels)).sum().item()

    accuracy = correct / total
    # print("Accuracy: ", correct, "/", total, " = ", accuracy)
    return loss, accuracy, correct, total

def emnist_generalized_test(net, train_data, test_data):
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    before_loss, before_accuracy, before_correct, before_total = emnist_test(net, test_data)

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    for epoch in range(epochs):
        
        for inputs, labels in train_data:
            labels = labels.numpy()
            inputs, labels = torch.from_numpy(inputs.numpy()).to(DEVICE), torch.from_numpy(labels).type(torch.LongTensor).to(DEVICE)
            
            optimizer.zero_grad()
            logits = net(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    after_loss, after_accuracy, after_correct, after_total = emnist_test(net, test_data)
    # print("Accuracy: ", correct, "/", total, " = ", accuracy)
    return before_loss, before_accuracy, after_loss, after_accuracy, after_correct, after_total

def shakespeare_train(net, train_data, test_data):
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(DEVICE)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    
    net.train()
    for epoch in range(epochs):
        
        accumulated_loss = 0
        sample_count = 0
        
        for chars, next_chars in train_data:
            chars, next_chars = torch.from_numpy(chars.numpy()).to(DEVICE), torch.from_numpy(next_chars.numpy()).to(DEVICE)
            optimizer.zero_grad()
            logits = net(chars)
            loss = criterion(logits, next_chars)
            accumulated_loss += loss.item()
            sample_count += chars.shape[0]
            loss.backward()
            optimizer.step()

    _, accuracy, _, _ = shakespeare_test(net, test_data)
    
    with open(checkpoint_dir + 'intermediate_'+METHOD+ '_' + str(num_clients) + '_' + str(total_clients) + '_' + str(total_rounds) + '_' + str(epochs) + '_' + str(lr).replace('.', '_') + '_test_accuracies.txt', 'a+') as f:
        f.write(str(accuracy)+"\n")

    return accumulated_loss / sample_count, sample_count

def shakespeare_test(net, test_data):
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(DEVICE)
    correct, total, loss = 0, 0, 0.0
    net.eval()

    with torch.no_grad():
        for chars, next_chars in test_data:
            chars, next_chars = torch.from_numpy(chars.numpy()).to(DEVICE), torch.from_numpy(next_chars.numpy()).to(DEVICE)
            logits = net(chars)
            loss += criterion(logits, next_chars).item()
            _, predicted = torch.max(logits, 1)
            paddings = ~(next_chars == 0)
            total += torch.count_nonzero(paddings).item()
            correct += ((predicted == next_chars) * paddings).sum().item()

    accuracy = correct / total
    # print("Accuracy: ", correct, "/", total, " = ", accuracy)
    return loss, accuracy, correct, total

def shakespeare_generalized_test(net, train_data, test_data):
    before_loss, before_accuracy, before_correct, before_total = shakespeare_test(net, test_data)
    after_loss, after_accuracy, after_correct, after_total = 0.0, 0.0, 0, 0

    criterion = nn.CrossEntropyLoss(ignore_index=0).to(DEVICE)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    
    for epoch in range(epochs):
        net.train()
        for chars, next_chars in train_data:
            chars, next_chars = torch.from_numpy(chars.numpy()).to(DEVICE), torch.from_numpy(next_chars.numpy()).to(DEVICE)
            optimizer.zero_grad()
            logits = net(chars)
            loss = criterion(logits, next_chars)
            loss.backward()
            optimizer.step()
    
        loss, accuracy, correct, total = shakespeare_test(net, test_data)
        if after_accuracy < accuracy:
            after_loss, after_accuracy, after_correct, after_total = loss, accuracy, correct, total

    return before_loss, before_accuracy, after_loss, after_accuracy, after_correct, after_total

def cifar10_train(net, train_data, test_data):
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    
    net.train()
    for epoch in range(epochs):
        
        accumulated_loss = 0
        sample_count = 0
        for inputs, labels in train_data:
            labels = labels.numpy()
            inputs, labels = torch.from_numpy(inputs.numpy()).to(DEVICE), torch.from_numpy(labels).to(DEVICE)
            inputs = inputs.swapaxes(1,3)
            inputs = inputs.swapaxes(2,3)

            optimizer.zero_grad()
            logits = net(inputs)
            loss = criterion(logits, labels)
            accumulated_loss += loss.item()
            sample_count += inputs.shape[0]
            loss.backward()
            optimizer.step()
    
    _, accuracy, _, _ = cifar10_test(net, test_data)

    with open(checkpoint_dir + 'intermediate_'+METHOD+ '_' + str(num_clients) + '_' + str(total_clients) + '_' + str(total_rounds) + '_' + str(epochs) + '_' + str(lr).replace('.', '_') + '_test_accuracies.txt', 'a+') as f:
        f.write(str(accuracy)+"\n")

    return accumulated_loss / sample_count, sample_count

def cifar10_test(net, test_data):
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    correct, total, loss = 0, 0, 0.0
    net.eval()

    with torch.no_grad():
        for inputs, labels in test_data:
            labels = labels.numpy()
            inputs, labels = torch.from_numpy(inputs.numpy()).to(DEVICE), torch.from_numpy(labels).to(DEVICE)
            inputs = inputs.swapaxes(1,3)
            inputs = inputs.swapaxes(2,3)

            logits = net(inputs)
            loss += criterion(logits, labels).item()
            _, predicted = torch.max(logits, 1)
            total += labels.shape[0]
            correct += ((predicted == labels)).sum().item()

    accuracy = correct / total
    return loss, accuracy, correct, total

def cifar10_generalized_test(net, train_data, test_data):
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    before_loss, before_accuracy, before_correct, before_total = cifar10_test(net, test_data)

    for epoch in range(epochs):
        net.train()
        
        for inputs, labels in train_data:
            labels = labels.numpy()
            inputs, labels = torch.from_numpy(inputs.numpy()).to(DEVICE), torch.from_numpy(labels).to(DEVICE)
            inputs = inputs.swapaxes(1,3)
            inputs = inputs.swapaxes(2,3)
            
            optimizer.zero_grad()
            logits = net(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
    
    after_loss, after_accuracy, after_correct, after_total = cifar10_test(net, test_data)
    
    return before_loss, before_accuracy, after_loss, after_accuracy, after_correct, after_total

def cifar100_train(net, train_data, test_data):
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    
    net.train()
    for epoch in range(epochs):
        
        accumulated_loss = 0
        sample_count = 0
        for inputs, labels in train_data:
            labels = labels.numpy()
            # inputs, labels = inputs.to(DEVICE), labels.type(torch.LongTensor).to(DEVICE)
            inputs, labels = torch.from_numpy(inputs.numpy()).to(DEVICE), torch.from_numpy(labels).to(DEVICE)
            inputs = inputs.swapaxes(1,3)
            inputs = inputs.swapaxes(2,3)
            
            optimizer.zero_grad()
            logits = net(inputs)
            loss = criterion(logits, labels)
            accumulated_loss += loss.item()
            sample_count += inputs.shape[0]
            loss.backward()
            optimizer.step()
    
    _, accuracy, _, _ = cifar100_test(net, test_data)

    with open(checkpoint_dir + 'intermediate_'+METHOD+ '_' + str(num_clients) + '_' + str(total_clients) + '_' + str(total_rounds) + '_' + str(epochs) + '_' + str(lr).replace('.', '_') + '_test_accuracies.txt', 'a+') as f:
        f.write(str(accuracy)+"\n")

    return accumulated_loss / sample_count, sample_count

def cifar100_test(net, test_data):
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    correct, total, loss = 0, 0, 0.0
    net.eval()

    with torch.no_grad():
        for inputs, labels in test_data:
            labels = labels.numpy()
            inputs, labels = torch.from_numpy(inputs.numpy()).to(DEVICE), torch.from_numpy(labels).to(DEVICE)
            inputs = inputs.swapaxes(1,3)
            inputs = inputs.swapaxes(2,3)

            logits = net(inputs)
            loss += criterion(logits, labels).item()
            _, predicted = torch.max(logits, 1)
            total += labels.shape[0]
            correct += ((predicted == labels)).sum().item()

    accuracy = correct / total
    return loss, accuracy, correct, total

def cifar100_generalized_test(net, train_data, test_data):
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    before_loss, before_accuracy, before_correct, before_total = cifar100_test(net, test_data)

    for epoch in range(epochs):
        net.train()
        
        for inputs, labels in train_data:
            labels = labels.numpy()
            inputs, labels = torch.from_numpy(inputs.numpy()).to(DEVICE), torch.from_numpy(labels).to(DEVICE)
            inputs = inputs.swapaxes(1,3)
            inputs = inputs.swapaxes(2,3)

            optimizer.zero_grad()
            logits = net(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    after_loss, after_accuracy, after_correct, after_total = cifar100_test(net, test_data)
    
    return before_loss, before_accuracy, after_loss, after_accuracy, after_correct, after_total

def stackoverflow_lr_train(net, train_data, test_data):
    criterion = nn.BCELoss(reduction="sum").to(DEVICE)
    optimizer = optim.SGD(net.parameters(), lr=lr)#, momentum=0.9)
    
    net.train()
    for epoch in range(epochs):

        accumulated_loss = 0
        sample_count = 0
        
        for tokens, tags in train_data:
            tokens_, tags_ = torch.from_numpy(tokens.numpy()).to(DEVICE), torch.from_numpy(tags.numpy()).to(DEVICE)
            optimizer.zero_grad()
            logits = net(tokens_)
            loss = criterion(logits, tags_)
            accumulated_loss += loss.item()
            sample_count += tokens.shape[0]
            loss.backward()
            optimizer.step()
    
    _, accuracy, _, _ = stackoverflow_lr_test(net, test_data)

    with open(checkpoint_dir + 'intermediate_'+METHOD+ '_' + str(num_clients) + '_' + str(total_clients) + '_' + str(total_rounds) + '_' + str(epochs) + '_' + str(lr).replace('.', '_') + '_test_accuracies.txt', 'a+') as f:
        f.write(str(accuracy)+"\n")

    return accumulated_loss / sample_count, sample_count

def stackoverflow_lr_test(net, test_data):
    criterion = nn.BCELoss(reduction="sum").to(DEVICE)
    correct, total, loss = 0, 0, 0.0
    net.eval()

    with torch.no_grad():
        for tokens, tags in test_data:
            tokens_, tags_ = torch.from_numpy(tokens.numpy()).to(DEVICE), torch.from_numpy(tags.numpy()).to(DEVICE)
            logits = net(tokens_)
            loss += criterion(logits, tags_).item()

            # correct += predicted.eq(tags_).sum(axis=-1).eq(tags_.size(1)).sum().item()
            # true_positive = ((tags_ * predicted) > 0.1).int().sum(axis=-1)
            # recall at top 5
            k = 5
            logits_topk = torch.topk(input=logits, k=k, dim=-1)
            # predicted_topk = (logits_topk.values > 0.5).int()
            ## correct_topk = torch.zeros(logits_topk.values.shape)
            # correct += torch.sum(predicted_topk).item()
            ## total += predicted_topk.shape[0] * predicted_topk.shape[1]
            # all_tags = torch.sum(tags_, dim=-1)
            # top_tags = torch.where(all_tags < k, all_tags, 5)
            # total += torch.sum(top_tags).item()
            for idx in range(tags_.shape[0]):
                # row = tags_[idx, logits_topk.indices[idx, :]]
                # correct_topk[idx] = tags_[idx, logits_topk.indices[idx, :]]
                correct += torch.sum(tags_[idx, logits_topk.indices[idx, :]]).item()
                total += k #torch.sum(tags_[idx, :]).item()
            # print(correct, total)
            # input('test')

    recall = correct / total
    # print("Recall: ", recall)
    return loss, recall, correct, total

def stackoverflow_lr_generalized_test(net, train_data, test_data):
    criterion = nn.BCELoss(reduction="sum").to(DEVICE)
    optimizer = optim.SGD(net.parameters(), lr=lr)#, momentum=0.9)

    before_loss, before_recall, before_correct, before_total = stackoverflow_lr_test(net, test_data)
    after_loss, after_recall, after_correct, after_total = 0.0, 0.0, 0, 0

    for epoch in range(epochs):
        net.train()
        
        for tokens, tags in train_data:
            tokens_, tags_ = torch.from_numpy(tokens.numpy()).to(DEVICE), torch.from_numpy(tags.numpy()).to(DEVICE)
            optimizer.zero_grad()
            logits = net(tokens_)
            loss = criterion(logits, tags_)
            loss.backward()
            optimizer.step()

        loss, recall, correct, total = stackoverflow_lr_test(net, test_data)
        if after_recall < recall:
            after_loss, after_recall, after_correct, after_total = loss, recall, correct, total
    
    # after_accuracy = after_correct / after_total
    # after_recall = after_true_positive / after_total
    # print("Accuracy: ", correct, "/", total, " = ", accuracy)
    return before_loss, before_recall, after_loss, after_recall, after_correct, after_total



class StackoverflowNWPClient():

    def __init__(self):# -> None:
        pass

    def fit(self):
        for idx, cid in enumerate(clients):
            print("Index: ", idx)
            net = StackoverflowNWPNet(vocab_size + 4, embedding_size, hidden_size, num_layers).to(DEVICE)
            train_data_ = train_data.create_tf_dataset_for_client(cid.strip())
            test_data_ = test_data.create_tf_dataset_for_client(cid.strip())
            loss, count = stackoverflow_nwp_train(net, train_data_, test_data_)

class SyntheticClient():

    def __init__(self):# -> None:
        pass

    def fit(self):
        len_clients = len(clients)
        for idx, cid in enumerate(clients):
            print("Index: ", idx, "/", len_clients)

            net = SyntheticNet(input_dim, hidden_dim, output_dim).to(DEVICE)
            train_data_ = train_data[cid.strip()]
            test_data_ = test_data[cid.strip()]
            loss, count, max_epoch = synthetic_train(net, train_data_, test_data_)

class EMNISTClient():

    def fit(self):
        len_clients = len(clients)
        for idx, cid in enumerate(clients):
            print("Index: ", idx, "/", len_clients)

            net = EMNISTNet().to(DEVICE)
            train_data_ = train_data.create_tf_dataset_for_client(cid.strip())
            test_data_ = test_data.create_tf_dataset_for_client(cid.strip())
            loss, count = emnist_train(net, train_data_, test_data_)

class ShakespeareClient():

    def fit(self):
        len_clients = len(clients)
        for idx, cid in enumerate(clients):
            cid = train_data.client_ids[cid]
            print("Index: ", idx, "/", len_clients)
            net = ShakespeareNet(embedding_size, vocab_size, hidden_size).to(DEVICE)
            train_data_ = train_data.create_tf_dataset_for_client(cid)
            test_data_ = test_data.create_tf_dataset_for_client(cid)
            loss, count = shakespeare_train(net, train_data_, test_data_)

class Cifar10Client():

    def fit(self):
        len_clients = len(clients)
        for idx, cid in enumerate(clients):
            cid = str(cid)
            print("Index: ", idx, "/", len_clients)

            net = CifarResNet(num_classes=10).to(DEVICE)
            train_data_ = train_data.create_tf_dataset_for_client(cid)
            test_data_ = test_data.create_tf_dataset_for_client(cid)
            loss, count = cifar10_train(net, train_data_, test_data_)

class Cifar100Client():

    def fit(self):
        len_clients = len(clients)
        for idx, cid in enumerate(clients):
            cid = str(cid)
            print("Index: ", idx, "/", len_clients)

            net = CifarResNet(num_classes=100).to(DEVICE)
            train_data_ = train_data.create_tf_dataset_for_client(cid)
            test_data_ = test_data.create_tf_dataset_for_client(cid)
            loss, count = cifar100_train(net, train_data_, test_data_)

class StackoverflowLRClient():

    def fit(self):
        len_clients = len(clients)
        for idx, cid in enumerate(clients):
            cid = cid.strip()
            print("Index: ", idx, "/", len_clients)
            net = StackoverflowLRNet(word_vocab_size, tag_vocab_size).to(DEVICE)
            train_data_ = train_data.create_tf_dataset_for_client(cid)
            test_data_ = test_data.create_tf_dataset_for_client(cid)
            loss, count, max_epoch = stackoverflow_lr_train(net, train_data_, test_data_)


if __name__  == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help = 'Enter the dataset you want to train your algorithm on.')
    args = parser.parse_args()

    DATASET = args.dataset
    METHOD = 'Local'

    if not (DATASET in ['stackoverflow_nwp', 'stackoverflow_lr', 'synthetic', 'emnist', 'cifar100', 'cifar10', 'shakespeare']):
        print('Dataset not recognized, try again!')
        sys.exit()    
    hyperparameters = eval(DATASET + '_' + METHOD.lower()) 
    
    num_clients = hyperparameters['num_clients']
    total_clients = hyperparameters['total_clients']
    rounds = hyperparameters['rounds']
    total_rounds = hyperparameters['rounds']
    epochs = hyperparameters['epochs']
    checkpoint_interval = hyperparameters['checkpoint_interval']
    checkpoint_dir = hyperparameters['checkpoint_dir']
    dataset_dir = hyperparameters['dataset_dir']
    batch_size = hyperparameters['batch_size']
    lr = hyperparameters['lr']
    

    if DATASET == 'stackoverflow_nwp':
        vocab_size = hyperparameters['vocab_size']
        embedding_size = hyperparameters['embedding_size']
        hidden_size = hyperparameters['hidden_size']
        num_layers = hyperparameters['num_layers']
        sequence_length = hyperparameters['sequence_length']
        train_data, test_data = stackoverflow_nwp_dataloader.get_federated_datasets(vocab_size, sequence_length, train_client_batch_size=batch_size)
        with open(dataset_dir + 'available_clients.txt') as f:
            clients = f.readlines()[:total_rounds * num_clients]

        client_object = StackoverflowNWPClient()
        client_object.fit()

        with open(checkpoint_dir + 'intermediate_'+METHOD+ '_' + str(num_clients) + '_' + str(total_clients) + '_' + str(total_rounds) + '_' + str(epochs) + '_' + str(lr).replace('.', '_') + '_test_accuracies.txt', 'r') as f:
            sum_accuracy = 0.0
            accuracies = [float(line.rstrip()) for line in f]
            print("Stadard Deviation: ", np.std(accuracies, axis=0))
            print("Average accuracy: ", sum(accuracies) / len(accuracies))

    elif DATASET == 'synthetic':
        input_dim = hyperparameters['input_dim']
        hidden_dim = hyperparameters['hidden_dim']
        output_dim = hyperparameters['output_dim']
        with open(dataset_dir + 'train_0_5_0_5_0.json') as f:
            synthetic_data = json.load(f)
            train_data = synthetic_data["user_data"]
        with open(dataset_dir + 'test_0_5_0_5_0.json') as f:
            synthetic_data = json.load(f)
            test_data = synthetic_data["user_data"]

        clients = list(synthetic_data["users"])[:min(total_rounds * num_clients, total_clients)]
        
        client_object = SyntheticClient()
        client_object.fit()

        with open(checkpoint_dir + 'intermediate_'+METHOD+ '_' + str(num_clients) + '_' + str(total_clients) + '_' + str(total_rounds) + '_' + str(epochs) + '_' + str(lr).replace('.', '_') + '_test_accuracies.txt', 'r') as f:
            sum_accuracy = 0.0
            accuracies = [float(line.rstrip()) for line in f]
            print("Stadard Deviation: ", np.std(accuracies, axis=0))
            print("Average accuracy: ", sum(accuracies) / len(accuracies))

    elif DATASET == 'emnist':
        train_data, test_data = emnist_dataloader.get_federated_datasets(train_client_batch_size=batch_size)
        with open(dataset_dir + 'available_train_clients.txt', 'r') as f:
            clients = f.readlines()[:min(total_clients, num_clients * total_rounds)]

        client_object = EMNISTClient()
        client_object.fit()

        with open(checkpoint_dir + 'intermediate_'+METHOD+ '_' + str(num_clients) + '_' + str(total_clients) + '_' + str(total_rounds) + '_' + str(epochs) + '_' + str(lr).replace('.', '_') + '_test_accuracies.txt', 'r') as f:
            sum_accuracy = 0.0
            accuracies = [float(line.rstrip()) for line in f]
            print("Stadard Deviation: ", np.std(accuracies, axis=0))
            print("Average accuracy: ", sum(accuracies) / len(accuracies))

    elif DATASET == 'cifar100':
        dirichlet_parameter = hyperparameters["dirichlet_parameter"]
        train_data, test_data = cifar100_dataloader.get_federated_datasets(dirichlet_parameter=dirichlet_parameter, total_clients=total_clients,)
        clients = range(total_clients)[:min(total_clients, num_clients * total_rounds)]

        client_object = Cifar100Client()
        client_object.fit()

        with open(checkpoint_dir + 'intermediate_'+METHOD+ '_' + str(num_clients) + '_' + str(total_clients) + '_' + str(total_rounds) + '_' + str(epochs) + '_' + str(lr).replace('.', '_') + '_test_accuracies.txt', 'r') as f:
            sum_accuracy = 0.0
            accuracies = [float(line.rstrip()) for line in f]
            print("Stadard Deviation: ", np.std(accuracies, axis=0))
            print("Average accuracy: ", sum(accuracies) / len(accuracies))

    elif DATASET == 'shakespeare':
        embedding_size = hyperparameters['embedding_size']
        hidden_size = hyperparameters['hidden_size']
        vocab_size = hyperparameters['vocab_size']
        train_data, test_data = shakespeare_dataloader.get_federated_datasets()
        clients = range(total_clients)[:min(total_clients, num_clients * total_rounds)]

        client_object = ShakespeareClient()
        client_object.fit()

        with open(checkpoint_dir + 'intermediate_'+METHOD+ '_' + str(num_clients) + '_' + str(total_clients) + '_' + str(total_rounds) + '_' + str(epochs) + '_' + str(lr).replace('.', '_') + '_test_accuracies.txt', 'r') as f:
            sum_accuracy = 0.0
            accuracies = [float(line.rstrip()) for line in f]
            print("Stadard Deviation: ", np.std(accuracies, axis=0))
            print("Average accuracy: ", sum(accuracies) / len(accuracies))

    elif DATASET == 'stackoverflow_lr':
        word_vocab_size = hyperparameters['word_vocab_size']
        tag_vocab_size = hyperparameters['tag_vocab_size']
        train_data, test_data = stackoverflow_lr_dataloader.get_federated_datasets(word_vocab_size, tag_vocab_size, train_client_batch_size=batch_size)
        with open(dataset_dir + 'available_clients.txt') as f:
            clients = f.readlines()[:min(total_clients, num_clients * total_rounds)]

        client_object = StackoverflowLRClient()
        client_object.fit()

        with open(checkpoint_dir + 'intermediate_'+METHOD+ '_' + str(num_clients) + '_' + str(total_clients) + '_' + str(total_rounds) + '_' + str(epochs) + '_' + str(lr).replace('.', '_') + '_test_accuracies.txt', 'r') as f:
            sum_accuracy = 0.0
            accuracies = [float(line.rstrip()) for line in f]
            print("Stadard Deviation: ", np.std(accuracies, axis=0))
            print("Average accuracy: ", sum(accuracies) / len(accuracies))

    elif DATASET == 'cifar10':
        dirichlet_parameter = hyperparameters["dirichlet_parameter"]
        train_data, test_data = cifar10_dataloader.get_federated_datasets(dirichlet_parameter=dirichlet_parameter, total_clients=total_clients,)
        clients = range(total_clients)[:min(total_clients, num_clients * total_rounds)]

        client_object = Cifar10Client()
        client_object.fit()

        with open(checkpoint_dir + 'intermediate_'+METHOD+ '_' + str(num_clients) + '_' + str(total_clients) + '_' + str(total_rounds) + '_' + str(epochs) + '_' + str(lr).replace('.', '_') + '_test_accuracies.txt', 'r') as f:
            sum_accuracy = 0.0
            accuracies = [float(line.rstrip()) for line in f]
            print("Stadard Deviation: ", np.std(accuracies, axis=0))
            print("Average accuracy: ", sum(accuracies) / len(accuracies))

    # initial_parameters, rounds = load_parameters_from_disk()

    print(Fore.RED + "Availble Device: " + str(DEVICE) + ", Count: " + str(AVAILABLE_GPUS) + Fore.WHITE)

    ## Stackoverflow Test - Single Client
    # train_data_ = train_data.create_tf_dataset_for_client('06580021')
    # test_data_ = test_data.create_tf_dataset_for_client('06580021')
    # net = StackoverflowNet(vocab_size + 4, embedding_size, hidden_size, num_layers).to(DEVICE)
    # # client = StackoverflowClient('06580021', net)
    # # accuracy, count = client.fit()
    # accuracy, count = stackoverflow_train(net, train_data_)
    # print("Accuracy: ", accuracy, ", Count: ", count)
    # # loss, accuracy, correct, count = client.evaluate()
    # loss, accuracy, correct, count = stackoverflow_test(net, test_data_)
    # print("Loss: ", loss, ", Accuracy: ", accuracy, ", Count: ", count)
   
    ## Movielens Test - Single Client
    # user_id = 123
    # train_data_ = movielens_dataloader.create_tf_dataset_for_user(ratings, 1, personal_model=True, batch_size=hyperparameters['batch_size'])
    # test_data_ = movielens_dataloader.create_tf_dataset_for_user(ratings, 3, personal_model=True, batch_size=hyperparameters['batch_size'])
    # train_data_ = tf_train_datasets[user_id]
    # test_data_ = tf_test_datasets[user_id]
    # net = MovielensCF(latent_dimension, total_users, total_items).to(DEVICE)
    # loss, count = movielens_train(net, torch.tensor(user_id).to(DEVICE), train_data_)
    # print("Loss: ", loss, ", Count: ", count)
    # loss, accuracy, correct, count = movielens_test(net, torch.tensor(user_id).to(DEVICE), test_data_)
    # print("Loss: ", loss, ", Accuracy: ", accuracy, ", Count: ", count)

    ## Synthetic Test - Single Client
    # user_id = 'f_00000'
    # train_data_ = train_data[user_id]
    # test_data_ = test_data[user_id]
    # net = SyntheticNet(input_dim, hidden_dim, output_dim).to(DEVICE)    
    # loss, count = synthetic_train(net, train_data_)
    # print("Loss: ", loss, ", Count: ", count)
    # loss, accuracy, correct, count = synthetic_test(net, test_data_)
    # print("Loss: ", loss, ", Accuracy: ", accuracy, ", Count: ", count)

    ## Amazon Test - Single Client
    # data_ = amazon_dataloader_object.create_tf_dataset_for_user(clients[2].strip())
    # net = AmazonNCF(word_embedding_size, user_embedding_size, item_embedding_size, vocab_size, total_clients, num_items).to(DEVICE)
    # accuracy, count = amazon_train(net, data_, data_, torch.tensor(1).to(DEVICE))
    # print("Accuracy: ", accuracy, ", Count: ", count)
    # loss, accuracy, correct, count = amazon_test(net, data_, torch.tensor(1).to(DEVICE))
    # print("Loss: ", loss, ", Accuracy: ", accuracy, ", Count: ", count)
    
    ## EMNIST Test - Single Client
    # train_data_ = train_data.create_tf_dataset_for_client('f0000_14')
    # test_data_ = test_data.create_tf_dataset_for_client('f0000_14')
    # net = EMNISTNet().to(DEVICE)
    # emnist_generalized_test(net, train_data_, test_data_)

    ## Cifar100 Test - Single Client
    # train_data_ = train_data.create_tf_dataset_for_client('99')
    # test_data_ = test_data.create_tf_dataset_for_client('99')
    # net = CifarResNet().to(DEVICE)
    # print(cifar100_generalized_test(net, train_data_, test_data_))

    ## Stackoverflow LR Test - Single Client
    # train_data_ = train_data.create_tf_dataset_for_client('06580021')
    # test_data_ = test_data.create_tf_dataset_for_client('06580021')
    # net = StackoverflowLR(word_vocab_size, tag_vocab_size).to(DEVICE)
    # stackoverflow_lr_train(net, train_data_, test_data_)

    ## Cifar10 Test - Single client
    # train_data_ = train_data.create_tf_dataset_for_client('99')
    # test_data_ = test_data.create_tf_dataset_for_client('94')
    # net = Cifar10ResNet().to(DEVICE)
    # cifar10_generalized_test(net, train_data_, test_data_)


