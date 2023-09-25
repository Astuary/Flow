#!/usr/bin/python
# -*- coding: utf-8 -*-
# src: https://github.com/FedML-AI/FedML/blob/ecd2d81222301d315ca3a84be5a5ce4f33d6181c/python/fedml/model/nlp/rnn.py

import torch
import torch.nn as nn


class ShakespeareNet(nn.Module):

    """Creates a RNN model using LSTM layers for Shakespeare language models (next character prediction task).
    This replicates the model structure in the paper:
    Communication-Efficient Learning of Deep Networks from Decentralized Data
      H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Agueray Arcas. AISTATS 2017.
      https://arxiv.org/abs/1602.05629
    This is also recommended model by "Adaptive Federated Optimization. ICML 2020" (https://arxiv.org/pdf/2003.00295.pdf)
    Args:
      vocab_size: the size of the vocabulary, used as a dimension in the input embedding.
      sequence_length: the length of input sequences.
    Returns:
      An uncompiled `torch.nn.Module`.
    """

    def __init__(
        self,
        embedding_dim=8,
        vocab_size=90,
        hidden_size=256,
        ):
        super(ShakespeareNet, self).__init__()
		
        self.hidden_size = hidden_size
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)

        # self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=2, batch_first=True,)

        self.global_lstm_1 = nn.LSTMCell(input_size=embedding_dim, hidden_size=hidden_size)
        self.global_lstm_2 = nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)

        self.local_lstm_1 = nn.LSTMCell(input_size=embedding_dim, hidden_size=hidden_size)
        self.local_lstm_2 = nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)

        self.global_fc = nn.Linear(hidden_size, vocab_size)
        self.local_fc = nn.Linear(hidden_size, vocab_size)

        self.prob_linear_1 = nn.Sequential(nn.Linear(embedding_dim + 4
                * hidden_size, 2 * hidden_size), nn.ReLU(True),
                nn.Linear(2 * hidden_size, hidden_size), nn.ReLU(True),
                nn.Linear(hidden_size, 2))

        self.prob_linear_2 = nn.Sequential(nn.Linear(embedding_dim + 4
                * hidden_size, 2 * hidden_size), nn.ReLU(True),
                nn.Linear(2 * hidden_size, hidden_size), nn.ReLU(True),
                nn.Linear(hidden_size, 2))

    def initialize(self, batch_size, cell_size):
        init_cell = torch.Tensor(batch_size, cell_size).zero_()
        if torch.cuda.is_available():
            init_cell = init_cell.cuda()
        return init_cell

    def forward(self, input_seq, local=False):
        embeds = self.embeddings(input_seq)

        (batch_size, sequence_length) = input_seq.shape
        local_hidden_1 = self.initialize(batch_size, self.hidden_size)
        local_hidden_2 = self.initialize(batch_size, self.hidden_size)
        local_cell_1 = self.initialize(batch_size, self.hidden_size)
        local_cell_2 = self.initialize(batch_size, self.hidden_size)

        # probs_hidden = None
        hidden_states = None

        if local:
            for i in range(sequence_length):
                embedding = embeds[:, i, :]
                (local_hidden_1, local_cell_1) = self.local_lstm_1(embedding, (local_hidden_1, local_cell_1))
                (local_hidden_2, local_cell_2) = self.local_lstm_2(local_hidden_1, (local_hidden_2, local_cell_2))
                if hidden_states == None:
                    hidden_states = local_hidden_2[:, None, :]
                else:
                    hidden_states = torch.cat([hidden_states, local_hidden_2[:, None, :]], dim=1)
        else:

            embedding_dim = embeds.shape[2]

            global_hidden_1 = self.initialize(batch_size, self.hidden_size)
            global_hidden_2 = self.initialize(batch_size, self.hidden_size)
            global_cell_1 = self.initialize(batch_size, self.hidden_size)
            global_cell_2 = self.initialize(batch_size, self.hidden_size)

            for i in range(sequence_length):
                embedding = embeds[:, i, :] 
                
                global_hidden_1, global_cell_1 = self.global_lstm_1(embedding, (global_hidden_1, global_cell_1))
                local_hidden_1, local_cell_1 = self.local_lstm_1(embedding, (local_hidden_1, local_cell_1))

                probabilities_hidden_1 = torch.softmax(self.prob_linear_1(torch.cat([embedding.contiguous().view(-1, embedding_dim), global_hidden_1, global_cell_1, local_hidden_1, local_cell_1], 1)), dim=1).unsqueeze(1)
                
                hidden_tilde = torch.cat([local_hidden_1[:, None, :],
                                        global_hidden_1[:, None, :]],
                                        dim=1)
                cell_tilde = torch.cat([local_cell_1[:, None, :],
                                        global_cell_1[:, None, :]],
                                        dim=1)
               
                global_hidden_1 = torch.bmm(probabilities_hidden_1, hidden_tilde).squeeze()
                global_cell_1 = torch.bmm(probabilities_hidden_1, cell_tilde).squeeze()

                global_hidden_2, global_cell_2 = self.global_lstm_2(global_hidden_1, (global_hidden_2, global_cell_2))
                local_hidden_2, local_cell_2 = self.local_lstm_2(local_hidden_1, (local_hidden_2, local_cell_2))
        
                probabilities_hidden_2 = torch.softmax(self.prob_linear_2(torch.cat([embedding.contiguous().view(-1, embedding_dim), global_hidden_2, global_cell_2, local_hidden_2, local_cell_2], 1)), dim=1).unsqueeze(1)

                hidden_tilde = torch.cat([local_hidden_2[:, None, :],
                                        global_hidden_2[:, None, :]],
                                        dim=1)
                cell_tilde = torch.cat([local_cell_2[:, None, :],
                                        global_cell_2[:, None, :]],
                                        dim=1)
        
                global_hidden_2 = torch.bmm(probabilities_hidden_2, hidden_tilde).squeeze()
                global_cell_2 = torch.bmm(probabilities_hidden_2, cell_tilde).squeeze()

                if hidden_states == None:
                    # probs_hidden = probabilities_hidden_2[:, None, :]
                    hidden_states = global_hidden_2[:, None, :]
                else:
                    # probs_hidden = torch.cat([probs_hidden, probabilities_hidden_2[:, None, :]], dim=1)
                    hidden_states = torch.cat([hidden_states, global_hidden_2[:, None, :]], dim=1)

        global_output = self.global_fc(hidden_states)
        local_output = self.local_fc(hidden_states)

        output = 0.5 * global_output + 0.5 * local_output
        output = torch.transpose(output, 1, 2)
        return output
