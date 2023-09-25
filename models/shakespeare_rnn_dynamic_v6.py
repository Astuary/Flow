# src: https://github.com/FedML-AI/FedML/blob/ecd2d81222301d315ca3a84be5a5ce4f33d6181c/python/fedml/model/nlp/rnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def __init__(self, embedding_dim=8, vocab_size=90, hidden_size=256):
        super(ShakespeareNet, self).__init__()
        
        self.policy_net = ShakespearePolicyNet(embedding_dim, vocab_size, hidden_size)

        self.hidden_size = hidden_size
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        # self.lstm = nn.LSTM(
        #     input_size=embedding_dim,
        #     hidden_size=hidden_size,
        #     num_layers=2,
        #     batch_first=True,
        # )
        # self.fc = nn.Linear(hidden_size, vocab_size)
        self.client_embedding = nn.Embedding(715, 10)

        self.global_lstm_1 = nn.LSTMCell(input_size=embedding_dim, hidden_size=hidden_size)
        self.global_lstm_2 = nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)

        self.local_lstm_1 = nn.LSTMCell(input_size=embedding_dim, hidden_size=hidden_size)
        self.local_lstm_2 = nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)

        self.global_fc = nn.Linear(hidden_size, vocab_size)
        self.local_fc = nn.Linear(hidden_size, vocab_size)

    def initialize(self, batch_size, cell_size):
        init_cell = torch.Tensor(batch_size, cell_size).zero_()
        if torch.cuda.is_available():
            init_cell = init_cell.cuda()
        return init_cell

    def forward(self, input_seq, cid, mode = 'personalized', hard_decision = False):
        embeds = self.embeddings(input_seq)
        
        # Note that the order of mini-batch is random so there is no hidden relationship among batches.
        # So we do not input the previous batch's hidden state,
        # leaving the first hidden state zero `self.lstm(embeds, None)`.
        # lstm_out, _ = self.lstm(embeds)
        # # use the final hidden state as the next character prediction
        # final_hidden_state = lstm_out[:, -1]
        # # output = self.fc(final_hidden_state)
        # # For fed_shakespeare
        # output = self.fc(lstm_out[:,:])
        # output = torch.transpose(output, 1, 2)
        # return output

        (batch_size, sequence_length) = input_seq.shape

        local_hidden_1 = self.initialize(batch_size, self.hidden_size)
        local_hidden_2 = self.initialize(batch_size, self.hidden_size)
        local_cell_1 = self.initialize(batch_size, self.hidden_size)
        local_cell_2 = self.initialize(batch_size, self.hidden_size)

        global_hidden_1 = self.initialize(batch_size, self.hidden_size)
        global_hidden_2 = self.initialize(batch_size, self.hidden_size)
        global_cell_1 = self.initialize(batch_size, self.hidden_size)
        global_cell_2 = self.initialize(batch_size, self.hidden_size)

        probs_hidden = None
        hidden_states = None
        
        if mode == 'local':
            for i in range(sequence_length):
                embedding = embeds[:, i, :]

                (local_hidden_1, local_cell_1) = self.local_lstm_1(embedding, (local_hidden_1, local_cell_1))
                (local_hidden_2, local_cell_2) = self.local_lstm_2(local_hidden_1, (local_hidden_2, local_cell_2))
                
                if hidden_states == None:
                    hidden_states = local_hidden_2[:, None, :]
                else:
                    hidden_states = torch.cat([hidden_states, local_hidden_2[:, None, :]], dim=1)
             
            output = self.local_fc(hidden_states[:, :])

        elif mode == 'global':
            for i in range(sequence_length):
                embedding = embeds[:, i, :]

                (global_hidden_1, global_cell_1) = self.global_lstm_1(embedding, (global_hidden_1, global_cell_1))
                (global_hidden_2, global_cell_2) = self.global_lstm_2(global_hidden_1, (global_hidden_2, global_cell_2))
                
                if hidden_states == None:
                    hidden_states = global_hidden_2[:, None, :]
                else:
                    hidden_states = torch.cat([hidden_states, global_hidden_2[:, None, :]], dim=1)
             
            output = self.global_fc(hidden_states[:, :])

        elif mode == 'personalized':
            embedding_dim = embeds.shape[2]

            cid_emb = self.client_embedding(cid)

            for i in range(sequence_length):
                embedding = embeds[:, i, :] 
                
                global_hidden_1, global_cell_1 = self.global_lstm_1(embedding, (global_hidden_1, global_cell_1))
                local_hidden_1, local_cell_1 = self.local_lstm_1(embedding, (local_hidden_1, local_cell_1))
                
                probabilities_hidden_1 = torch.softmax(self.policy_net(cid_emb, lstm = True), dim=2)#.unsqueeze(1)
                
                if hard_decision:
                    probabilities_hidden_1 = torch.round(probabilities_hidden_1)

                hidden_tilde = torch.cat([local_hidden_1[:, None, :],
                                        global_hidden_1[:, None, :]],
                                        dim=1)
                cell_tilde = torch.cat([local_cell_1[:, None, :],
                                        global_cell_1[:, None, :]],
                                        dim=1)
               
                global_hidden_1 = torch.bmm(probabilities_hidden_1, hidden_tilde).squeeze(dim = 1)
                global_cell_1 = torch.bmm(probabilities_hidden_1, cell_tilde).squeeze(dim = 1)

                global_hidden_2, global_cell_2 = self.global_lstm_2(global_hidden_1, (global_hidden_2, global_cell_2))
                local_hidden_2, local_cell_2 = self.local_lstm_2(local_hidden_1, (local_hidden_2, local_cell_2))
        
                probabilities_hidden_2 = torch.softmax(self.policy_net(cid_emb, lstm = True), dim=2)#.unsqueeze(1)

                if hard_decision:
                    probabilities_hidden_2 = torch.round(probabilities_hidden_2)
                
                hidden_tilde = torch.cat([local_hidden_2[:, None, :],
                                        global_hidden_2[:, None, :]],
                                        dim=1)
                cell_tilde = torch.cat([local_cell_2[:, None, :],
                                        global_cell_2[:, None, :]],
                                        dim=1)
        
                global_hidden_2 = torch.bmm(probabilities_hidden_2, hidden_tilde).squeeze(dim = 1)
                global_cell_2 = torch.bmm(probabilities_hidden_2, cell_tilde).squeeze(dim = 1)

                if hidden_states == None:
                    probs_hidden = probabilities_hidden_2#[:, None, :]
                    hidden_states = global_hidden_2[:, None, :]
                else:
                    probs_hidden = torch.cat([probs_hidden, probabilities_hidden_2], dim=1)
                    hidden_states = torch.cat([hidden_states, global_hidden_2[:, None, :]], dim=1)
            
            output = self.global_fc(hidden_states)

        output = torch.transpose(output, 1, 2)
        return output, probs_hidden

class ShakespearePolicyNet(nn.Module):

    def __init__(self, embedding_size, vocab_size, hidden_size) -> None:
        super().__init__()

        self.lstm_linear_2 = nn.Linear(10, 100)
        self.lstm_linear_3 = nn.Linear(100, 50)
        self.lstm_linear_4 = nn.Linear(50, 25)
        self.lstm_linear_exit = nn.Linear(25, 2)

        self.linear_1 = nn.Linear(10, 25)
        self.linear_1_exit = nn.Linear(25, 2)

    def forward(self, x, lstm = False):
        
        if lstm:
            # x = F.relu(self.lstm_linear_1(x))
            # x = F.dropout(x, p = 0.3)
            x = F.relu(self.lstm_linear_2(x))
            x = F.dropout(x, p = 0.3)
            x = F.relu(self.lstm_linear_3(x))
            x = F.dropout(x, p = 0.3)
            x = F.relu(self.lstm_linear_4(x))
            x = F.dropout(x, p = 0.3)
            x = self.lstm_linear_exit(x)

            return x

        else:
            x = F.relu(self.linear_1(x))
            x = F.dropout(x, p = 0.3)
            y = F.relu(self.linear_1_exit(x))

            return y