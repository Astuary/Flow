import torch
import torch.nn as nn
import torch.nn.functional as F

class StackoverflowNet(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, sequence_length, num_layers):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.policy_net = StackoverflowPolicyNet(vocab_size, embedding_size, hidden_size, sequence_length)
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        
        self.global_lstm = nn.LSTMCell(input_size=embedding_size, hidden_size=hidden_size)
        self.local_lstm = nn.LSTMCell(input_size=embedding_size, hidden_size=hidden_size)
        
        self.global_linear_1 = nn.Linear(hidden_size, embedding_size)
        self.local_linear_1 = nn.Linear(hidden_size, embedding_size)
        
        self.global_linear_2 = nn.Linear(embedding_size, vocab_size)
        self.local_linear_2 = nn.Linear(embedding_size, vocab_size)

        # self.prob_linear_1 = nn.Linear(embedding_size, 2)
        # self.prob_linear_2 = nn.Linear(embedding_size, 2)

    def initialize(self, batch_size, cell_size):
        init_cell =  torch.Tensor(batch_size, cell_size).zero_()
        if torch.cuda.is_available():
            init_cell = init_cell.cuda()
        return init_cell

    def forward(self, x, mode = 'personalized', hard_decision = False):
        # self.local_lstm.flatten_parameters()
        embeddings = self.embedding(x)

        batch_size, sequence_length = x.shape

        local_hidden = self.initialize(batch_size, self.hidden_size)
        local_cell = self.initialize(batch_size, self.hidden_size)
        
        global_hidden = self.initialize(batch_size, self.hidden_size)
        global_cell = self.initialize(batch_size, self.hidden_size)

        probs_hidden = None
        hidden_states = None

        if mode == 'local':  
            for i in range(sequence_length):
                embedding = embeddings[:, i, :]
                local_hidden, local_cell = self.local_lstm(embedding, (local_hidden, local_cell))
                
                if hidden_states == None:
                    hidden_states = local_hidden[:, None, :]
                else:
                    hidden_states = torch.cat([hidden_states, local_hidden[:, None, :]], dim=1)

                local_x = self.local_linear_1(hidden_states[:,:])
                local_logits = self.local_linear_2(local_x)
                output = torch.transpose(local_logits, 1, 2)
        
        elif mode == 'global':  
            for i in range(sequence_length):
                embedding = embeddings[:, i, :]
                global_hidden, global_cell = self.global_lstm(embedding, (global_hidden, global_cell))
                
                if hidden_states == None:
                    hidden_states = global_hidden[:, None, :]
                else:
                    hidden_states = torch.cat([hidden_states, global_hidden[:, None, :]], dim=1)

                global_x = self.global_linear_1(hidden_states[:,:])
                global_logits = self.global_linear_2(global_x)
                output = torch.transpose(global_logits, 1, 2)

        elif mode == 'personalized':
            global_hidden = self.initialize(batch_size, self.hidden_size)
            global_cell = self.initialize(batch_size, self.hidden_size)

            for i in range(sequence_length):
                embedding = embeddings[:, i, :] 
                global_hidden, global_cell = self.global_lstm(embedding, (global_hidden, global_cell))
                local_hidden, local_cell = self.local_lstm(embedding, (local_hidden, local_cell))

                probabilities = torch.softmax(self.policy_net(torch.cat([embedding.contiguous().view(-1, self.embedding_size), global_hidden, local_hidden], 1), lstm = True), dim = 1).unsqueeze(1)
                if hard_decision:
                    probabilities = torch.round(probabilities)
                
                hidden_tilde = torch.cat([local_hidden[:, None, :], global_hidden[:, None, :]], dim=1)
                cell_tilde = torch.cat([local_cell[:, None, :], global_cell[:, None, :]], dim=1)
                # # print(hidden_tilde.shape)
                global_hidden = torch.bmm(probabilities, hidden_tilde).squeeze(dim=1)
                global_cell = torch.bmm(probabilities, cell_tilde).squeeze(dim=1)
        
                if hidden_states == None:
                    probs_hidden = probabilities[:, :, None, :]
                    hidden_states = global_hidden[:, None, :]
                else:
                    probs_hidden = torch.cat([probs_hidden, probabilities[:, :, None, :]], dim=1)
                    hidden_states = torch.cat([hidden_states, global_hidden[:, None, :]], dim=1)

            global_x = self.global_linear_1(hidden_states[:,:])
            local_x = self.local_linear_1(hidden_states[:,:])
            
            stacked_flattened_x = torch.flatten(torch.cat([global_x[:, :, None, :], local_x[:, :, None, :]], dim=2), start_dim = 2)
            # stacked_flattened_x = torch.cat([torch.flatten(global_x, start_dim=1), torch.flatten(local_x, start_dim=1)], dim=1)
            p1, p2 = self.policy_net(stacked_flattened_x, lstm = False)
            
            probabilities = torch.softmax(p1, dim=2)
            if hard_decision:
                probabilities = torch.round(probabilities)
            
            probs_hidden = torch.cat([probs_hidden.repeat(1,1,20,1), probabilities[:, None, :, :]], dim=1)
            x = probabilities[:, :, 0].unsqueeze(2).expand_as(global_x) * global_x + probabilities[:, :, 1].unsqueeze(2).expand_as(local_x) * local_x
            
            global_logits = self.global_linear_2(x)
            local_logits = self.local_linear_2(x)
            
            probabilities = torch.softmax(p2, dim=2)
            if hard_decision:
                probabilities = torch.round(probabilities)

            probs_hidden = torch.cat([probs_hidden, probabilities[:, None, : ,:]], dim=1)

            logits =  global_logits * probabilities[:, :, 0].unsqueeze(2).expand_as(global_logits) + local_logits * probabilities[:, :, 1].unsqueeze(2).expand_as(local_logits)
            
            
            output = torch.transpose(logits, 1, 2)

        return output, probs_hidden

# net = StackoverflowNet(10004, 96, 670, 1)
# pytorch_total_params = sum(p.numel() for p in net.parameters())
# print(pytorch_total_params)

class StackoverflowPolicyNet(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, sequence_length) -> None:
        super().__init__()

        self.lstm_linear_1 = nn.Linear(embedding_size + 2 * hidden_size, 500)
        self.lstm_linear_2 = nn.Linear(500, 250)
        self.lstm_linear_3 = nn.Linear(250, 100)
        self.lstm_linear_4 = nn.Linear(100, 50)
        self.lstm_linear_exit = nn.Linear(50, 2)

        self.linear_1 = nn.Linear(192, 75)
        self.linear_2 = nn.Linear(75, 30)
        self.linear_2_exit = nn.Linear(30, 2)
        self.linear_3 = nn.Linear(30, 10)
        self.linear_3_exit = nn.Linear(10, 2)

    def forward(self, x, lstm = True):
        
        if lstm:
            x = F.relu(self.lstm_linear_1(x))
            x = F.dropout(x, p = 0.3)
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
            x = F.relu(self.linear_2(x))
            x = F.dropout(x, p = 0.3)
            y = F.relu(self.linear_2_exit(x))

            x = F.relu(self.linear_3(x))
            x = F.dropout(x, p = 0.3)
            z = F.relu(self.linear_3_exit(x))

            return y, z