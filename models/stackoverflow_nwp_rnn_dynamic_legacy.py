import torch
import torch.nn as nn

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class StackoverflowNet(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, sequence_length, num_layers):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        
        self.global_lstm = nn.LSTMCell(input_size=embedding_size, hidden_size=hidden_size)
        self.local_lstm = nn.LSTMCell(input_size=embedding_size, hidden_size=hidden_size)
        
        self.global_linear_1 = nn.Linear(hidden_size, embedding_size)
        self.local_linear_1 = nn.Linear(hidden_size, embedding_size)
        
        self.global_linear_2 = nn.Linear(embedding_size, vocab_size)
        self.local_linear_2 = nn.Linear(embedding_size, vocab_size)

        self.prob_lstm = nn.Linear(embedding_size + 2 * hidden_size, 2)
        self.prob_linear_1 = nn.Linear(embedding_size * sequence_length * 2, 2)
        self.prob_linear_2 = nn.Linear(vocab_size * sequence_length * 2, 2)
        # self.prob_linear_1 = nn.Linear(embedding_size, 2)
        # self.prob_linear_2 = nn.Linear(embedding_size, 2)

    def initialize(self, batch_size, cell_size):
        init_cell =  torch.Tensor(batch_size, cell_size).zero_()
        if torch.cuda.is_available():
            init_cell = init_cell.cuda()
        return init_cell

    def forward(self, x, local = False):
        # self.local_lstm.flatten_parameters()
        embeddings = self.embedding(x)

        batch_size, sequence_length = x.shape

        local_hidden = self.initialize(batch_size, self.hidden_size)
        local_cell = self.initialize(batch_size, self.hidden_size)

        probs_hidden = None
        hidden_states = None

        if local:  
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

        else:
            global_hidden = self.initialize(batch_size, self.hidden_size)
            global_cell = self.initialize(batch_size, self.hidden_size)

            for i in range(sequence_length):
                embedding = embeddings[:, i, :] 
                global_hidden, global_cell = self.global_lstm(embedding, (global_hidden, global_cell))
                local_hidden, local_cell = self.local_lstm(embedding, (local_hidden, local_cell))

                probabilities = torch.softmax(self.prob_lstm(torch.cat([embedding.contiguous().view(-1, self.embedding_size), global_hidden, local_hidden], 1)), dim = 1).unsqueeze(1)
                hidden_tilde = torch.cat([local_hidden[:, None, :], global_hidden[:, None, :]], dim=1)
                cell_tilde = torch.cat([local_cell[:, None, :], global_cell[:, None, :]], dim=1)
                # # print(probabilities.shape)
                # # print(hidden_tilde.shape)
                global_hidden = torch.bmm(probabilities, hidden_tilde).squeeze(dim=1)
                global_cell = torch.bmm(probabilities, cell_tilde).squeeze(dim=1)
        
                if hidden_states == None:
                    # probs_hidden = probabilities[:, None, :]
                    hidden_states = global_hidden[:, None, :]
                else:
                    # probs_hidden = torch.cat([probs_hidden, probabilities[:, None, :]], dim=1)
                    hidden_states = torch.cat([hidden_states, global_hidden[:, None, :]], dim=1)

            global_x = self.global_linear_1(hidden_states[:,:])
            local_x = self.local_linear_1(hidden_states[:,:])
            
            stacked_flattened_x = torch.cat([torch.flatten(global_x, start_dim=1), torch.flatten(local_x, start_dim=1)], dim=1)
            probabilities = torch.softmax(self.prob_linear_1(stacked_flattened_x), dim=1)
            x = probabilities[:, 0].unsqueeze(1).unsqueeze(2).expand_as(global_x) * global_x+ probabilities[:, 1].unsqueeze(1).unsqueeze(2).expand_as(local_x) * local_x
            
            # probabilities = torch.softmax(self.prob_linear_1(embeddings), dim=2)
            # x =  global_x * probabilities[:, :, 0].unsqueeze(2).expand_as(global_x) + local_x * probabilities[:, :, 1].unsqueeze(2).expand_as(local_x)
            
            # x =  global_x * 0.5 + local_x * 0.5

            global_logits = self.global_linear_2(x)
            local_logits = self.local_linear_2(x)
            
            stacked_flattened_logits = torch.cat([torch.flatten(global_logits, start_dim=1), torch.flatten(local_logits, start_dim=1)], dim=1)
            probabilities = torch.softmax(self.prob_linear_2(stacked_flattened_logits), dim=1)
            logits =  global_logits * probabilities[:, 0].unsqueeze(1).unsqueeze(2).expand_as(global_logits) + local_logits * probabilities[:, 1].unsqueeze(1).unsqueeze(2).expand_as(local_logits)
            
            # probabilities = torch.softmax(self.prob_linear_2(embeddings), dim=2)
            # logits =  global_logits * probabilities[:, :, 0].unsqueeze(2).expand_as(global_logits) + local_logits * probabilities[:, :, 1].unsqueeze(2).expand_as(local_logits)
            
            # logits =  global_logits * 0.5 + local_logits * 0.5

            output = torch.transpose(logits, 1, 2)

        return output

# net = StackoverflowNet(10004, 96, 670, 1)
# pytorch_total_params = sum(p.numel() for p in net.parameters())
# print(pytorch_total_params)