import torch
import torch.nn as nn
import torch.nn.functional as F

class StackoverflowNet(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers):
        super().__init__()
        self.global_embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.local_embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)

        self.global_lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers)
        self.local_lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers)
        
        self.global_linear_1 = nn.Linear(hidden_size, embedding_size)
        self.local_linear_1 = nn.Linear(hidden_size, embedding_size)

        self.global_linear_2 = nn.Linear(embedding_size, vocab_size)
        self.local_linear_2 = nn.Linear(embedding_size, vocab_size)

    def forward(self, x, alpha):
        self.global_lstm.flatten_parameters()
        self.local_lstm.flatten_parameters()

        alpha = F.gumbel_softmax(alpha)
        A = torch.argmax(alpha, dim=1)

        global_embeddings = self.global_embedding(x)
        local_embeddings = self.local_embedding(x)

        embeddings = A[0] * global_embeddings + (1 - A[0]) * local_embeddings

        global_x, _ = self.global_lstm(embeddings)
        local_x, _ = self.local_lstm(embeddings)
        
        x = A[1] * global_x + (1 - A[1]) * local_x

        global_x = self.global_linear_1(x[:, :])
        local_x = self.local_linear_1(x[:, :])

        x = A[2] * global_x + (1 - A[2]) * local_x

        global_logits = self.global_linear_2(x)
        local_logits = self.local_linear_2(x)

        logits = A[3] * global_logits + (1 - A[3]) * local_logits

        outputs = torch.transpose(logits, 1, 2)
        return outputs

# net = StackoverflowNet(10004, 96, 670, 1)
# pytorch_total_params = sum(p.numel() for p in net.parameters())
# print(pytorch_total_params)