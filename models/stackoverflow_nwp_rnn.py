import torch
import torch.nn as nn

class StackoverflowNet(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers)
        self.linear_1 = nn.Linear(hidden_size, embedding_size)
        self.linear_2 = nn.Linear(embedding_size, vocab_size)

    def forward(self, x):
        self.lstm.flatten_parameters()

        embeddings = self.embedding(x)
        x, _ = self.lstm(embeddings)
        x = self.linear_1(x[:, :])
        logits = self.linear_2(x)
        outputs = torch.transpose(logits, 1, 2)
        return outputs

# net = StackoverflowNet(10004, 96, 670, 1)
# pytorch_total_params = sum(p.numel() for p in net.parameters())
# print(pytorch_total_params)