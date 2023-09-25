import torch
import torch.nn as nn

class StackoverflowLogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StackoverflowLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs