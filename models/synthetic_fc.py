from numpy import dtype
import torch
import torch.nn as nn
import torch.nn.functional as F

class SyntheticNet(nn.Module):
    def __init__(self, input_dim = 60, mid_dim = 20, output_dim = 10):
        super(SyntheticNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, output_dim)
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x