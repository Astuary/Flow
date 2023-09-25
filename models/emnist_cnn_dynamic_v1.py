# src: https://github.com/FedML-AI/FedML/blob/ecd2d81222301d315ca3a84be5a5ce4f33d6181c/python/fedml/model/cv/cnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class EMNISTNet(nn.Module):
    """
    Recommended model by "Adaptive Federated Optimization" (https://arxiv.org/pdf/2003.00295.pdf)
    Used for EMNIST experiments.
    When `only_digits=True`, the summary of returned model is
    ```
    Model:
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    reshape (Reshape)            (None, 28, 28, 1)         0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 26, 26, 32)        320
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 24, 24, 64)        18496
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 12, 12, 64)        0
    _________________________________________________________________
    dropout (Dropout)            (None, 12, 12, 64)        0
    _________________________________________________________________
    flatten (Flatten)            (None, 9216)              0
    _________________________________________________________________
    dense (Dense)                (None, 128)               1179776
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 128)               0
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                1290
    =================================================================
    Total params: 1,199,882
    Trainable params: 1,199,882
    Non-trainable params: 0
    ```
    Args:
      only_digits: If True, uses a final layer with 10 outputs, for use with the
        digits only MNIST dataset (http://yann.lecun.com/exdb/mnist/).
        If False, uses 62 outputs for Federated Extended MNIST (FEMNIST)
        EMNIST: Extending MNIST to handwritten letters: https://arxiv.org/abs/1702.05373.
    Returns:
      A `torch.nn.Module`.
    """

    def __init__(self, only_digits=False, batch_size=20):
        super(EMNISTNet, self).__init__()

        self.batch_size = batch_size
        self.prob_policy_net = EMNISTPolicyNet()

        self.global_conv2d_1 = torch.nn.Conv2d(1, 32, kernel_size=3)
        self.local_conv2d_1 = torch.nn.Conv2d(1, 32, kernel_size=3)
        
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        
        self.global_conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=3)
        self.local_conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=3)
        
        self.dropout_1 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        
        self.global_linear_1 = nn.Linear(9216, 128)
        self.local_linear_1 = nn.Linear(9216, 128)
        
        self.dropout_2 = nn.Dropout(0.5)
        
        self.global_linear_2 = nn.Linear(128, 10 if only_digits else 62)
        self.local_linear_2 = nn.Linear(128, 10 if only_digits else 62)
        
        self.relu = nn.ReLU()

    def forward(self, x, mode = 'personalized'):
        x = torch.squeeze(x)
        if len(x.shape) == 2:
            x = torch.unsqueeze(x, 0)
        x = torch.unsqueeze(x, 1)

        p1, p2, p3, p4 = None, None, None, None
        # print(x.shape)
        # print(p1.shape)
        # print(self.global_conv2d_1.weight.shape)
        # print(self.local_conv2d_1.bias.shape)

        if mode == 'local':
            x = self.local_conv2d_1(x)
            x = self.relu(x)
            x = self.local_conv2d_2(x)
            x = self.relu(x)
            x = self.max_pooling(x)
            x = self.dropout_1(x)
            x = self.flatten(x)
            x = self.local_linear_1(x)
            x = self.relu(x)
            x = self.dropout_2(x)
            x = self.local_linear_2(x)

        elif mode == 'global':
            x = self.global_conv2d_1(x)
            x = self.relu(x)
            x = self.global_conv2d_2(x)
            x = self.relu(x)
            x = self.max_pooling(x)
            x = self.dropout_1(x)
            x = self.flatten(x)
            x = self.global_linear_1(x)
            x = self.relu(x)
            x = self.dropout_2(x)
            x = self.global_linear_2(x)

        elif mode == 'personalized':
            p1, p2, p3, p4 = self.prob_policy_net(x)
            p1, p2, p3, p4 = torch.round(p1), torch.round(p2), torch.round(p3), torch.round(p4)
            global_x = self.global_conv2d_1(x)
            local_x = self.local_conv2d_1(x)

            x = p1[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(global_x) * global_x + p1[:, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(local_x) * local_x
            x = self.relu(x)
            
            global_x = self.global_conv2d_2(x)
            local_x = self.local_conv2d_2(x)

            x = p2[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(global_x) * global_x + p2[:, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(local_x) * local_x
            x = self.relu(x)
            x = self.max_pooling(x)
            x = self.dropout_1(x)
            x = self.flatten(x)
            
            global_x = self.global_linear_1(x)
            local_x = self.local_linear_1(x)

            x = p3[:, 0].unsqueeze(1).expand_as(global_x) * global_x + p3[:, 1].unsqueeze(1).expand_as(local_x) * local_x
            x = self.relu(x)
            x = self.dropout_2(x)
            
            global_x = self.global_linear_2(x)
            local_x = self.local_linear_2(x)
            
            x = p4[:, 0].unsqueeze(1).expand_as(global_x) * global_x + p4[:, 1].unsqueeze(1).expand_as(local_x) * local_x
        
        return x, p1, p2, p3, p4


    def forward_input_mixture(self, x, local = False):
        x = torch.squeeze(x)
        if len(x.shape) == 2:
          x = torch.unsqueeze(x, 0)
        x = torch.unsqueeze(x, 1)

        p1, p2, p3, p4 = self.prob_policy_net(x)

        # print(x.shape)
        # print(p1.shape)
        # print(self.global_conv2d_1.weight.shape)
        # print(self.local_conv2d_1.bias.shape)

        if local:
            x = self.local_conv2d_1(x)
            x = self.relu(x)
            x = self.local_conv2d_2(x)
            x = self.relu(x)
            x = self.max_pooling(x)
            x = self.dropout_1(x)
            x = self.flatten(x)
            x = self.local_linear_1(x)
            x = self.relu(x)
            x = self.dropout_2(x)
            x = self.local_linear_2(x)

        else:
            global_x = self.global_conv2d_1(x)
            local_x = self.local_conv2d_1(x)

            x = p1[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(global_x) * global_x + p1[:, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(local_x) * local_x
            x = self.relu(x)
            
            global_x = self.global_conv2d_2(x)
            local_x = self.local_conv2d_2(x)

            x = p2[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(global_x) * global_x + p2[:, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(local_x) * local_x
            x = self.relu(x)
            x = self.max_pooling(x)
            x = self.dropout_1(x)
            x = self.flatten(x)
            
            global_x = self.global_linear_1(x)
            local_x = self.local_linear_1(x)

            x = p3[:, 0].unsqueeze(1).expand_as(global_x) * global_x + p3[:, 1].unsqueeze(1).expand_as(local_x) * local_x
            x = self.relu(x)
            x = self.dropout_2(x)
            
            global_x = self.global_linear_2(x)
            local_x = self.local_linear_2(x)
            
            x = p4[:, 0].unsqueeze(1).expand_as(global_x) * global_x + p4[:, 1].unsqueeze(1).expand_as(local_x) * local_x
        
        return x


class EMNISTPolicyNet(nn.Module):
    def __init__(self,):
        super(EMNISTPolicyNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 350)
        self.fc2 = nn.Linear(350, 150)
        self.fc3 = nn.Linear(150, 50)
        self.fc4 = nn.Linear(50, 2)

        self.fc1_exit = nn.Sequential(
            nn.Dropout(0.3,),
            nn.ReLU(),
            nn.Linear(350, 2)
        )
        self.fc2_exit = nn.Sequential(
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(150, 2)
        )
        self.fc3_exit = nn.Sequential(
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(50, 2)
        )
        
    def forward(self, x):
        x = torch.flatten(x, start_dim = 1)
        
        intermediate_x = self.fc1(x)

        y = self.fc1_exit(intermediate_x)
        y1 = torch.softmax(y, dim=1)

        x = F.relu(intermediate_x)
        intermediate_x = self.fc2(x)

        y = self.fc2_exit(intermediate_x)
        y2 = torch.softmax(y, dim=1)

        x = F.relu(intermediate_x)
        intermediate_x = self.fc3(x)

        y = self.fc3_exit(intermediate_x)
        y3 = torch.softmax(y, dim=1)

        x = F.relu(intermediate_x)
        x = self.fc4(x)
        y4 = torch.softmax(x, dim=1)

        return y1, y2, y3, y4