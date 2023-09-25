# src: https://github.com/FedML-AI/FedML/blob/ecd2d81222301d315ca3a84be5a5ce4f33d6181c/python/fedml/model/cv/cnn.py

import torch
import torch.nn as nn

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

    def __init__(self, only_digits=False):
        super(EMNISTNet, self).__init__()

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

        self.prob_linear_1 = nn.Sequential(
            nn.Linear(2 * 26 * 26, 250),
            nn.Linear(250, 50),
            nn.Linear(50, 2),
        ) 

        self.prob_linear_2 = nn.Sequential(
            nn.Linear(2 * 24 * 24, 200),
            nn.Linear(200, 50),
            nn.Linear(50, 2),
        ) 

        self.prob_linear_3 = nn.Sequential(
            nn.Linear(2 * 128, 50),
            nn.Linear(50, 2),
        ) 

        self.prob_linear_4 = nn.Sequential(
            nn.Linear((10 if only_digits else 62) * 2, 2),
        ) 

    def forward(self, x, local = False):
        x = torch.squeeze(x)
        if len(x.shape) == 2:
          x = torch.unsqueeze(x, 0)
        x = torch.unsqueeze(x, 1)

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

            probabilities_1 = torch.softmax(self.prob_linear_1(torch.cat((torch.flatten(torch.mean(global_x, dim = 1), start_dim=1), torch.flatten(torch.mean(global_x, dim = 1), start_dim=1)), dim = 1)), dim = 1)
            x = probabilities_1[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(global_x) * global_x + probabilities_1[:, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(local_x) * local_x
            x = self.relu(x)
            
            global_x = self.global_conv2d_2(x)
            local_x = self.local_conv2d_2(x)

            probabilities_2 = torch.softmax(self.prob_linear_2(torch.cat((torch.flatten(torch.mean(global_x, dim = 1), start_dim=1), torch.flatten(torch.mean(global_x, dim = 1), start_dim=1)), dim = 1)), dim = 1)
            x = probabilities_2[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(global_x) * global_x + probabilities_2[:, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(local_x) * local_x
            x = self.relu(x)
            x = self.max_pooling(x)
            x = self.dropout_1(x)
            x = self.flatten(x)
            
            global_x = self.global_linear_1(x)
            local_x = self.local_linear_1(x)

            probabilities_3 = torch.softmax(self.prob_linear_3(torch.cat((global_x, local_x), dim = 1)), dim = 1)
            x = probabilities_3[:, 0].unsqueeze(1).expand_as(global_x) * global_x + probabilities_3[:, 1].unsqueeze(1).expand_as(local_x) * local_x
            x = self.relu(x)
            x = self.dropout_2(x)
            
            global_x = self.global_linear_2(x)
            local_x = self.local_linear_2(x)
            
            probabilities_4 = torch.softmax(self.prob_linear_4(torch.cat((global_x, local_x), dim = 1)), dim = 1)
            x = probabilities_4[:, 0].unsqueeze(1).expand_as(global_x) * global_x + probabilities_4[:, 1].unsqueeze(1).expand_as(local_x) * local_x
        
        return x