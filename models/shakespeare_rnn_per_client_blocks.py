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
        self.global_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.local_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.global_lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=2, batch_first=True,)
        self.local_lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=2, batch_first=True,)
        self.global_fc = nn.Linear(hidden_size, vocab_size)
        self.local_fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq, alpha):
        alpha = F.gumbel_softmax(alpha)
        A = torch.argmax(alpha, dim=1)

        global_embeds = self.global_embeddings(input_seq)
        local_embeds = self.local_embeddings(input_seq)

        embeds = A[0] * global_embeds + (1 - A[0]) * local_embeds
        # Note that the order of mini-batch is random so there is no hidden relationship among batches.
        # So we do not input the previous batch's hidden state,
        # leaving the first hidden state zero `self.lstm(embeds, None)`.
        global_lstm_out, _ = self.global_lstm(embeds)
        local_lstm_out, _ = self.local_lstm(embeds)
        # use the final hidden state as the next character prediction

        lstm_out = A[0] * global_lstm_out + (1 - A[0]) * local_lstm_out
        # output = self.fc(final_hidden_state)
        # For fed_shakespeare
        global_output = self.global_fc(lstm_out[:,:])
        local_output = self.local_fc(lstm_out[:,:])

        output = A[0] * global_output + (1 - A[0]) * local_output

        output = torch.transpose(output, 1, 2)
        return output