import torch.nn as nn


class RNN_sentiment(nn.Module):

    # For each element in the input sequence, each layer computes the following function:

    # h_t = ReLU(W_ih*x_t+b_ih + W_hh*h_(t-1)+b_hh)

    # where h_t is the hidden state at time t, x_t is the input at time t, and h_(t-1)
    # is the hidden state of the previous layer at time t-1 or the initial hidden state at time 0

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers):
        # Constructor
        super().__init__()

        # embedding layer
        # A simple lookup table that stores embeddings of a fixed dictionary and size.
        # This module is often used to store word embeddings and retrieve them using indices.
        # The input to the module is a list of indices, and the output is the corresponding word embeddings.
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # rnn layer
        self.rnn = nn.RNN(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          batch_first=True)

        # linear layer towards output
        self.linear = nn.Linear(hidden_dim, output_dim)

        # activation function
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, text, text_length, prev_hidden=None):
        embedded = self.embedding(text)
        # input can be of size T x B x * where T is the length of the longest sequence (equal to lengths[0]),
        # B is the batch size, and * is any number of dimensions (including 0).
        # If batch_first is True, B x T x * input is expected.
        packed_embedding = nn.utils.rnn.pack_padded_sequence(embedded, text_length, batch_first=True)
        # print('packed_embedding:')
        # print(packed_embedding)
        if prev_hidden is None:
            # print('prev_hidden is None')
            packed_output, hidden = self.rnn(packed_embedding)
        else:
            # print('prev_hidden is NOT None')
            # print(prev_hidden)
            packed_output, hidden = self.rnn(packed_embedding, prev_hidden)

        last_layer_hidden = hidden.clone()
        last_layer_hidden = last_layer_hidden[-1:].squeeze()
        linear_outputs = self.linear(last_layer_hidden)
        output = self.softmax(linear_outputs)
        return output, hidden, packed_output
