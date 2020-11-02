import json
import numpy as np

num_of_json = 70
file = 'yelp_dataset/yelp_academic_dataset_review.json'
parsed_file = 'yelp_review_small.json'
list_of_reviews_rate = []
with open(file) as f:
    with open(parsed_file, 'w') as outf:
        for i, line in enumerate(f):
            pl = json.loads(line)
            json.dump({"text": pl["text"], "label": pl["stars"]}, outf)
            outf.write('\n')
            if i == num_of_json:
                break

# %%

import torch
from torchtext import data

# # Reproducing same results
# SEED = 2019
#
# # Torch
# torch.manual_seed(SEED)

TEXT = data.Field(tokenize='spacy', batch_first=True, include_lengths=True)
LABEL = data.LabelField(dtype=torch.long, batch_first=True)
datafield = {"text": ("text", TEXT),
             "label": ("label", LABEL)
             }

# loading custom dataset
training_data = data.TabularDataset(path='yelp_review_small.json', format='json', fields=datafield)

# print preprocessed text
print(len(training_data.examples))

# %%

import random

# train_data, test_data = training_data.split(split_ratio=0.7, random_state=random.seed(SEED))
train_data, test_data = training_data.split(split_ratio=0.7)

# %%

# initialize glove embeddings
TEXT.build_vocab(train_data, min_freq=1, vectors="glove.6B.100d")
LABEL.build_vocab(train_data)

# No. of unique tokens in text
print("Size of TEXT vocabulary:", len(TEXT.vocab))

# No. of unique tokens in label
print("Size of LABEL vocabulary:", len(LABEL.vocab))

# Commonly used words
print(TEXT.vocab.freqs.most_common(10))

# Word dictionary
print(TEXT.vocab.stoi)

# %%

# set batch size
BATCH_SIZE = 64

# Load an iterator
train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data),
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.text),
    sort_within_batch=True)

# %%

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
        self.softmax = nn.LogSoftmax(dim=2)

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

        linear_outputs = self.linear(hidden)
        output = self.softmax(linear_outputs)
        return output, hidden


# %%


# hyper-parameters
size_of_vocab = len(TEXT.vocab)
embedding_dim = 100
num_hidden_nodes = 32
num_output_nodes = 5
num_layers = 1

# instantiate the Rnn sentiment classification model for Yelp
model = RNN_sentiment(size_of_vocab, embedding_dim, num_hidden_nodes, num_output_nodes, num_layers)

# %%

# architecture
print(model)


# No. of trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

# Initialize the pretrained embedding
pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

print(pretrained_embeddings.shape)

# %%

import torch.optim as opt

criterion = nn.NLLLoss()
optimizer = opt.Adam(model.parameters())


def rate_from_prediction(prediction):
    top_n, top_i = prediction.topk(1)
    rate_i = top_i[0].item()
    return rate_i


def train(model, iterator, optimizer, criterion):
    # initialize every epoch
    epoch_loss = 0
    epoch_acc = 0

    # set the model in training phase
    model.train().double()

    for batch in iterator:
        # resets the gradients after every batch
        optimizer.zero_grad()

        # retrieve text and no. of words
        text, text_lengths = batch.text

        # convert to 1D tensor
        predictions, hidden_val = model(text, text_lengths)
        predictions = torch.squeeze(predictions)
        hidden_val = torch.squeeze(hidden_val)

        # compute the loss
        loss = criterion(predictions, batch.label)

        # compute the accuracy
        guess = rate_from_prediction(predictions)
        correct = (guess == batch.label).float()
        acc = correct.sum() / len(correct)

        # backpropage the loss and compute the gradients
        loss.backward()

        # update the weights
        optimizer.step()

        # loss and accuracy
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator), hidden_val


# %%


def evaluate(model, iterator, criterion):
    # initialize every epoch
    epoch_loss = 0
    epoch_acc = 0

    # deactivating dropout layers
    model.eval()

    # deactivates autograd
    with torch.no_grad():
        for batch in iterator:
            # retrieve text and no. of words
            text, text_lengths = batch.text

            # convert to 1d tensor
            predictions, _ = model(text, text_lengths)
            predictions = torch.squeeze(predictions)

            # compute loss and accuracy
            loss = criterion(predictions, batch.label)

            guess = rate_from_prediction(predictions)
            correct = (guess == batch.label).float()
            acc = correct.sum() / len(correct)

            # keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# %%


N_EPOCHS = 1
best_valid_loss = float('inf')
tensor = torch.ones(())
hidden_list = tensor.new_tensor([])
print(hidden_list)

for epoch in range(N_EPOCHS):

    # train the model
    train_loss, train_acc, hidden_values = train(model, train_iterator, optimizer, criterion)
    hidden_list = torch.cat((hidden_list, hidden_values), 0)
    # print('hidden_list:')
    # print(hidden_list)
    # print(hidden_list.shape)

    # evaluate the model
    valid_loss, valid_acc = evaluate(model, test_iterator, criterion)

    # save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')

    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
# %%

"""
    find approximate fixed points that are state vectors {h_1*, h_2*, h_3*, ...} where h∗i ≈F(h∗i,x=0)
    defining a loss function q = 1/N * ||(h - F(h,0) || _2 ^2
    and then minimizing q with respect to hidden states, h, using auto-differentiation methods.
    Run this optimization multiple times starting from different initial values of h.
    These initial conditions sampled randomly from the distribution of state activations explored by
    the trained network, which was done to intentionally sample states related to the operation of the RNN.
"""


# generate random integer to sample some random state of activations:

def loss_func(_hidden_state):
    # tensor full of zeros
    zero_input = torch.zeros(1, 1, dtype=torch.long)
    len_zero_input = torch.empty(1)
    len_zero_input[0] = zero_input.shape[1]
    # create hidden state tensor of previous step
    _hs_prev = torch.from_numpy(_hidden_state)
    _hs_prev = _hs_prev.reshape((1, 1, num_hidden_nodes))

    # len_zero_input holds the length of the zero input vector
    # deactivates autograd
    with torch.no_grad():
        # run through 1 time step, thus h_(t) = f(h_(t-1), 0), where
        # input x = 0, and h_(t-1) is the hidden state for time step t-1
        _, f_of_zero_input = model(zero_input, len_zero_input, _hs_prev)
        f_of_zero_input = f_of_zero_input.detach().numpy()
        elem_wise_sub = np.subtract(_hidden_state, f_of_zero_input)
        elem_wise_square = np.square(elem_wise_sub)
        elem_sum = np.sum(elem_wise_square)
        # q is the loss function we want to minimize as written in the paper
        q = elem_sum / elem_wise_sub.size
        return q


from scipy.optimize import minimize

fixed_point_list = []
for _ in range(20):
    # print(hidden_list.shape[0])
    rand_int = random.randrange(hidden_list.shape[0])
    # print(rand_int)
    hidden_state = hidden_list[rand_int]
    hidden_state = hidden_state.detach().numpy()

    # print(hidden_state)
    # use minimize()
    res = minimize(loss_func, hidden_state, method='nelder-mead')
    fixed_point_list.append(res.success)

result = filter(lambda x: x, fixed_point_list)
print(list(result))
