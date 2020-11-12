# run analysis on given trained model

# hyper-parameters
import torch
from torchtext import data
import matplotlib.pyplot as plt
import numpy as np


from LSTM_sentiment import LSTM_sentiment
from RNN_sentiment import RNN_sentiment
from analysis import dmd_alg, pca_hs_2
from model_runner import save_load_pickle

posm = 'lstm_model/'
hyper_param = save_load_pickle('load', posm + 'hyper_params')
[text, label] = save_load_pickle('load', posm + 'text_label')
hidden_states_list = save_load_pickle('load', posm + 'hidden_states_list')
hidden_state_from_test = save_load_pickle('load', posm + 'hidden_state_from_test')
data_field = save_load_pickle('load', posm + 'data_field')


size_of_vocab = hyper_param['size_of_vocab']
embedding_dim = hyper_param['embedding_dim']
num_hidden_nodes = hyper_param['num_hidden_nodes']
num_output_nodes = hyper_param['num_output_nodes']
num_layers = hyper_param['num_layers']

# instantiate the Rnn sentiment classification model for Yelp
# load model (choose model: RNN, LSTM etc.):
model = LSTM_sentiment(size_of_vocab, embedding_dim, num_hidden_nodes, num_output_nodes, num_layers)
model.load_state_dict(torch.load(posm + 'saved_model_LSTM_200k.pt'))
model.eval()


def run_some_examples(_model, iterator):
    num_of_examples = 0

    # deactivates autograd
    with torch.no_grad():
        for batch in iterator:
            num_of_examples += batch.batch_size
            # retrieve text and no. of words
            _text, text_lengths = batch.text
            predictions, hidden, packed_hidden = _model(_text, text_lengths)
            predictions = torch.squeeze(predictions)
            if num_of_examples < 1000:
                collected_states = torch.cat((collected_states, packed_hidden.data), 0)
    return collected_states


def plot_dmd_eigenvalues(_hidden_state_from_test):
    hsft_np = _hidden_state_from_test.numpy()
    hsft_np = np.transpose(hsft_np)
    print(hsft_np)
    print(hsft_np.size)
    x_mat = hsft_np[:-1]
    y_mat = hsft_np[1:]
    print(x_mat)
    print(y_mat)
    a_mat, eigenval, _ = dmd_alg(x_mat, y_mat)
    print('######## eigenvalues #########')
    print(eigenval)
    # plot unit circle
    t = np.linspace(0, 2 * np.pi, 101)
    plt.plot(np.cos(t), np.sin(t))
    # plot data
    plt.plot(eigenval.real, eigenval.imag, 'ro')
    # make box bigger
    plt.axes().set_aspect('equal')
    m = max(max(abs(eigenval.real)), max(abs(eigenval.imag)))
    plt.xlim(-1.1 * m, 1.1 * m)
    plt.ylim(-1.1 * m, 1.1 * m)
    # draw horizontal and vertical axes
    plt.plot([0, 0], [-1.1 * m, 1.1 * m], 'k', linewidth=0.5)
    plt.plot([-1.1 * m, 1.1 * m], [0, 0], 'k', linewidth=0.5)
    plt.show()

    # plot matrix
    plt.matshow(a_mat)
    plt.colorbar()
    plt.show()


pca_hs_2(hidden_state_from_test)
plot_dmd_eigenvalues(hidden_state_from_test)




# # Load an iterator
# train_iterator, test_iterator = data.BucketIterator.splits(
#     (train_data, test_data),
#     batch_size=BATCH_SIZE,
#     sort_key=lambda x: len(x.text),
#     sort_within_batch=True)
#
#
#
# seen_rands = set()
# fixed_points_list = find_fixed_points()
# print(len(fixed_point_list))
# print(collected_states)
# collected_states = np.array(collected_states)
# print(len(collected_states))
# print(collected_states)
#
# pca_hs(collected_states)
# # pca_hs_2(collected_states)

