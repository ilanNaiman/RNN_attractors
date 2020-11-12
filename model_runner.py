import json
import torch
from torchtext import data
import torch.nn as nn
import torch.optim as opt
import pickle
import random
import numpy as np

from LSTM_sentiment import LSTM_sentiment
from RNN_sentiment import RNN_sentiment


# use: send operation as string and also file name
def save_load_pickle(operation, filename, obj=None):
    if operation == "save":
        with open(filename, 'wb') as f:
            pickle.dump(obj, f)
        # print('data saved')
    elif operation == "load":
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
    else:
        print('Invalid save_load_pickle option')
    return obj


def create_json():
    num_of_json = 200000
    file = 'yelp_dataset/yelp_academic_dataset_review.json'
    _parsed_file = 'yelp_review_small_200k.json'
    with open(file) as f:
        with open(_parsed_file, 'w') as outf:
            for i, line in enumerate(f):
                pl = json.loads(line)
                json.dump({"text": pl["text"], "label": pl["stars"]}, outf)
                outf.write('\n')
                if i == num_of_json:
                    break
    return _parsed_file


# No. of trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def rate_from_prediction(prediction):
    top_n, top_i = prediction.topk(1)
    rate_i = top_i[0].item()
    return rate_i


def train(_model, iterator, _optimizer, _criterion, n_epochs, _g_step, epoch):
    # initialize every epoch
    epoch_loss = 0
    epoch_acc = 0
    epoch_step = 0
    hidden_val = None
    _eval_loss = len(iterator) // 100
    train_loss_list = []
    epoch_step_list = []

    # set the model in training phase
    _model.train().double()

    for batch in iterator:
        # resets the gradients after every batch
        _optimizer.zero_grad()

        # retrieve text and no. of words
        _text, text_lengths = batch.text

        # convert to 1D tensor
        predictions, hidden_val, _ = _model(_text, text_lengths)
        predictions = torch.squeeze(predictions)
        hidden_val = torch.squeeze(hidden_val)

        # compute the loss
        loss = _criterion(predictions, batch.label)

        # compute the accuracy
        guess = rate_from_prediction(predictions)
        correct = (guess == batch.label).float()
        acc = correct.sum() / len(correct)

        # backpropage the loss and compute the gradients
        loss.backward()

        # update the weights
        _optimizer.step()

        # loss and accuracy
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        _g_step += 1
        epoch_step += 1
        if _g_step % _eval_loss == 0:
            average_train_loss = epoch_loss / epoch_step
            train_loss_list.append(average_train_loss)
            epoch_step_list.append(_g_step)
            # print progress
            print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}'
                  .format(epoch + 1, n_epochs, _g_step, n_epochs * len(iterator),
                          average_train_loss))

    return epoch_loss / len(iterator), epoch_acc / len(iterator), hidden_val, _g_step


def evaluate(_model, iterator, _criterion):
    # initialize every epoch
    epoch_loss = 0
    epoch_acc = 0

    # deactivating dropout layers
    _model.eval()
    tensor_emp = torch.ones(())
    collected_states = tensor_emp.new_tensor([])
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


            # compute loss and accuracy
            loss = _criterion(predictions, batch.label)

            guess = rate_from_prediction(predictions)
            correct = (guess == batch.label).float()
            acc = correct.sum() / len(correct)

            # keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator), collected_states


def main():
    # path of the saved model and params
    posm = 'lstm_model/'


    # # Reproducing same results
    # SEED = 2020
    #
    # # Torch
    # torch.manual_seed(SEED)

    parsed_file = create_json()

    text = data.Field(tokenize='spacy', batch_first=True, include_lengths=True)
    label = data.LabelField(dtype=torch.long, batch_first=True)
    data_field = {"text": ("text", text),
                  "label": ("label", label)
                  }
    # loading custom dataset
    training_data = data.TabularDataset(path=parsed_file, format='json', fields=data_field)

    # print preprocessed text
    # print(len(training_data.examples))

    # train_data, test_data = training_data.split(split_ratio=0.7, random_state=random.seed(SEED))
    train_data, test_data = training_data.split(split_ratio=0.7)

    # initialize glove embeddings
    text.build_vocab(train_data, min_freq=1, vectors="glove.6B.100d")
    label.build_vocab(train_data)

    # # No. of unique tokens in text
    # print("Size of TEXT vocabulary:", len(text.vocab))
    #
    # # No. of unique tokens in label
    # print("Size of LABEL vocabulary:", len(label.vocab))
    #
    # # Commonly used words
    # print(text.vocab.freqs.most_common(10))
    #
    # # Word dictionary
    # print(text.vocab.stoi)

    # set batch size
    batch_size = 64

    # Load an iterator
    train_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, test_data),
        batch_size=batch_size,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True)

    # hyper-parameters
    size_of_vocab = len(text.vocab)
    embedding_dim = 100
    num_hidden_nodes = 128
    num_output_nodes = 5
    num_layers = 2
    hyper_params = {'size_of_vocab': size_of_vocab, 'embedding_dim': embedding_dim,
                    'num_hidden_nodes': num_hidden_nodes,
                    'num_output_nodes': num_output_nodes, 'num_layers': num_layers}

    # save text, label and data of train and test
    save_load_pickle('save', posm + 'text_label', [text, label])
    save_load_pickle('save', posm + 'data_field', data_field)
    save_load_pickle('save', posm + 'hyper_params', hyper_params)

    # instantiate the Rnn sentiment classification model for Yelp
    model = LSTM_sentiment(size_of_vocab, embedding_dim, num_hidden_nodes, num_output_nodes, num_layers)

    # architecture
    print(model)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    # Initialize the pretrained embedding
    pretrained_embeddings = text.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)

    # print(pretrained_embeddings.shape)

    criterion = nn.NLLLoss()
    optimizer = opt.Adam(model.parameters())
    g_step = 0

    n_epochs = 3
    best_valid_loss = float('inf')
    tensor = torch.ones(())
    hidden_list = tensor.new_tensor([])
    # print(hidden_list)

    for epoch in range(n_epochs):
        # train the model
        train_loss, train_acc, hidden_values, g_step = train(model, train_iterator, optimizer,
                                                             criterion, n_epochs, g_step, epoch)
        hidden_list = torch.cat((hidden_list, hidden_values), 0)

        # evaluate the model
        valid_loss, valid_acc, collected_states = evaluate(model, test_iterator, criterion)

        # collect RNN states concatenated across 1,000 test examples.

        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_load_pickle('save', posm + 'hidden_states_list', hidden_list)
            save_load_pickle('save', posm + 'hidden_state_from_test', collected_states)
            torch.save(model.state_dict(), posm + 'saved_model_LSTM_200k.pt')

        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')


if __name__ == "__main__":
    main()
