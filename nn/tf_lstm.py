import time
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tensorflow as tf

from nn.net import TransformerLSTMSingleId
from nn.dataset import SingleIdTrainDataset
# --------------------------------------------------------------------------------------------------
# constant and hyper parameters

USER_FILE = 'data/train_preliminary/user.csv'
ID_EMBED = 'data/embed/ad_id.npy'
CORPUS_FILE = 'data/corpus/ad_id.pkl'

n_out = 10
n_embed = 128
n_lstm_hid = 128
n_lstm_layer = 2
n_tf_hid = 512
n_tf_layer = 2
n_tf_head = 4
lstm_dropout = 0
lstm_bidirect = True
tf_dropout = 0.1

np.random.seed(20)
torch.manual_seed(20)

max_len = 85
batch_size = 250
lr = 0.0002
epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Training with {device}')
# --------------------------------------------------------------------------------------------------
# load data, embedding, feature

user = pd.read_csv(USER_FILE)

with open(CORPUS_FILE, 'rb') as f:
    corpus = pickle.load(f)

id_embed = np.load(ID_EMBED)
id_embed = torch.FloatTensor(id_embed)

train = corpus[:900000]

features = tf.keras.preprocessing.sequence.pad_sequences(train, maxlen=max_len, value=0)
encoded_labels = user.age.values - 1
# --------------------------------------------------------------------------------------------------
# split dataset

split_frac = 0.8

# split data into training, validation, and test data (features and labels, x and y)
split_idx = int(features.shape[0] * split_frac)
train_x, remaining_x = features[:split_idx], features[split_idx:]
train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

test_idx = int(len(remaining_x) * 0.5)
val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

# print out the shapes of your resultant feature data
print("\t\t\tFeature Shapes:")
print(
    "Train set: \t\t{}".format(train_x.shape), "\nValidation set: \t{}".format(val_x.shape),
    "\nTest set: \t\t{}".format(test_x.shape)
)

# create Tensor datasets
train_data = SingleIdTrainDataset(train_x, train_y)
valid_data = SingleIdTrainDataset(val_x, val_y)
test_data = SingleIdTrainDataset(test_x, test_y)

# make sure the SHUFFLE your training data
train_loader = torch.utils.data.DataLoader(
    train_data, shuffle=True, drop_last=True, batch_size=batch_size
)
valid_loader = torch.utils.data.DataLoader(
    valid_data, shuffle=False, drop_last=True, batch_size=batch_size
)
test_loader = torch.utils.data.DataLoader(
    test_data, shuffle=False, drop_last=True, batch_size=batch_size
)

# obtain one batch of training data
dataiter = iter(train_loader)
sample_x, sample_y = dataiter.next()

print('Sample input creative_id size: ', sample_x.size())  # batch_size, seq_length
print('Sample input creative_id: \n', sample_x)
print()
print('Sample label size: ', sample_y.size())  # batch_size
print('Sample label: \n', sample_y)
# --------------------------------------------------------------------------------------------------
# initialize model and optimizer

net = TransformerLSTMSingleId(
    n_out=n_out, n_embed=n_embed, n_lstm_hid=n_lstm_hid, n_lstm_layer=n_lstm_layer,
    n_tf_hid=n_tf_hid, n_tf_layer=n_tf_layer, n_tf_head=n_tf_head, id_embed=id_embed,
    lstm_dropout=lstm_dropout, lstm_bidirect=lstm_bidirect, tf_dropout=tf_dropout
)
print(net)

# mincount 3 window 50, iter 5 ad
net.to(device)
step_size = len(train_loader)

# initialise loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
# --------------------------------------------------------------------------------------------------
# train

for e in range(epochs):
    start = time.perf_counter()
    # batch loop
    train_losses = list()
    num_correct = 0
    for x, labels in train_loader:
        x, labels = x.to(device), labels.to(device)

        output = net(x)
        loss = criterion(output, labels)
        train_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_correct += torch.sum(output.argmax(axis=1) == labels).item()
    train_acc = num_correct / len(train_loader.dataset)

    val_losses = list()
    num_correct = 0
    net.eval()
    with torch.no_grad():
        for x, labels in valid_loader:
            x, labels = x.to(device), labels.to(device)

            output = net(x)

            val_loss = criterion(output, labels)
            val_losses.append(val_loss.item())

            num_correct += torch.sum(output.argmax(axis=1) == labels).item()
    val_losses = np.mean(val_losses)
    valid_acc = num_correct / len(valid_loader.dataset)
    duration = time.perf_counter() - start
    print(
        "Epoch: {}/{}...".format(e + 1, epochs), "Train acc: {:.6f}...".format(train_acc),
        "Loss: {:.6f}...".format(np.mean(train_losses)), "Val acc: {:.6f}...".format(valid_acc),
        "Val Loss: {:.6f}...".format(val_losses), f"Elapse time: {duration:.2f}s"
    )
    net.train()
# --------------------------------------------------------------------------------------------------
# test

test_losses = list()
num_correct = 0

net.eval()
with torch.no_grad():
    for x, labels in test_loader:
        x, labels = x.to(device), labels.to(device)

        output = net(x)

        # calculate loss
        test_loss = criterion(output, labels)
        test_losses.append(test_loss.item())

        num_correct += torch.sum(output.argmax(axis=1) == labels).item()

test_acc = num_correct / len(test_loader.dataset)

print("Test loss: {:.3f}".format(np.mean(test_losses)))
print("Test accuracy: {:.3f}".format(test_acc))
