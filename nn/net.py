import torch
import torch.nn.functional as F


class TransformerLSTMSingleId(torch.nn.Module):
    def __init__(
        self, n_out, n_embed, n_lstm_hid, n_lstm_layer, n_tf_hid, n_tf_layer, n_tf_head, id_embed,
        lstm_dropout=0, lstm_bidirect=True, tf_dropout=0.1
    ):
        super().__init__()
        # embedding initialization
        self.id_embed = torch.nn.Embedding.from_pretrained(id_embed)

        # lstm layer initialization
        self.rnn = torch.nn.GRU(
            input_size=n_embed, hidden_size=n_lstm_hid, num_layers=n_lstm_layer, batch_first=True,
            bidirectional=lstm_bidirect, dropout=lstm_dropout
        )

        # transformer layer initialization
        encoder_layer = torch.nn.TransformerEncoderLayer(n_embed, n_tf_head, n_tf_hid, tf_dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, n_tf_layer)

        self.rnn_dropout = torch.nn.Dropout(0.3)
        self.fc1_dropout = torch.nn.Dropout(0.25)
        self.fc2_dropout = torch.nn.Dropout(0.25)

        indicator = 2 if lstm_bidirect else 1
        self.fc1 = torch.nn.Linear(n_embed * indicator, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, n_out)

        self.init_weights()

    def forward(self, x):
        x = self.id_embed(x)

        x = self.transformer_encoder(x)
        x, _ = self.rnn(self.rnn_dropout(x))
        x, _ = torch.max(x, dim=1)

        x = self.fc1(F.relu(self.fc1_dropout(x)))
        x = self.fc2(F.relu(self.fc2_dropout(x)))
        x = self.fc3(x)

        return x

    def init_weights(self):
        # rnn initialization
        for name, val in self.rnn.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(val)
            if 'bias' in name:
                torch.nn.init.zeros_(val)

        # fully connected layer initiail
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.zeros_(self.fc3.bias)


class TransformerLSTMTripleId(torch.nn.Module):
    def __init__(
        self, n_out, n_embed, n_lstm_hid, n_lstm_layer, n_tf_hid, n_tf_layer, n_tf_head, id1_embed,
        id2_embed, id3_embed, lstm_dropout=0, lstm_bidirect=True, tf_dropout=0.1
    ):
        super().__init__()
        # embedding initialization
        self.id1_embed = torch.nn.Embedding.from_pretrained(id1_embed)
        self.id2_embed = torch.nn.Embedding.from_pretrained(id2_embed)
        self.id3_embed = torch.nn.Embedding.from_pretrained(id3_embed)

        # lstm layer initialization
        self.rnn1 = torch.nn.GRU(
            input_size=n_embed, hidden_size=n_lstm_hid, num_layers=n_lstm_layer, batch_first=True,
            bidirectional=lstm_bidirect, dropout=lstm_dropout
        )
        self.rnn2 = torch.nn.GRU(
            input_size=n_embed, hidden_size=n_lstm_hid, num_layers=n_lstm_layer, batch_first=True,
            bidirectional=lstm_bidirect, dropout=lstm_dropout
        )
        self.rnn3 = torch.nn.GRU(
            input_size=n_embed, hidden_size=n_lstm_hid, num_layers=n_lstm_layer, batch_first=True,
            bidirectional=lstm_bidirect, dropout=lstm_dropout
        )

        # transformer layer initialization
        encoder_layer1 = torch.nn.TransformerEncoderLayer(n_embed, n_tf_head, n_tf_hid, tf_dropout)
        encoder_layer2 = torch.nn.TransformerEncoderLayer(n_embed, n_tf_head, n_tf_hid, tf_dropout)
        encoder_layer3 = torch.nn.TransformerEncoderLayer(n_embed, n_tf_head, n_tf_hid, tf_dropout)
        self.transformer_encoder1 = torch.nn.TransformerEncoder(encoder_layer1, n_tf_layer)
        self.transformer_encoder2 = torch.nn.TransformerEncoder(encoder_layer2, n_tf_layer)
        self.transformer_encoder3 = torch.nn.TransformerEncoder(encoder_layer3, n_tf_layer)

        self.rnn1_dropout = torch.nn.Dropout(0.3)
        self.rnn2_dropout = torch.nn.Dropout(0.3)
        self.rnn3_dropout = torch.nn.Dropout(0.3)
        self.fc1_dropout = torch.nn.Dropout(0.25)
        self.fc2_dropout = torch.nn.Dropout(0.25)

        indicator = 2 if lstm_bidirect else 1
        self.fc1 = torch.nn.Linear(n_embed * indicator, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, n_out)

        self.init_weights()

    def forward(self, x1, x2, x3):
        x1 = self.id1_embed(x1)
        x2 = self.id2_embed(x2)
        x3 = self.id3_embed(x3)

        x1 = self.transformer_encoder1(x1)
        x2 = self.transformer_encoder2(x2)
        x3 = self.transformer_encoder3(x3)

        x1, _ = self.rnn1(self.rnn1_dropout(x1))
        x2, _ = self.rnn2(self.rnn2_dropout(x2))
        x3, _ = self.rnn3(self.rnn3_dropout(x3))

        x1, _ = torch.max(x1, dim=1)
        x2, _ = torch.max(x2, dim=1)
        x3, _ = torch.max(x3, dim=1)

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.fc1(F.relu(self.fc1_dropout(x)))
        x = self.fc2(F.relu(self.fc2_dropout(x)))
        x = self.fc3(x)

        return x

    def init_weights(self):
        # rnn initialization
        for name, val in self.rnn1.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(val)
            if 'bias' in name:
                torch.nn.init.zeros_(val)
        for name, val in self.rnn2.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(val)
            if 'bias' in name:
                torch.nn.init.zeros_(val)
        for name, val in self.rnn3.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(val)
            if 'bias' in name:
                torch.nn.init.zeros_(val)

        # fully connected layer initiail
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.zeros_(self.fc3.bias)
