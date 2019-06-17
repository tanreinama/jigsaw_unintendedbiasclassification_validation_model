import pickle
import bz2
import io
import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F

#  model
class Attention(nn.Module):
    def __init__(self, feature_dim, max_seq_len=70):
        super().__init__()
        self.attention_fc = nn.Linear(feature_dim, 1)
        self.bias = nn.Parameter(torch.zeros(1, max_seq_len, 1, requires_grad=True))

    def forward(self, rnn_output):
        """
        forward attention scores and attended vectors
        :param rnn_output: (#batch,#seq_len,#feature)
        :return: attended_outputs (#batch,#feature)
        """
        attention_weights = self.attention_fc(rnn_output)
        seq_len = rnn_output.size(1)
        attention_weights = self.bias[:, :seq_len, :] + attention_weights
        attention_weights = torch.tanh(attention_weights)
        attention_weights = torch.exp(attention_weights)
        attention_weights_sum = torch.sum(attention_weights, dim=1, keepdim=True) + 1e-7
        attention_weights = attention_weights / attention_weights_sum
        attended = torch.sum(attention_weights * rnn_output, dim=1)
        return attended

class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super(GaussianNoise, self).__init__()

        self.stddev = stddev

    def forward(self, x):
        if self.training:
            noise = torch.empty_like(x)
            noise.normal_(0, self.stddev)
            return x + noise
        else:
            return x

class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

class NeuralNet(nn.Module):
    def __init__(self, model_factor, num_units1, num_units2, embedding_matrix, max_features, num_aux_targets, max_seq_len, num_feats):
        super(NeuralNet, self).__init__()
        self.model_factor = model_factor if model_factor is not None else [0,3,1,0,1,1]
        self.num_feats = num_feats
        embed_size = embedding_matrix.shape[1]

        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(self.model_factor[4]/10.0)

        if self.model_factor[2] == 0:
            self.layer1 = nn.LSTM(embed_size, num_units1, bidirectional=True, batch_first=True)
            self.layer2 = nn.LSTM(num_units1 * 2, num_units2, bidirectional=True, batch_first=True)
        elif self.model_factor[2] == 1:
            self.layer1 = nn.LSTM(embed_size, num_units1, bidirectional=True, batch_first=True)
            self.layer2 = nn.GRU(num_units1 * 2, num_units2, bidirectional=True, batch_first=True)
        elif self.model_factor[2] == 2:
            self.layer1 = nn.LSTM(embed_size, num_units1, bidirectional=True, batch_first=True)
            self.layer2 = nn.Conv1d(num_units1 * 2, num_units2 * 2, kernel_size=3, stride=1, padding=2)
        elif self.model_factor[2] == 3:
            self.layer1 = nn.Conv1d(embed_size, num_units1 * 2, kernel_size=3, stride=1, padding=2)
            self.layer2 = nn.LSTM(num_units1 * 2, num_units2, bidirectional=True, batch_first=True)

        if self.model_factor[1] == 0:
            num_dense_units = num_units2 * 4
        elif self.model_factor[1] == 1:
            self.attention1 = Attention(num_units1*2, max_seq_len)
            num_dense_units = num_units1 * 2 + num_units2 * 4
        elif self.model_factor[1] == 2:
            self.attention2 = Attention(num_units2*2, max_seq_len)
            num_dense_units = num_units2 * 6
        elif self.model_factor[1] == 3:
            self.attention1 = Attention(num_units1*2, max_seq_len)
            self.attention2 = Attention(num_units2*2, max_seq_len)
            num_dense_units = num_units1 * 2 + num_units2 * 6

        if self.model_factor[0] == 0:
            self.dropout = nn.Dropout(self.model_factor[5]/10.0)
        elif self.model_factor[0] == 1:
            self.noise = GaussianNoise(self.model_factor[5]/10.0)
        elif self.model_factor[0] == 2:
            self.dropout = nn.Dropout(self.model_factor[5]/10.0)
            self.bn = nn.BatchNorm1d(num_dense_units, momentum=0.5)
        elif self.model_factor[0] == 3:
            self.noise = GaussianNoise(self.model_factor[5]/10.0)
            self.bn = nn.BatchNorm1d(num_dense_units, momentum=0.5)

        num_dense_units = num_dense_units + self.num_feats

        if self.model_factor[3] == 1:
            self.hidden1 = nn.Linear(num_dense_units, num_dense_units)
        elif self.model_factor[3] == 2:
            self.hidden1 = nn.Linear(num_dense_units, num_dense_units)
            self.hidden2 = nn.Linear(num_dense_units, num_dense_units)
        elif self.model_factor[3] == 3:
            self.hidden1 = nn.Linear(num_dense_units, num_dense_units)
            self.hidden2 = nn.Linear(num_dense_units, num_dense_units)
            self.hidden3 = nn.Linear(num_dense_units, num_dense_units)

        self.linear_out = nn.Linear(num_dense_units, 1)
        self.linear_aux_out = nn.Linear(num_dense_units, num_aux_targets)

    def forward(self, x):
        if self.num_feats > 0:
            sent = x[:,self.num_feats:]
            feat = x[:,:self.num_feats].to(torch.float)
        else:
            sent = x
        h_embedding = self.embedding(sent)
        h_embedding = self.embedding_dropout(h_embedding)

        if self.model_factor[2] == 0 or self.model_factor[2] == 1:
            h_layer_1, _ = self.layer1(h_embedding)
            h_layer_2, _ = self.layer2(h_layer_1)
        elif self.model_factor[2] == 2:
            h_layer_1, _ = self.layer1(h_embedding)
            h_layer_2 = F.relu(torch.transpose(self.layer2(torch.transpose(h_layer_1,1,2)),2,1))
        elif self.model_factor[2] == 3:
            h_layer_1 = F.relu(torch.transpose(self.layer1(torch.transpose(h_embedding,1,2)),2,1))
            h_layer_2, _ = self.layer2(h_layer_1)

        avg_pool = torch.mean(h_layer_2, 1)
        max_pool, _ = torch.max(h_layer_2, 1)

        if self.model_factor[1] == 0:
            h_conc = torch.cat((avg_pool, max_pool), 1)
        elif self.model_factor[1] == 1:
            h_atten_1 = self.attention1(h_layer_1)
            h_conc = torch.cat((h_atten_1, avg_pool, max_pool), 1)
        elif self.model_factor[1] == 2:
            h_atten_2 = self.attention2(h_layer_2)
            h_conc = torch.cat((h_atten_2, avg_pool, max_pool), 1)
        elif self.model_factor[1] == 3:
            h_atten_1 = self.attention1(h_layer_1)
            h_atten_2 = self.attention2(h_layer_2)
            h_conc = torch.cat((h_atten_1, h_atten_2, avg_pool, max_pool), 1)

        if self.model_factor[0] == 0:
            h_conc = self.dropout(h_conc)
        elif self.model_factor[0] == 1:
            h_conc = self.noise(h_conc)
        elif self.model_factor[0] == 2:
            h_conc = self.dropout(h_conc)
            h_conc = self.bn(h_conc)
        elif self.model_factor[0] == 3:
            h_conc = self.noise(h_conc)
            h_conc = self.bn(h_conc)

        if self.num_feats > 0:
            h_conc = torch.cat((h_conc, feat), 1)

        if self.model_factor[3] == 0:
            hidden = h_conc
        elif self.model_factor[3] == 1:
            h_conc_linear1 = F.relu(self.hidden1(h_conc))
            hidden = h_conc + h_conc_linear1
        elif self.model_factor[3] == 2:
            h_conc_linear1 = F.relu(self.hidden1(h_conc))
            h_conc_linear2 = F.relu(self.hidden2(h_conc))
            hidden = h_conc + h_conc_linear1 + h_conc_linear2
        elif self.model_factor[3] == 3:
            h_conc_linear1 = F.relu(self.hidden1(h_conc))
            h_conc_linear2 = F.relu(self.hidden2(h_conc))
            h_conc_linear3 = F.relu(self.hidden3(h_conc))
            hidden = h_conc + h_conc_linear1 + h_conc_linear2 + h_conc_linear3

        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        out = torch.cat([result, aux_result], 1)

        return out

    def save_model(self, filename):
        model = self
        params = [
            model.layer1.state_dict(),
            model.layer2.state_dict(),
            model.linear_out.state_dict(),
            model.linear_aux_out.state_dict() ]
        if model.model_factor[1] == 1:
            params.append(model.attention1.state_dict())
        elif model.model_factor[1] == 2:
            params.append(model.attention2.state_dict())
        elif model.model_factor[1] == 2:
            params.append(model.attention1.state_dict())
            params.append(model.attention2.state_dict())
        if model.model_factor[0] >= 2:
            params.append(model.bn.state_dict())
        if model.model_factor[3] == 1:
            params.append(model.hidden1.state_dict())
        elif model.model_factor[3] == 2:
            params.append(model.hidden1.state_dict())
            params.append(model.hidden2.state_dict())
        elif model.model_factor[3] == 3:
            params.append(model.hidden1.state_dict())
            params.append(model.hidden2.state_dict())
            params.append(model.hidden3.state_dict())
        with bz2.open(filename, 'wb') as fout:
            buffer = io.BytesIO()
            torch.save(params, buffer)
            fout.write(buffer.getbuffer())

    def load_model(self, filename, device=torch.device('cuda')):
        with bz2.open(filename, 'rb') as fin:
            buffer = io.BytesIO(fin.read())
            params = torch.load(buffer, map_location=device)
        self.layer1.load_state_dict(params.pop(0))
        self.layer1.to(device)
        self.layer2.load_state_dict(params.pop(0))
        self.layer2.to(device)
        self.linear_out.load_state_dict(params.pop(0))
        self.linear_out.to(device)
        self.linear_aux_out.load_state_dict(params.pop(0))
        self.linear_aux_out.to(device)
        if self.model_factor[1] == 1:
            self.attention1.load_state_dict(params.pop(0))
            self.attention1.to(device)
        elif self.model_factor[1] == 2:
            self.attention2.load_state_dict(params.pop(0))
            self.attention2.to(device)
        elif self.model_factor[1] == 2:
            self.attention1.load_state_dict(params.pop(0))
            self.attention1.to(device)
            self.attention2.load_state_dict(params.pop(0))
            self.attention2.to(device)
        if self.model_factor[0] >= 2:
            self.bn.load_state_dict(params.pop(0))
            self.bn.to(device)
        if self.model_factor[3] == 1:
            self.hidden1.load_state_dict(params.pop(0))
            self.hidden1.to(device)
        elif self.model_factor[3] == 2:
            self.hidden1.load_state_dict(params.pop(0))
            self.hidden1.to(device)
            self.hidden2.load_state_dict(params.pop(0))
            self.hidden2.to(device)
        elif self.model_factor[3] == 3:
            self.hidden1.load_state_dict(params.pop(0))
            self.hidden1.to(device)
            self.hidden2.load_state_dict(params.pop(0))
            self.hidden2.to(device)
            self.hidden3.load_state_dict(params.pop(0))
            self.hidden3.to(device)

def get_model(model_factor, num_units1, num_units2, embedding_matrix, max_features, num_aux_targets, max_seq_len, num_feats):
    return NeuralNet(model_factor, num_units1, num_units2, embedding_matrix, max_features, num_aux_targets, max_seq_len, num_feats)
