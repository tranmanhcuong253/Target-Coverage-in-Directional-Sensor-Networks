
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable
from deepctr_torch.layers.activation import activation_layer

class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=True)
        self.sigma_init = sigma_init
        self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))
        self.sigma_bias = Parameter(torch.Tensor(out_features))
        self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
        self.register_buffer('epsilon_bias', torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'sigma_weight'):
            init.uniform(self.weight, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
            init.uniform(self.bias, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
            init.constant(self.sigma_weight, self.sigma_init)
            init.constant(self.sigma_bias, self.sigma_init)

    def forward(self, input):
        return F.linear(input, self.weight + self.sigma_weight * Variable(self.epsilon_weight),
                        self.bias + self.sigma_bias * Variable(self.epsilon_bias))

    def sample_noise(self):
        self.epsilon_weight = torch.randn(self.out_features, self.in_features)
        self.epsilon_bias = torch.randn(self.out_features)

    def remove_noise(self):
        self.epsilon_weight = torch.zeros(self.out_features, self.in_features)
        self.epsilon_bias = torch.zeros(self.out_features)


class BiRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device, head_name):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if 'lstm' in head_name:
            self.lstm = True
        else:
            self.lstm = False
        if self.lstm:
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True).to(device)
        else:
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True).to(device)
        self.feature_dim = hidden_size * 2
        self.device = device

    def forward(self, x, state=None):
        # Set initial states

        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)

        # Forward propagate LSTM
        if self.lstm:
            out, (_, hn) = self.rnn(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        else:
            out, hn = self.rnn(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        return out, hn

class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device, head_name):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if 'lstm' in head_name:
            self.lstm = True
        else:
            self.lstm = False
        if self.lstm:
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)
        else:
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True).to(device)
        self.feature_dim = hidden_size
        self.device = device

    def forward(self, x, state=None):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        # Forward propagate LSTM
        if self.lstm:
            out, (_, hn) = self.rnn(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        else:
            out, hn = self.rnn(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        return out, hn


def xavier_init(layer):
    torch.nn.init.xavier_uniform_(layer.weight)
    torch.nn.init.constant_(layer.bias, 0)
    return layer


class AttentionLayer(torch.nn.Module):
    def __init__(self, feature_dim, weight_dim, device):
        super(AttentionLayer, self).__init__()
        self.in_dim = feature_dim
        self.device = device

        self.Q = xavier_init(nn.Linear(self.in_dim, weight_dim))
        self.K = xavier_init(nn.Linear(self.in_dim, weight_dim))
        self.V = xavier_init(nn.Linear(self.in_dim, weight_dim))

        self.feature_dim = weight_dim

    def forward(self, x):
        """
        inference
        :param x: [num_agent, num_target, feature_dim]
        :return z: [num_agent, num_target, weight_dim]
        """
        # z = softmax(Q,K)*V
        q = torch.tanh(self.Q(x))  # [batch_size, sequence_len, weight_dim]
        k = torch.tanh(self.K(x))  # [batch_size, sequence_len, weight_dim]
        v = torch.tanh(self.V(x))  # [batch_size, sequence_len, weight_dim]

        z = torch.bmm(F.softmax(torch.bmm(q, k.permute(0, 2, 1)), dim=2), v)  # [batch_size, sequence_len, weight_dim]

        global_feature = z.sum(dim=1)
        return z, global_feature


class FirstAwareBranch(nn.Module):
    """
    First order-aware Component

    Arguments
    - **sparse_feat_num**: number of feature.
    - **embedding_size**: embedding size.
    - **seed**: A Python integer to use as random seed.
    """
    def __init__(self, sparse_feat_num, embedding_size, seed=1024, device='cpu'):
        super(FirstAwareBranch, self).__init__()
        self.W = nn.Parameter(torch.Tensor(sparse_feat_num, embedding_size))
        self.seed = seed

        for tensor in self.parameters():
            nn.init.normal_(tensor, mean=0.0, std=0.05)

        self.to(device)

    def forward(self, inputs):
        outputs = torch.mul(inputs, self.W)
        outputs = outputs.reshape(outputs.shape[0], -1)
        return outputs

class SecAwareBranch(nn.Module):
    """
    Second order-aware Component
    """

    def __init__(self):
        super(SecAwareBranch, self).__init__()

    def forward(self, inputs):
        field_size = inputs.shape[1]

        fm_input = inputs.unsqueeze(1).repeat(1, field_size, 1, 1)
        square = torch.pow(inputs, 2)

        cross_term = torch.sum(torch.mul(fm_input, inputs.unsqueeze(2)), dim = 2)
        cross_term = cross_term - square
        cross_term = cross_term.reshape(cross_term.shape[0], -1)
        return cross_term

class MLP(nn.Module):

    """The Multi Layer Percetron
      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.
      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.

      Arguments
        - **inputs_dim**: input feature dimension.
        - **hidden_units**:list of positive integer, the layer number and units in each layer.
        - **activation**: Activation function to use.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.
        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.
        - **use_ln**: bool. Whether use LayerNormalization before activation or not.
        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, inputs_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_ln=False,
                 init_std=0.0001, dice_dim=3, seed=1024, device='cpu'):
        super(MLP, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_ln = use_ln

        if use_ln:
            self.ln = nn.ModuleList(
                    [nn.LayerNorm(hidden_units[i]) for i in range(len(hidden_units))])

        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)

        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])


        self.activation_layers = nn.ModuleList(
            [activation_layer(activation, hidden_units[i + 1], dice_dim) for i in range(len(hidden_units) - 1)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

        self.to(device)

    def forward(self, inputs):
        deep_input = inputs

        for i in range(len(self.linears)):
            fc = self.linears[i](deep_input)

            if self.use_ln:
                fc = self.ln[i](fc)

            fc = self.activation_layers[i](fc)
            fc = self.dropout(fc)

            deep_input = fc
        return deep_input
class PredictionLayer(nn.Module):
    """
      Arguments
         - **task**: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
         - **use_bias**: bool.Whether add bias term or not.
    """

    def __init__(self, task='binary', use_bias=True, **kwargs):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary,multiclass or regression")

        super(PredictionLayer, self).__init__()
        self.use_bias = use_bias
        self.task = task
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros((1,)))
    def forward(self, X):
        output = X
        if self.use_bias:
            output += self.bias
        if self.task == "binary":
            output = torch.sigmoid(output)
        return output
