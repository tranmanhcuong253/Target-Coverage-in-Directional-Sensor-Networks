
from __future__ import division
import torch
import numpy as np
import torch.nn as nn
from gym import spaces
import torch.nn.functional as F
from torch.autograd import Variable

from utils import norm_col_init, weights_init
from perception import NoisyLinear, BiRNN, AttentionLayer,FirstAwareBranch,SecAwareBranch,MLP,PredictionLayer


def build_model(obs_space, action_space, args, device):
    name = args.model

    if 'single' in name:
        model = A3C_Single(obs_space, action_space, args, device)
    elif 'multi' in name:
        model = A3C_Multi(obs_space, action_space, args, device)
    elif 'fm' in name:
        model = A3C_Multi_FM(obs_space, action_space, args, device)
    model.train()
    return model


def wrap_action(self, action):
    action = np.squeeze(action)
    out = action * (self.action_high - self.action_low) / 2 + (self.action_high + self.action_low) / 2.0
    return out


def sample_action(mu_multi, sigma_multi, device, test=False):
    # discrete
    logit = mu_multi
    prob = F.softmax(logit, dim=-1)
    log_prob = F.log_softmax(logit, dim=-1)
    entropy = -(log_prob * prob).sum(-1, keepdim=True)
    if test:
        action = prob.max(-1)[1].data
        action_env = action.cpu().numpy()  # np.squeeze(action.cpu().numpy(), axis=0)
    else:
        action = prob.multinomial(1).data
        log_prob = log_prob.gather(1, Variable(action))  # [num_agent, 1] # comment for sl slave
        action_env = action.squeeze(0)

    return action_env, entropy, log_prob


class ValueNet(nn.Module):
    def __init__(self, input_dim, head_name, num=1):
        super(ValueNet, self).__init__()
        if 'ns' in head_name:
            self.noise = True
            self.critic_linear = NoisyLinear(input_dim, num, sigma_init=0.017)
        else:
            self.noise = False
            self.critic_linear = nn.Linear(input_dim, num)
            self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 0.1)
            self.critic_linear.bias.data.fill_(0)

    def forward(self, x):
        value = self.critic_linear(x)
        return value

    def sample_noise(self):
        if self.noise:
            self.critic_linear.sample_noise()

    def remove_noise(self):
        if self.noise:
            self.critic_linear.sample_noise()


class AMCValueNet(nn.Module):
    def __init__(self, input_dim, head_name, num=1, device=torch.device('cpu')):
        super(AMCValueNet, self).__init__()
        self.head_name = head_name
        self.device = device

        if 'ns' in head_name:
            self.noise = True
            self.critic_linear = NoisyLinear(input_dim, num, sigma_init=0.017)
        if 'onlyJ' in head_name:
            self.noise = False
            self.critic_linear = nn.Linear(input_dim, num)
            self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 0.1)
            self.critic_linear.bias.data.fill_(0)
        else:
            self.noise = False
            self.critic_linear = nn.Linear(2 * input_dim, num)
            self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 0.1)
            self.critic_linear.bias.data.fill_(0)

            self.attention = AttentionLayer(input_dim, input_dim, device)
        self.feature_dim = input_dim

    def forward(self, x, goal):
        _, feature_dim = x.shape
        value = []

        coalition = x.view(-1, feature_dim)
        n = coalition.shape[0]

        feature = torch.zeros([self.feature_dim]).to(self.device)
        value.append(self.critic_linear(torch.cat([feature, coalition[0]])))
        for j in range(1, n):
            _, feature = self.attention(coalition[:j].unsqueeze(0))
            value.append(self.critic_linear(torch.cat([feature.squeeze(), coalition[j]])))  # delta f = f[:j]-f[:j-1]

        # mean and sum
        value = torch.cat(value).sum()

        return value.unsqueeze(0)

    def sample_noise(self):
        if self.noise:
            self.critic_linear.sample_noise()

    def remove_noise(self):
        if self.noise:
            self.critic_linear.sample_noise()
class AMCValueNetFM(nn.Module):
    def __init__(self, input_dim, head_name, num=1, device=torch.device('cpu')):
        super(AMCValueNetFM, self).__init__()
        self.head_name = head_name
        self.device = device

        if 'ns' in head_name:
            self.noise = True
            self.critic_linear = NoisyLinear(input_dim, num, sigma_init=0.017)
        if 'onlyJ' in head_name:
            self.noise = False
            self.critic_linear = nn.Linear(input_dim, num)
            self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 0.1)
            self.critic_linear.bias.data.fill_(0)
        else:
            self.noise = False
            self.critic_linear = nn.Linear(20, num)
            self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 0.1)
            self.critic_linear.bias.data.fill_(0)

            self.attention = AttentionLayer(input_dim, input_dim, device)
        self.feature_dim = input_dim

    def forward(self, x, goal):
        
        x=x.squeeze()
        return (torch.max(x)).unsqueeze(0)
        

    def sample_noise(self):
        if self.noise:
            self.critic_linear.sample_noise()

    def remove_noise(self):
        if self.noise:
            self.critic_linear.sample_noise()

class PolicyNet(nn.Module):
    def __init__(self, input_dim, action_space, head_name, device):
        super(PolicyNet, self).__init__()
        self.head_name = head_name
        self.device = device
        num_outputs = action_space.n

        if 'ns' in head_name:
            self.noise = True
            self.actor_linear = NoisyLinear(input_dim, num_outputs, sigma_init=0.017)
        else:
            self.noise = False
            self.actor_linear = nn.Linear(input_dim, num_outputs)

            # init layers
            self.actor_linear.weight.data = norm_col_init(self.actor_linear.weight.data, 0.1)
            self.actor_linear.bias.data.fill_(0)

    def forward(self, x, test=False):
        mu = F.relu(self.actor_linear(x))
        sigma = torch.ones_like(mu)
        action, entropy, log_prob = sample_action(mu, sigma, self.device, test)
        return action, entropy, log_prob

    def sample_noise(self):
        if self.noise:
            self.actor_linear.sample_noise()
            self.actor_linear2.sample_noise()

class PolicyNetFM(nn.Module):
    def __init__(self, input_dim, action_space, head_name, device):
        super(PolicyNetFM, self).__init__()
        self.head_name = head_name
        self.device = device
        num_outputs = action_space.n
        self.out = PredictionLayer('binary', )
        if 'ns' in head_name:
            self.noise = True
        #     self.actor_linear = NoisyLinear(input_dim, num_outputs, sigma_init=0.017)
        else:
            self.noise = False
        #     self.actor_linear = nn.Linear(input_dim, num_outputs)

        #     # init layers
        #     self.actor_linear.weight.data = norm_col_init(self.actor_linear.weight.data, 0.1)
        #     self.actor_linear.bias.data.fill_(0)

    def forward(self, x, test=False):
        # mu = F.relu(self.actor_linear(x))
        # sigma = torch.ones_like(mu)
        # action, entropy, log_prob = sample_action(mu, sigma, self.device, test)
        # return action, entropy, log_prob
        prob=self.out(x)

        complement = 1 - prob
        prob = torch.cat((complement, prob), dim=1)
        log_prob = torch.log(prob)
        entropy = -(log_prob * prob).sum(-1, keepdim=True)
        if test:
          action = prob.max(-1)[1].data
          action_env = action.cpu().numpy()  # np.squeeze(action.cpu().numpy(), axis=0)
        else:
          action = prob.multinomial(1).data
          log_prob = log_prob.gather(1, Variable(action))  # [num_agent, 1] # comment for sl slave
          action_env = action.squeeze(0)

        return action_env, entropy, log_prob
    def sample_noise(self):
        if self.noise:
          pass
            # self.actor_linear.sample_noise()
            # self.actor_linear2.sample_noise()

    def remove_noise(self):
        if self.noise:
          pass
            # self.actor_linear.sample_noise()
            # self.actor_linear2.sample_noise()

class EncodeBiRNN(torch.nn.Module):
    def __init__(self, dim_in, lstm_out=128, head_name='birnn_lstm', device=None):
        super(EncodeBiRNN, self).__init__()
        self.head_name = head_name

        self.encoder = BiRNN(dim_in, int(lstm_out / 2), 1, device, 'gru')

        self.feature_dim = self.encoder.feature_dim
        self.global_feature_dim = self.encoder.feature_dim
        self.apply(weights_init)
        self.train()

    def forward(self, inputs):
        x = inputs
        cn, hn = self.encoder(x)

        feature = cn  # shape: [bs, num_camera, lstm_dim]

        global_feature = hn.permute(1, 0, 2).reshape(-1)

        return feature, global_feature

class EncodeLinear(torch.nn.Module):
    def __init__(self, dim_in, dim_out=32, head_name='lstm', device=None):
        super(EncodeLinear, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.ReLU(inplace=True),
            nn.Linear(dim_out, dim_out),
            nn.ReLU(inplace=True)
        )

        self.head_name = head_name
        self.feature_dim = dim_out
        self.train()

    def forward(self, inputs):
        x = inputs
        feature = self.features(x)
        return feature

class A3C_Single(torch.nn.Module):  # single vision Tracking
    def __init__(self, obs_space, action_spaces, args, device=torch.device('cpu')):
        super(A3C_Single, self).__init__()
        self.n = len(obs_space)
        obs_dim = obs_space[0].shape[1]

        lstm_out = args.lstm_out
        head_name = args.model

        self.head_name = head_name

        self.encoder = AttentionLayer(obs_dim, lstm_out, device)
        self.critic = ValueNet(lstm_out, head_name, 1)
        self.actor = PolicyNet(lstm_out, action_spaces[0], head_name, device)

        self.train()
        self.device = device

    def forward(self, inputs, test=False):
        data = Variable(inputs, requires_grad=True)
        _, feature = self.encoder(data)

        actions, entropies, log_probs = self.actor(feature, test)
        values = self.critic(feature)

        return values, actions, entropies, log_probs

    def sample_noise(self):
        self.actor.sample_noise()
        self.actor.sample_noise()

    def remove_noise(self):
        self.actor.remove_noise()
        self.actor.remove_noise()


class A3C_Multi(torch.nn.Module):
    def __init__(self, obs_space, action_spaces, args, device=torch.device('cpu')):
        super(A3C_Multi, self).__init__()
        self.num_agents, self.num_targets, self.pose_dim = obs_space.shape

        lstm_out = args.lstm_out
        head_name = args.model
        self.head_name = head_name

        self.encoder = EncodeLinear(self.pose_dim, lstm_out, head_name, device)
        feature_dim = self.encoder.feature_dim

        self.attention = AttentionLayer(feature_dim, lstm_out, device)
        feature_dim = self.attention.feature_dim

        # create actor & critic
        self.actor = PolicyNet(feature_dim, spaces.Discrete(2), head_name, device)

        if 'shap' in head_name:
            self.ShapleyVcritic = AMCValueNet(feature_dim, head_name, 1, device)
        else:
            self.critic = ValueNet(feature_dim, head_name, 1)

        self.train()
        self.device = device

    def forward(self, inputs, test=False):
        pos_obs = inputs

        feature_target = Variable(pos_obs, requires_grad=True)
        feature_target = self.encoder(feature_target)  # num_agent, num_target, feature_dim

        feature_target = feature_target.reshape(-1, self.encoder.feature_dim).unsqueeze(0)  # [1, agent*target, feature_dim]
        feature, global_feature = self.attention(feature_target)  # num_agents, feature_dim
        feature = feature.squeeze()

        actions, entropies, log_probs = self.actor(feature, test)
        actions = actions.reshape(self.num_agents, self.num_targets, -1)

        if 'shap' not in self.head_name:
            values = self.critic(global_feature)
        else:
            values = self.ShapleyVcritic(feature, actions)  # shape [1,1]

        return values, actions, entropies, log_probs

    def sample_noise(self):
        self.actor.sample_noise()
        self.actor.sample_noise()

    def remove_noise(self):
        self.actor.remove_noise()
        self.actor.remove_noise()
class FM(nn.Module):
    '''Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    '''

    def __init__(self):
        super(FM, self).__init__()

    def forward(self, inputs):
        fm_input = inputs

        square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)

        return cross_term
class A3C_Multi_FM(torch.nn.Module):
    def __init__(self, obs_space, action_spaces, args, device=torch.device('cpu')):
        super(A3C_Multi_FM, self).__init__()
        self.num_agents, self.num_targets, self.pose_dim = obs_space.shape

        lstm_out = args.lstm_out
        head_name = args.model
        self.head_name = head_name

        self.encoder = EncodeLinear(self.pose_dim, lstm_out, head_name, device)
        feature_dim = self.encoder.feature_dim

        self.attention = AttentionLayer(feature_dim, lstm_out, device)
        feature_dim = self.attention.feature_dim

        # create actor & critic
        self.actor = PolicyNetFM(feature_dim, spaces.Discrete(2), head_name, device)
        self.embedding = nn.Embedding(43 , 10,device=device)
        self.embedding.weight.data.uniform_(-.1, .1)
        torch.nn.init.xavier_normal_(self.embedding.weight.data, gain=1e-3)

        self.embedding2 = nn.Embedding(43 , 1,device=device)
        self.embedding2.weight.data.uniform_(-.1, .1)
        torch.nn.init.xavier_normal_(self.embedding2.weight.data, gain=1e-3)

        self.sec_integrate = SecAwareBranch()

        self.first_integrate = FirstAwareBranch(5, 10, device = device)

        self.first_aware = MLP(50,
                                (128,), activation='relu', l2_reg=0,
                                dropout_rate=0.1,
                                use_ln=True, init_std=0.0001, device=device)

        self.sec_aware = MLP(50,
                                (128,), activation='relu', l2_reg=0,
                                dropout_rate=0.1,
                                use_ln=True, init_std=0.0001, device=device)

        self.first_reweight = nn.Linear(
            128, 5, bias=False).to(device)
        self.sec_reweight = nn.Linear(
            128, 5, bias=False).to(device)

        self.linear=nn.Linear(1,128).to(device)
        self.relu = nn.ReLU().to(device)

        self.ln = nn.LayerNorm(50).to(device)
        self.ln2 = nn.LayerNorm(128).to(device)
        self.regularization_weight = []
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.first_aware.named_parameters()),
            l2=0)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.sec_aware.named_parameters()),
            l2=0)

        self.add_regularization_weight(self.sec_reweight.weight, l2=0)
        self.add_regularization_weight(self.first_reweight.weight, l2=0)
        self.add_regularization_weight(self.linear.weight, l2=0)

        self.fm = FM()
        if 'shap' in head_name:
            self.ShapleyVcritic = AMCValueNetFM(feature_dim, head_name, 1, device)
        else:
            self.critic = ValueNet(feature_dim, head_name, 1)

        self.train()
        self.device = device
    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        # For generators, filters and ParameterLists, convert them to a list of tensors to avoid bugs.
        # e.g., we can't pickle generator objects when we save the model.
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def forward(self, inputs, test=False):
        inputs=torch.reshape(inputs,(20,4))

        inputs = Variable(inputs, requires_grad=True)
        pos_obs=[]
        target_offset = torch.tensor(4)
        angle_offset = torch.tensor(8)
        distance_offset = torch.tensor(20)
        visible_offset =torch.tensor(41)
        for obs in inputs:
          sensor = obs[0] * 4
          target = obs[1] * 5
          angle = torch.ceil(torch.abs(obs[2]) *(180.0/15.0))
          distance = torch.clip(torch.ceil(obs[3] *(2000.0/100)),max=20)
          if (torch.abs(obs[2]) < (45.0/180.0)) and (obs[3] <= (800.0/2000.0)):
            visible=torch.tensor(1)
          else:
            visible=torch.tensor(0)

          pos_obs.append(torch.Tensor([sensor,target + target_offset,angle + angle_offset,distance + distance_offset,visible+visible_offset]))

        pos_obs = torch.stack(pos_obs).type(torch.int32).to(self.device)
        embed = self.embedding(pos_obs).to(self.device)
        sec_out = self.sec_integrate(embed)
        sec_out = self.ln(sec_out)
        sec_out = self.sec_aware(sec_out)
        m_sec = self.sec_reweight(sec_out)

        first_out = self.first_integrate(embed)
        first_out = self.ln(first_out)
        first_out = self.first_aware(first_out)
        m_first= self.first_reweight(first_out)

        m_final = m_sec + m_first

        refined = embed * m_final.unsqueeze(-1)

        feature = self.fm(refined)

        embed2 = self.embedding2(pos_obs).to(self.device)
        embed2 = embed2.reshape(20,1,-1)
        embed2 = embed2*m_final.unsqueeze(1)
        logit = torch.sum(embed2, dim=-1, keepdim=False)

        feature+=logit
        
        #feature = self.ln2(self.relu(self.linear(feature)))
        
        actions, entropies, log_probs = self.actor(feature, test)
        actions = actions.reshape(self.num_agents, self.num_targets, -1)

        # if 'shap' not in self.head_name:
        #     values = self.critic(global_feature)
        # else:
        values = self.ShapleyVcritic(feature, actions)  # shape [1,1]

        return values, actions, entropies, log_probs

    def sample_noise(self):
        self.actor.sample_noise()
        self.actor.sample_noise()

    def remove_noise(self):
        self.actor.remove_noise()
        self.actor.remove_noise()

