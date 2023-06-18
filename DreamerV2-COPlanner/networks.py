import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd

import tools


class RSSM(nn.Module):

  def __init__(
      self, stoch=30, deter=200, hidden=200, layers_input=1, layers_output=1,
      rec_depth=1, shared=False, discrete=False, act=nn.ELU,
      mean_act='none', std_act='softplus', temp_post=True, min_std=0.1,
      cell='gru',
      num_actions=None, embed = None, device=None):
    super(RSSM, self).__init__()
    self._stoch = stoch
    self._deter = deter
    self._hidden = hidden
    self._min_std = min_std
    self._layers_input = layers_input
    self._layers_output = layers_output
    self._rec_depth = rec_depth
    self._shared = shared
    self._discrete = discrete
    self._act = act
    self._mean_act = mean_act
    self._std_act = std_act
    self._temp_post = temp_post
    self._embed = embed
    self._device = device

    inp_layers = []
    if self._discrete:
      inp_dim = self._stoch * self._discrete + num_actions
    else:
      inp_dim = self._stoch + num_actions
    if self._shared:
      inp_dim += self._embed
    for i in range(self._layers_input):
      inp_layers.append(nn.Linear(inp_dim, self._hidden))
      inp_layers.append(self._act())
      if i == 0:
        inp_dim = self._hidden
    self._inp_layers = nn.Sequential(*inp_layers)

    if cell == 'gru':
      self._cell = GRUCell(self._hidden, self._deter)
    elif cell == 'gru_layer_norm':
      self._cell = GRUCell(self._hidden, self._deter, norm=True)
    else:
      raise NotImplementedError(cell)

    img_out_layers = []
    inp_dim = self._deter
    for i in range(self._layers_output):
      img_out_layers.append(nn.Linear(inp_dim, self._hidden))
      img_out_layers.append(self._act())
      if i == 0:
        inp_dim = self._hidden
    self._img_out_layers = nn.Sequential(*img_out_layers)

    obs_out_layers = []
    if self._temp_post:
      inp_dim = self._deter + self._embed
    else:
      inp_dim = self._embed
    for i in range(self._layers_output):
      obs_out_layers.append(nn.Linear(inp_dim, self._hidden))
      obs_out_layers.append(self._act())
      if i == 0:
        inp_dim = self._hidden
    self._obs_out_layers = nn.Sequential(*obs_out_layers)

    if self._discrete:
      self._ims_stat_layer = nn.Linear(self._hidden, self._stoch*self._discrete)
      self._obs_stat_layer = nn.Linear(self._hidden, self._stoch*self._discrete)
    else:
      self._ims_stat_layer = nn.Linear(self._hidden, 2*self._stoch)
      self._obs_stat_layer = nn.Linear(self._hidden, 2*self._stoch)
 
  def initial(self, batch_size):
    deter = torch.zeros(batch_size, self._deter).to(self._device)
    if self._discrete:
      state = dict(
          logit=torch.zeros([batch_size, self._stoch, self._discrete]).to(self._device),
          stoch=torch.zeros([batch_size, self._stoch, self._discrete]).to(self._device),
          deter=deter)
    else:
      state = dict(
          mean=torch.zeros([batch_size, self._stoch]).to(self._device),
          std=torch.zeros([batch_size, self._stoch]).to(self._device),
          stoch=torch.zeros([batch_size, self._stoch]).to(self._device),
          deter=deter)
    return state

  def observe(self, embed, action, state=None):
    swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(action.shape[0])
    embed, action = swap(embed), swap(action)
    post, prior = tools.static_scan(
        lambda prev_state, prev_act, embed: self.obs_step(
            prev_state[0], prev_act, embed),
        (action, embed), (state, state))
    post = {k: swap(v) for k, v in post.items()}
    prior = {k: swap(v) for k, v in prior.items()}
    return post, prior

  def imagine(self, action, state=None):
    swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(action.shape[0])
    assert isinstance(state, dict), state
    action = action
    action = swap(action)
    prior = tools.static_scan(self.img_step, [action], state)
    prior = prior[0]
    prior = {k: swap(v) for k, v in prior.items()}
    return prior

  def get_feat(self, state):
    stoch = state['stoch']
    if self._discrete:
      shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
      stoch = stoch.reshape(shape)
    return torch.cat([stoch, state['deter']], -1)

  def get_dist(self, state, dtype=None):
    if self._discrete:
      logit = state['logit']
      dist = torchd.independent.Independent(tools.OneHotDist(logit), 1)
    else:
      mean, std = state['mean'], state['std']
      dist = tools.ContDist(torchd.independent.Independent(
          torchd.normal.Normal(mean, std), 1))
    return dist

  def obs_step(self, prev_state, prev_action, embed, sample=True):
    prior = self.img_step(prev_state, prev_action, None, sample)
    if self._shared:
      post = self.img_step(prev_state, prev_action, embed, sample)
    else:
      if self._temp_post:
        x = torch.cat([prior['deter'], embed], -1)
      else:
        x = embed
      x = self._obs_out_layers(x)
      stats = self._suff_stats_layer('obs', x)
      if sample:
        stoch = self.get_dist(stats).sample()
      else:
        stoch = self.get_dist(stats).mode()
      post = {'stoch': stoch, 'deter': prior['deter'], **stats}
    return post, prior

  def img_step(self, prev_state, prev_action, embed=None, sample=True):
    prev_stoch = prev_state['stoch']
    if self._discrete:
      shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
      prev_stoch = prev_stoch.reshape(shape)
    if self._shared:
      if embed is None:
        shape = list(prev_action.shape[:-1]) + [self._embed]
        embed = torch.zeros(shape)
      x = torch.cat([prev_stoch, prev_action, embed], -1)
    else:
      x = torch.cat([prev_stoch, prev_action], -1)
    x = self._inp_layers(x)
    for _ in range(self._rec_depth): # rec depth is not correctly implemented
      deter = prev_state['deter']
      x, deter = self._cell(x, [deter])
      deter = deter[0]  # Keras wraps the state in a list.
    x = self._img_out_layers(x)
    stats = self._suff_stats_layer('ims', x)
    if sample:
      stoch = self.get_dist(stats).sample()
    else:
      stoch = self.get_dist(stats).mode()
    prior = {'stoch': stoch, 'deter': deter, **stats}
    return prior

  def _suff_stats_layer(self, name, x):
    if self._discrete:
      if name == 'ims':
        x = self._ims_stat_layer(x)
      elif name == 'obs':
        x = self._obs_stat_layer(x)
      else:
        raise NotImplementedError
      logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
      return {'logit': logit}
    else:
      if name == 'ims':
        x = self._ims_stat_layer(x)
      elif name == 'obs':
        x = self._obs_stat_layer(x)
      else:
        raise NotImplementedError
      mean, std = torch.split(x, [self._stoch]*2, -1)
      mean = {
          'none': lambda: mean,
          'tanh5': lambda: 5.0 * torch.tanh(mean / 5.0),
      }[self._mean_act]()
      std = {
          'softplus': lambda: torch.softplus(std),
          'abs': lambda: torch.abs(std + 1),
          'sigmoid': lambda: torch.sigmoid(std),
          'sigmoid2': lambda: 2 * torch.sigmoid(std / 2),
      }[self._std_act]()
      std = std + self._min_std
      return {'mean': mean, 'std': std}

  def kl_loss(self, post, prior, forward, balance, free, scale):
    kld = torchd.kl.kl_divergence
    dist = lambda x: self.get_dist(x)
    sg = lambda x: {k: v.detach() for k, v in x.items()}
    lhs, rhs = (prior, post) if forward else (post, prior)
    mix = balance if forward else (1 - balance)
    if balance == 0.5:
      value = kld(dist(lhs) if self._discrete else dist(lhs)._dist,
                  dist(rhs) if self._discrete else dist(rhs)._dist)
      loss = torch.mean(torch.maximum(value, free))
    else:
      value_lhs = value = kld(dist(lhs) if self._discrete else dist(lhs)._dist,
                              dist(sg(rhs)) if self._discrete else dist(sg(rhs))._dist)
      value_rhs = kld(dist(sg(lhs)) if self._discrete else dist(sg(lhs))._dist,
                      dist(rhs) if self._discrete else dist(rhs)._dist)
      loss_lhs = torch.maximum(torch.mean(value_lhs), torch.Tensor([free])[0])
      loss_rhs = torch.maximum(torch.mean(value_rhs), torch.Tensor([free])[0])
      loss = mix * loss_lhs + (1 - mix) * loss_rhs
    loss *= scale
    return loss, value


class ConvEncoder(nn.Module):

  def __init__(self, grayscale=False,
               depth=32, act=nn.ReLU, kernels=(4, 4, 4, 4)):
    super(ConvEncoder, self).__init__()
    self._act = act
    self._depth = depth
    self._kernels = kernels

    layers = []
    for i, kernel in enumerate(self._kernels):
      if i == 0:
        if grayscale:
          inp_dim = 1
        else:
          inp_dim = 3
      else:
        inp_dim = 2 ** (i-1) * self._depth
      depth = 2 ** i * self._depth
      layers.append(nn.Conv2d(inp_dim, depth, kernel, 2))
      layers.append(act())
    self.layers = nn.Sequential(*layers)

  def __call__(self, obs):
    x = obs['image'].reshape((-1,) + tuple(obs['image'].shape[-3:]))
    x = x.permute(0, 3, 1, 2)
    x = self.layers(x)
    x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
    shape = list(obs['image'].shape[:-3]) + [x.shape[-1]]
    print(shape)
    return x.reshape(shape)


class ConvDecoder(nn.Module):

  def __init__(
      self, inp_depth,
      depth=32, act=nn.ReLU, shape=(3, 64, 64), kernels=(5, 5, 6, 6),
      thin=True):
    super(ConvDecoder, self).__init__()
    self._inp_depth = inp_depth
    self._act = act
    self._depth = depth
    self._shape = shape
    self._kernels = kernels
    self._thin = thin

    if self._thin:
      self._linear_layer = nn.Linear(inp_depth, 32 * self._depth)
    else:
      self._linear_layer = nn.Linear(inp_depth, 128 * self._depth)
    inp_dim = 32 * self._depth

    cnnt_layers = []
    for i, kernel in enumerate(self._kernels):
      depth = 2 ** (len(self._kernels) - i - 2) * self._depth
      act = self._act
      if i == len(self._kernels) - 1:
        #depth = self._shape[-1]
        depth = self._shape[0]
        act = None
      if i != 0:
        inp_dim = 2 ** (len(self._kernels) - (i-1) - 2) * self._depth
      cnnt_layers.append(nn.ConvTranspose2d(inp_dim, depth, kernel, 2))
      if act is not None:
        cnnt_layers.append(act())
    self._cnnt_layers = nn.Sequential(*cnnt_layers)

  def __call__(self, features, dtype=None):
    if self._thin:
      x = self._linear_layer(features)
      x = x.reshape([-1, 1, 1, 32 * self._depth])
      x = x.permute(0,3,1,2)
    else:
      x = self._linear_layer(features)
      x = x.reshape([-1, 2, 2, 32 * self._depth])
      x = x.permute(0,3,1,2)
    x = self._cnnt_layers(x)
    mean = x.reshape(features.shape[:-1] + self._shape)
    mean = mean.permute(0, 1, 3, 4, 2)
    return tools.ContDist(torchd.independent.Independent(
      torchd.normal.Normal(mean, 1), len(self._shape)))


class DenseHead(nn.Module):

  def __init__(
      self, inp_dim,
      shape, layers, units, act=nn.ELU, dist='normal', std=1.0):
    super(DenseHead, self).__init__()
    self._shape = (shape,) if isinstance(shape, int) else shape
    if len(self._shape) == 0:
      self._shape = (1,)
    self._layers = layers
    self._units = units
    self._act = act
    self._dist = dist
    self._std = std

    mean_layers = []
    for index in range(self._layers):
      mean_layers.append(nn.Linear(inp_dim, self._units))
      mean_layers.append(act())
      if index == 0:
        inp_dim = self._units
    mean_layers.append(nn.Linear(inp_dim, np.prod(self._shape)))
    self._mean_layers = nn.Sequential(*mean_layers)

    if self._std == 'learned':
      self._std_layer = nn.Linear(self._units, np.prod(self._shape))

  def __call__(self, features, dtype=None):
    x = features
    mean = self._mean_layers(x)
    if self._std == 'learned':
      std = self._std_layer(x)
      std = torch.softplus(std) + 0.01
    else:
      std = self._std
    if self._dist == 'normal':
      return tools.ContDist(torchd.independent.Independent(
        torchd.normal.Normal(mean, std), len(self._shape)))
    if self._dist == 'huber':
      return tools.ContDist(torchd.independent.Independent(
          tools.UnnormalizedHuber(mean, std, 1.0), len(self._shape)))
    if self._dist == 'binary':
      return tools.Bernoulli(torchd.independent.Independent(
        torchd.bernoulli.Bernoulli(logits=mean), len(self._shape)))
    raise NotImplementedError(self._dist)


class ActionHead(nn.Module):

  def __init__(
      self, inp_dim, size, layers, units, act=nn.ELU, dist='trunc_normal',
      init_std=0.0, min_std=0.1, action_disc=5, temp=0.1, outscale=0):
    super(ActionHead, self).__init__()
    self._size = size
    self._layers = layers
    self._units = units
    self._dist = dist
    self._act = act
    self._min_std = min_std
    self._init_std = init_std
    self._action_disc = action_disc
    self._temp = temp() if callable(temp) else temp
    self._outscale = outscale

    pre_layers = []
    for index in range(self._layers):
      pre_layers.append(nn.Linear(inp_dim, self._units))
      pre_layers.append(act())
      if index == 0:
        inp_dim = self._units
    self._pre_layers = nn.Sequential(*pre_layers)

    if self._dist in ['tanh_normal','tanh_normal_5','normal','trunc_normal']:
      self._dist_layer = nn.Linear(self._units, 2 * self._size)
    elif self._dist in ['normal_1','onehot','onehot_gumbel']:
      self._dist_layer = nn.Linear(self._units, self._size)

  def __call__(self, features, dtype=None):
    x = features
    x = self._pre_layers(x)
    if self._dist == 'tanh_normal':
      x = self._dist_layer(x)
      mean, std = torch.split(x, 2, -1)
      mean = torch.tanh(mean)
      std = F.softplus(std + self._init_std) + self._min_std
      dist = torchd.normal.Normal(mean, std)
      dist = torchd.transformed_distribution.TransformedDistribution(
          dist, tools.TanhBijector())
      dist = torchd.independent.Independent(dist, 1)
      dist = tools.SampleDist(dist)
    elif self._dist == 'tanh_normal_5':
      x = self._dist_layer(x)
      mean, std = torch.split(x, 2, -1)
      mean = 5 * torch.tanh(mean / 5)
      std = F.softplus(std + 5) + 5
      dist = torchd.normal.Normal(mean, std)
      dist = torchd.transformed_distribution.TransformedDistribution(
          dist, tools.TanhBijector())
      dist = torchd.independent.Independent(dist, 1)
      dist = tools.SampleDist(dist)
    elif self._dist == 'normal':
      x = self._dist_layer(x)
      mean, std = torch.split(x, 2, -1)
      std = F.softplus(std + self._init_std) + self._min_std
      dist = torchd.normal.Normal(mean, std)
      dist = tools.ContDist(torchd.independent.Independent(dist, 1))
    elif self._dist == 'normal_1':
      x = self._dist_layer(x)
      dist = torchd.normal.Normal(mean, 1)
      dist = tools.ContDist(torchd.independent.Independent(dist, 1))
    elif self._dist == 'trunc_normal':
      x = self._dist_layer(x)
      mean, std = torch.split(x, [self._size]*2, -1)
      mean = torch.tanh(mean)
      std = 2 * torch.sigmoid(std / 2) + self._min_std
      dist = tools.SafeTruncatedNormal(mean, std, -1, 1)
      dist = tools.ContDist(torchd.independent.Independent(dist, 1))
    elif self._dist == 'onehot':
      x = self._dist_layer(x)
      dist = tools.OneHotDist(x)
    elif self._dist == 'onehot_gumble':
      x = self._dist_layer(x)
      temp = self._temp
      dist = tools.ContDist(torchd.gumbel.Gumbel(x, 1/temp))
    else:
      raise NotImplementedError(self._dist)
    return dist


class GRUCell(nn.Module):

  def __init__(self, inp_size,
               size, norm=False, act=torch.tanh, update_bias=-1):
    super(GRUCell, self).__init__()
    self._inp_size = inp_size
    self._size = size
    self._act = act
    self._norm = norm
    self._update_bias = update_bias
    self._layer = nn.Linear(inp_size+size, 3*size,
                            bias=norm is not None)
    if norm:
      self._norm = nn.LayerNorm(3*size)

  @property
  def state_size(self):
    return self._size

  def forward(self, inputs, state):
    state = state[0]  # Keras wraps the state in a list.
    parts = self._layer(torch.cat([inputs, state], -1))
    if self._norm:
      parts = self._norm(parts)
    reset, cand, update = torch.split(parts, [self._size]*3, -1)
    reset = torch.sigmoid(reset)
    cand = self._act(reset * cand)
    update = torch.sigmoid(update + self._update_bias)
    output = update * cand + (1 - update) * state
    return output, [output]
