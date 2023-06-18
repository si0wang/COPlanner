import time

import torch
from torch import nn
from torch import distributions as torchd
from torch.optim import Adam

import models
import networks
import tools


class Random(nn.Module):

  def __init__(self, config):
    self._config = config

  def actor(self, feat):
    shape = feat.shape[:-1] + [self._config.num_actions]
    if self._config.actor_dist == 'onehot':
      return tools.OneHotDist(torch.zeros(shape))
    else:
      ones = torch.ones(shape)
      return tools.ContDist(torchd.uniform.Uniform(-ones, ones))

  def train(self, start, context):
    return None, {}

class Ensemble(nn.Module):
  def __init__(self, networks):
    super(Ensemble, self).__init__()
    self.networks = nn.ModuleList(networks)

  def forward(self, x):
    outputs = []
    for net in self.networks:
      outputs.append(net(x))
    return outputs

class Plan2Explore(nn.Module):
  def __init__(self, config, world_model, reward=None):
    super(Plan2Explore, self).__init__()
    self._config = config
    self._reward = reward
    stoch_size = config.dyn_stoch
    if config.dyn_discrete:
      stoch_size *= config.dyn_discrete
    size = {
      "embed": 32 * config.cnn_depth,
      "stoch": stoch_size,
      "deter": config.dyn_deter,
      "feat": config.dyn_stoch + config.dyn_deter,
    }[self._config.disag_target]
    kw = dict(
      inp_dim=config.dyn_stoch + config.dyn_deter + config.num_actions,  # pytorch version
      shape=size,
      layers=config.disag_layers,
      units=config.disag_units,
      act=config.act,
    )
    self._networks = Ensemble([networks.DenseHead(**kw) for _ in range(config.disag_models)])

    self._opt = Adam(self._networks.parameters(), lr=config.expl_model_lr, eps=config.expl_eps, weight_decay=config.expl_weight_decay)


  def train(self, start, context, data):
    stoch = start["stoch"]
    if self._config.dyn_discrete:
      stoch = stoch.view(*stoch.shape[:-2], stoch.shape[-2] * stoch.shape[-1])
    target = {
      "embed": context["embed"],
      "stoch": stoch,
      "deter": start["deter"],
      "feat": context["feat"], # stoch + deter
    }[self._config.disag_target]
    inputs = context["feat"]
    if self._config.disag_action_cond:
      inputs = torch.concat([inputs, torch.from_numpy(data["action"]).to(torch.device('cuda'))], -1)

    if self._config.disag_offset:
      targets = target[:, self._config.disag_offset:]
      inputs = inputs[:, :-self._config.disag_offset]
    targets = targets.detach()
    inputs = inputs.detach()
    preds = self._networks(inputs)
    likes = [torch.mean(pred.log_prob(targets)) for pred in preds]
    loss = -torch.sum(torch.tensor(likes).float())

    loss.requires_grad_(requires_grad=True)

    self._opt.zero_grad()
    (loss).backward()
    self._opt.step()
    return loss

  def _intrinsic_reward(self, feat, state, action):
    inputs = feat
    if self._config.disag_action_cond:
      inputs = torch.cat([inputs, action], -1)
    preds = self._networks(inputs)
    preds = [pred.mode() for pred in preds]
    disag = torch.mean(torch.stack(preds).std(dim=0), -1)

    # if self._config.disag_log: ## default Ture
    # disag = torch.log(disag)
    reward = self._config.expl_intr_scale * disag

    return reward
