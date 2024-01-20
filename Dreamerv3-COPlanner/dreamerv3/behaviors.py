import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from . import agent
from . import expl
from . import ninjax as nj
from . import jaxutils


class Greedy(nj.Module):

  def __init__(self, wm, act_space, config):
    rewfn = lambda s: wm.heads['reward'](s).mean()[1:]
    if config.critic_type == 'vfunction':
      critics = {'extr': agent.VFunction(rewfn, config, name='critic')}
    else:
      raise NotImplementedError(config.critic_type)
    self.ac = agent.ImagActorCritic(
        critics, {'extr': 1.0}, act_space, config, name='ac')
    self.rewfn = lambda s: wm.heads['reward'](s).mean()
    self.expl_rewfn = wm.expl_rewfn
    self.config = config
    self.wm = wm

  def initial(self, batch_size):
    return self.ac.initial(batch_size)

  def policy(self, latent, state):
    if self.config.is_mpc == 1:
      state_copy = latent.copy()
      state_copy["stoch"] = jnp.repeat(state_copy["stoch"][jnp.newaxis, :, :, :], repeats=self.config.n_trajs, axis=0)
      state_copy["deter"] = jnp.repeat(state_copy["deter"][jnp.newaxis, :, :], repeats=self.config.n_trajs, axis=0)
      state_copy["logit"] = jnp.repeat(state_copy["logit"][jnp.newaxis, :, :, :], repeats=self.config.n_trajs, axis=0)
      actions_dis, _ = self.ac.policy(state_copy, {})
      actions = actions_dis['action'].sample(seed=nj.rng())
      action_candidates = actions
      mpc_state = state_copy.copy()
      mpc_actions = actions
      for h in range(self.config.coplanner_horizon):
        mpc_next_state = self.wm.rssm.img_step(mpc_state, mpc_actions)
        mpc_next_actions_dis, _ = self.ac.policy(mpc_next_state, {})
        mpc_next_actions = mpc_next_actions_dis['action'].sample(seed=nj.rng())

        state_copy["stoch"] = jnp.concatenate((state_copy["stoch"], mpc_next_state["stoch"]), axis=1)
        state_copy["deter"] = jnp.concatenate((state_copy["deter"], mpc_next_state["deter"]), axis=1)
        state_copy["logit"] = jnp.concatenate((state_copy["logit"], mpc_next_state["logit"]), axis=1)
        actions = jnp.concatenate((actions, mpc_next_actions), axis=1)
        mpc_state = mpc_next_state
        mpc_actions = mpc_next_actions

      state_copy["action"] = actions
      rew = self.rewfn(state_copy)
      rew = rew.reshape(self.config.n_trajs, self.config.coplanner_horizon + 1, -1)
      total_reward = jnp.sum(rew, axis=1)
      expl_rew = self.expl_rewfn(state_copy)
      expl_rew = expl_rew.reshape(self.config.n_trajs, self.config.coplanner_horizon + 1, -1)
      model_uncert = jnp.sum(expl_rew, axis=1)
      total_reward = total_reward + self.config.optimistic_rate * model_uncert

      index = jnp.argmax(total_reward, axis=0)
      final_actions = action_candidates[index, jnp.arange(index.shape[0]), :]
      return self.ac.policy(latent, state), final_actions
      # final_actions_dis = actions_dis[index, jnp.arange(index.shape[0]), :]
      # return self.ac.policy(latent, state), final_actions_dis, final_actions
    else:
      return self.ac.policy(latent, state)

  def train(self, imagine, start, data):
    return self.ac.train(imagine, start, data)

  def report(self, data):
    return {}


class Random(nj.Module):

  def __init__(self, wm, act_space, config):
    self.config = config
    self.act_space = act_space

  def initial(self, batch_size):
    return jnp.zeros(batch_size)

  def policy(self, latent, state):
    batch_size = len(state)
    shape = (batch_size,) + self.act_space.shape
    if self.act_space.discrete:
      dist = jaxutils.OneHotDist(jnp.zeros(shape))
    else:
      dist = tfd.Uniform(-jnp.ones(shape), jnp.ones(shape))
      dist = tfd.Independent(dist, 1)
    return {'action': dist}, state

  def train(self, imagine, start, data):
    return None, {}

  def report(self, data):
    return {}


class Explore(nj.Module):

  REWARDS = {
      'disag': expl.Disag,
  }

  def __init__(self, wm, act_space, config):
    self.config = config
    self.rewards = {}
    critics = {}
    for key, scale in config.expl_rewards.items():
      if not scale:
        continue
      if key == 'extr':
        rewfn = lambda s: wm.heads['reward'](s).mean()[1:]
        critics[key] = agent.VFunction(rewfn, config, name=key)
      else:
        rewfn = self.REWARDS[key](
            wm, act_space, config, name=key + '_reward')
        critics[key] = agent.VFunction(rewfn, config, name=key)
        self.rewards[key] = rewfn
    scales = {k: v for k, v in config.expl_rewards.items() if v}
    self.ac = agent.ImagActorCritic(
        critics, scales, act_space, config, name='ac')

  def initial(self, batch_size):
    return self.ac.initial(batch_size)

  def policy(self, latent, state):
    return self.ac.policy(latent, state)

  def train(self, imagine, start, data):
    metrics = {}
    for key, rewfn in self.rewards.items():
      mets = rewfn.train(data)
      metrics.update({f'{key}_k': v for k, v in mets.items()})
    traj, mets = self.ac.train(imagine, start, data)
    metrics.update(mets)
    return traj, metrics

  def report(self, data):
    return {}
