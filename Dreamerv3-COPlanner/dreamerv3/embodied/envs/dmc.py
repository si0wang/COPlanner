import functools
import os

import embodied
import numpy as np


class DMC(embodied.Env):

  DEFAULT_CAMERAS = dict(
      locom_rodent=1,
      quadruped=2,
  )

  def __init__(self, env, repeat=1, render=True, size=(64, 64), camera=-1):
    # TODO: This env variable is meant for headless GPU machines but may fail
    # on CPU-only machines.
    if 'MUJOCO_GL' not in os.environ:
      os.environ['MUJOCO_GL'] = 'egl'
    if isinstance(env, str):
      domain, task = env.split('_', 1)
      if camera == -1:
        camera = self.DEFAULT_CAMERAS.get(domain, 0)
      if domain == 'cup':  # Only domain with multiple words.
        domain = 'ball_in_cup'
      if domain == 'manip':
        from dm_control import manipulation
        env = manipulation.load(task + '_vision')
      elif domain == 'locom':
        from dm_control.locomotion.examples import basic_rodent_2020
        env = getattr(basic_rodent_2020, task)()
      elif env == 'reach_duplo':
        name = f'{domain}_{task}_vision'
        from dm_control import manipulation
        env = manipulation.load(name)
      else:
        from dm_control import suite
        env = suite.load(domain, task)
    self._dmenv = env
    from . import from_dm
    self._env = from_dm.FromDM(self._dmenv)
    self._env = embodied.wrappers.ExpandScalars(self._env)
    self._env = embodied.wrappers.ActionRepeat(self._env, repeat)
    self._render = render
    self._size = size
    self._camera = camera

  @functools.cached_property
  def obs_space(self):
    spaces = self._env.obs_space.copy()
    if self._render:
      spaces['image'] = embodied.Space(np.uint8, self._size + (3,))
    return spaces

  @functools.cached_property
  def act_space(self):
    return self._env.act_space

  def step(self, action):
    for key, space in self.act_space.items():
      if not space.discrete:
        assert np.isfinite(action[key]).all(), (key, action[key])
    obs = self._env.step(action)
    if self._render:
      obs['image'] = self.render()
    return obs

  def render(self):
    return self._dmenv.physics.render(*self._size, camera_id=self._camera)

def make(name, frame_stack, action_repeat, seed):
  domain, task = name.split('_', 1)
  # overwrite cup to ball_in_cup
  domain = dict(cup='ball_in_cup').get(domain, domain)
  # make sure reward is not visualized
  if (domain, task) in suite.ALL_TASKS:
    env = suite.load(domain,
                     task,
                     task_kwargs={'random': seed},
                     visualize_reward=False)
    pixels_key = 'pixels'
  else:
    name = f'{domain}_{task}_vision'
    env = manipulation.load(name, seed=seed)
    pixels_key = 'front_close'
  # add wrappers
  env = ActionDTypeWrapper(env, np.float32)
  env = ActionRepeatWrapper(env, action_repeat)
  env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
  # add renderings for clasical tasks
  if (domain, task) in suite.ALL_TASKS:
    # zoom in camera for quadruped
    camera_id = dict(quadruped=2).get(domain, 0)
    render_kwargs = dict(height=84, width=84, camera_id=camera_id)
    env = pixels.Wrapper(env,
                         pixels_only=True,
                         render_kwargs=render_kwargs)
  # stack several frames
  env = ExtendedTimeStepWrapper(env)
  return env