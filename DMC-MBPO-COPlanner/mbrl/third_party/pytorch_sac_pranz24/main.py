import argparse
import itertools
import logging
import os

import gym
import numpy as np
import torch
from sac import SAC
import dmc2gym

import mbrl.constants
from mbrl.util.logger import Logger
from mbrl.util.replay_buffer import ReplayBuffer

parser = argparse.ArgumentParser(description="PyTorch Soft Actor-Critic Args")
parser.add_argument('--domain_name', default="walker",
                    help='Mujoco Gym environment (default: Hopper-v2)')
parser.add_argument('--task_name', default="run",
                    help='Mujoco Gym environment (default: Hopper-v2)')
parser.add_argument(
    "--policy",
    default="Gaussian",
    help="Policy Type: Gaussian | Deterministic (default: Gaussian)",
)
parser.add_argument(
    "--eval",
    type=bool,
    default=True,
    help="Evaluates a policy a policy every 10 episode (default: True)",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.99,
    metavar="G",
    help="discount factor for reward (default: 0.99)",
)
parser.add_argument(
    "--tau",
    type=float,
    default=0.005,
    metavar="G",
    help="target smoothing coefficient(τ) (default: 0.005)",
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.0003,
    metavar="G",
    help="learning rate (default: 0.0003)",
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.2,
    metavar="G",
    help="Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)",
)
parser.add_argument(
    "--automatic_entropy_tuning",
    type=bool,
    default=True,
    metavar="G",
    help="Automaically adjust α (default: False)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=123456,
    metavar="N",
    help="random seed (default: 123456)",
)
parser.add_argument(
    "--batch_size", type=int, default=256, metavar="N", help="batch size (default: 256)"
)
parser.add_argument(
    "--num_steps",
    type=int,
    default=1000001,
    metavar="N",
    help="maximum number of steps (default: 1000000)",
)
parser.add_argument(
    "--hidden_size",
    type=int,
    default=256,
    metavar="N",
    help="hidden size (default: 256)",
)
parser.add_argument(
    "--updates_per_step",
    type=int,
    default=1,
    metavar="N",
    help="model updates per simulator step (default: 1)",
)
parser.add_argument(
    "--start_steps",
    type=int,
    default=10000,
    metavar="N",
    help="Steps sampling random actions (default: 10000)",
)
parser.add_argument(
    "--target_update_interval",
    type=int,
    default=1,
    metavar="N",
    help="Value target update per no. of updates per step (default: 1)",
)
parser.add_argument(
    "--replay_size",
    type=int,
    default=1000000,
    metavar="N",
    help="size of replay buffer (default: 10000000)",
)
parser.add_argument(
    "--target_entropy",
    type=float,
    default=0,
    help="If given, a target entropy to use (default: none --> -dim(|A|))",
)
parser.add_argument('--exp_name', default='exp1',
                        help='your model save path')
parser.add_argument('--save_dir', default='./exp_sac/',
                    help='your model save path')
parser.add_argument("--logdir", type=str, help="Directory to log results to.")
parser.add_argument("--device", type=str)
args = parser.parse_args()




exp_name = args.exp_name
# Initial logger
args.exp_dir = os.path.join(args.save_dir, exp_name)
if not os.path.isdir(args.exp_dir):
    os.makedirs(args.exp_dir)
log_file = os.path.join(args.exp_dir, '{}.txt'.format(exp_name))
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler(log_file)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Environment
# env = NormalizedActions(gym.make(args.env_name))
# env = gym.make(args.env_name)
# env.seed(args.seed)
# env.action_space.seed(args.seed)

env = dmc2gym.make(domain_name=args.domain_name, task_name=args.task_name, seed=args.seed, visualize_reward=False)
test_env = dmc2gym.make(domain_name=args.domain_name, task_name=args.task_name, seed=args.seed, visualize_reward=False)

print(env.observation_space.shape, env.action_space.shape)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)

# Memory
memory = ReplayBuffer(
    args.replay_size,
    env.observation_space.shape,
    env.action_space.shape,
    rng=np.random.default_rng(seed=args.seed),
)

# Training Loop
total_numsteps = 0
updates = 0

# logger = Logger(args.logdir, enable_back_compatible=True)
# logger.register_group(
#     mbrl.constants.RESULTS_LOG_NAME,
#     [
#         ("episode", "E", "int"),
#         ("reward", "R", "float"),
#     ],
#     color="green",
#     dump_frequency=1,
# )

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                (
                    critic_1_loss,
                    critic_2_loss,
                    policy_loss,
                    ent_loss,
                    alpha,
                ) = agent.update_parameters(
                    memory, args.batch_size, updates, logger=logger
                )
                # if updates % 100 == 0:
                #     logger.dump(updates, save=True)

                updates += 1

        next_state, reward, done, _ = env.step(action)  # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        mask = True if episode_steps == env._max_episode_steps else not done

        memory.add(state, action, next_state, reward, mask)

        state = next_state

        if total_numsteps % 1000 == 0 and args.eval == True:
            avg_reward = 0.
            # episodes = 10
            # horizon = 0.
            # for i in range(episodes):
            #     print("--------------------episode {}--------------------".format(i))
            state = test_env.reset()
            episode_reward_ = 0
            done = False
            for h in range(1000):
                if not done:
                    action = agent.select_action(state, evaluate=True)
                    next_state, reward, done, _ = test_env.step(action)
                    episode_reward_ += reward
                    state = next_state
                else:
                    break
            # horizon += h
            avg_reward += episode_reward_
            # avg_reward /= episodes
            # horizon /= episodes

            print("----------------------------------------")
            logger.info("Total Steps: " + str(total_numsteps) + "Test Reward: " + str(avg_reward))

            print("----------------------------------------")

        if total_numsteps % 100000 == 0:
            model_file = os.path.join(args.exp_dir, 'model_last_{}.pt'.format(total_numsteps))
            # model_file = os.path.join(args.exp_dir, 'model_{}.pt'.format(total_step))
            torch.save({'Policy': agent.policy.state_dict(),
                        'Critic': agent.critic.state_dict(),
                        'CriticTarget': agent.critic_target.state_dict(),
                        }, model_file)

    if total_numsteps > args.num_steps:
        break

    # print(
    #     "Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(
    #         i_episode, total_numsteps, episode_steps, round(episode_reward, 2)
    #     )
    # )
    #
    # if i_episode % 10 == 0 and args.eval is True:
    #     avg_reward = 0.0
    #     episodes = 10
    #     for _ in range(episodes):
    #         state = env.reset()
    #         episode_reward = 0
    #         done = False
    #         while not done:
    #             action = agent.select_action(state, evaluate=True)
    #
    #             next_state, reward, done, _ = env.step(action)
    #             episode_reward += reward
    #
    #             state = next_state
    #         avg_reward += episode_reward
    #     avg_reward /= episodes
    #
    #     logger.log_data(
    #         mbrl.constants.RESULTS_LOG_NAME,
    #         {"episode": i_episode, "reward": avg_reward},
    #     )

env.close()
