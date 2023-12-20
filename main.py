# https://deeplearningcourses.com/c/cutting-edge-artificial-intelligence


from qagent import Qagent

import os

import gymnasium as gym
import argparse
import logging
from pathlib import Path
from atari_wrappers import make_atari, wrap_deepmind, Monitor
from deep_q_agent import DeepQAgent


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Mute missing instructions errors

MODEL_PATH = 'models'
SEED = 0


def get_args():
    # Get some basic command line arguements
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env', help='environment ID', default="MountainCar-v0")
    parser.add_argument('-s', '--steps', help='number of episodes', type=int, default=int(150000))
    parser.add_argument('-l', '--load_dir', help='Load pretrained agent from file', type=Path)
    parser.add_argument('-d', '--deep', action='store_true', help='Flag for using deep q learning')
    return parser.parse_args()


def train_qagent(env_id, num_episodes, load_dir):
    print(env_id)
    env = gym.make(env_id)
    gym.logger.setLevel(logging.WARN)
    qagent = Qagent(env, SEED, save_dir=load_dir)
    qagent.learn(num_episodes=num_episodes)
    env.close()

def train_deep_qagent(env_id, steps, load_dir):
    print(f"training deep q agent on env {env_id}")
    env = make_atari(env_id)
    #env = gym.make(env_id)
    env.seed(0)
    gym.logger.setLevel(logging.WARN)
    env = wrap_deepmind(env)
    #env = Monitor(env)

    dq_agent = DeepQAgent(env)
    if load_dir is not None:
        dq_agent.load_model(load_dir)
    dq_agent.learn(steps)

def main():
    args = get_args()
    os.makedirs(MODEL_PATH, exist_ok=True)
    if args.deep:
        train_deep_qagent(args.env, args.steps, args.load_dir)
    else:
        train_qagent(args.env, args.steps, args.load_dir)


if __name__ == "__main__":
    main()
