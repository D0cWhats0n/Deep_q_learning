# https://deeplearningcourses.com/c/cutting-edge-artificial-intelligence
import argparse
import os
import numpy as np
from atari_wrappers import make_atari, wrap_deepmind, Monitor
from qagent import Qagent
import gymnasium as gym
import glob
from pathlib import Path
from deep_q_agent import DeepQAgent


def get_args():
    # Get some basic command line arguements
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env', help='environment ID', default='MountainCar-v0')
    parser.add_argument('-l', '--load_dir', help='Load pretrained agent from file', type=Path)
    return parser.parse_args()


def get_agent(env, load_dir="agent"):
    if Path(load_dir).exist():
        file_names = [el.stem for el in glob.glob(".")]
        if "model.pt" in file_names:
            agent = DeepQAgent(env, epsilon=0.0, save_path=load_dir)
        else:
            agent = Qagent(env, 42, epsilon=0.0, save_dir=load_dir)
    else:
        raise ValueError(f"Folder {load_dir} does not exist!")
    return agent


def main():
    env_id = get_args().env
    #env = gym.make(env_id, render_mode="human")

    env = make_atari(env_id, render_mode="human")
    env = wrap_deepmind(env, clip_rewards=False, episode_life=True)
    env = Monitor(env)
    # rewards will appear higher than during training since rewards are not clipped

    agent = get_agent(env)

    # check for save path
    agent.load()
    obs, _ = env.reset()
    while True:
        obs_disc = agent.get_discr_obj(obs)
        a = agent.get_action(obs_disc)
        #print(human_readable_action(a))
        obs, reward, done, truncated, info = env.step(a)
        print(f"info: {info}, reward: {reward}")
        if done or truncated:
            env.reset()


def human_readable_action(action):
    match action:
        case 0:
            return "push left!"
        case 1:
            return "don't push!"
        case 2:
            return "push right"

if __name__ == '__main__':
    main()
