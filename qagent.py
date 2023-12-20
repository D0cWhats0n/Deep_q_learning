import gymnasium as gym
from collections import defaultdict
import numpy as np
import pickle
from pathlib import Path


## Class was inital test and only works for a single game

class Qagent():
    def __init__(self, env: gym.Env, SEED, epsilon=0.5, learning_rate=0.1, discount=0.99, save_dir=None):
        self.seed = SEED
        self.env = env
        self.bin_num = 20
        self.q_values = self.init_q_table()
        self.epsilon = epsilon
        self.lr = learning_rate
        self.discount = discount
        self.save_dir = save_dir
        if self.save_dir:
            self.save_path = Path(save_dir) / "q_vals"

    def learn(self, num_episodes):
        if self.save_path is not None:
            self.load()
        obs, _ = self.env.reset()

        for i in range(num_episodes):
            next_obs, reward, terminated, _, info=self.env.step(0)
            done = False
            obs, _ = self.env.reset()

            max_q_val = -1e3
            max_x_val = -1.5
            while not done:
                obs_disc = self.get_discr_obj(obs)
                action = self.get_action(obs_disc)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                if next_obs[0] > max_x_val:
                    max_x_val = next_obs[0]
                next_obs_disc = self.get_discr_obj(next_obs)

                self.q_values[obs_disc][action] = self.q_values[obs_disc][action] + self.lr * (
                    reward + np.max(self.q_values[next_obs_disc]) - self.q_values[obs_disc][action])
                
                done = (terminated or truncated)
                obs = next_obs

                # overpay successfull termination of game
                if terminated:
                    print("Goal achieved!")
                    self.epsilon_decay()
                    self.q_values[obs_disc][action] = 10.0

                if self.q_values[obs_disc][action] > max_q_val:
                    max_q_val = self.q_values[obs_disc][action]

            if (i%100==0):
                print(f"Number of episodes {i} with max q val {max_q_val} and x_max {max_x_val}")
        self.save()


    def get_action(self, obs: tuple) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < (self.epsilon):
            return self.env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            pred = self.q_values[obs]
            print(pred)
            return int(np.argmax(pred))
        
    def get_discr_obj(self, obj):
        x = obj[0]
        v = obj[1]
        x_max = 0.6
        x_min = -1.2
        v_max = 0.07
        v_min = -v_max
        
        x_bin_size = (x_max - x_min) / self.bin_num
        v_bin_size = (v_max - v_min) / self.bin_num

        x_bin = int((x - x_min) / x_bin_size)
        v_bin = int((v - v_min) / v_bin_size)
        return x_bin, v_bin

    def init_q_table(self):
        return np.zeros(shape=(self.bin_num, self.bin_num, self.env.action_space.n))

    def epsilon_decay(self):
        self.epsilon=self.epsilon * 0.9

    def save(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, "wb") as file:
            pickle.dump(self.q_values, file)
            print(f"Saved agent's q-values to {self.save_dir}")

    def load(self):
        if self.save_path.exists():
            with open(self.save_path, "rb") as file:
                self.q_values = pickle.load(file)
                print("Loaded q-values from file")
        else:
            print("Could not load model from file")

