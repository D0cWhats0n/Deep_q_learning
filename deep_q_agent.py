import gymnasium as gym
import numpy as np
import collections
import random
from typing import NamedTuple
from conv_net import ConvNet
import torch
from pathlib import Path
import copy
import pickle
import matplotlib.pyplot as plt

class ReplayBuffer:
    def __init__(self, max_length):
        self.buffer = collections.deque(maxlen=max_length)

    def add_item(self, item):
        self.buffer.append(item)

    def draw_random_sample(self, sample_size):
        if len(self.buffer) < sample_size:
            raise ValueError("Sample size exceeds buffer length")
        
        # Making sure to only draw from buffer samples with enough neighbours to fill channels
        idxs = random.sample(range(2, len(self.buffer)-2), k=sample_size)
        batch = []
        for i in idxs:
            item = self.buffer[i]
            state = np.stack(
                (self.buffer[i-2].state, self.buffer[i-1].state, self.buffer[i].state)
            ).squeeze()
            next_state = np.stack(
                (self.buffer[i-1].state, self.buffer[i].state, self.buffer[i+1].state)
            ).squeeze()
            batch.append(BufferItem(state, item.action, item.reward, next_state, item.done))
        return batch  


class BufferItem(NamedTuple):
    state: list
    action: int
    reward: float
    next_state: list
    done: bool

    
class DeepQAgent():
    '''Deep Q-Agent for atari games. As a state is represented by pixels of one (or several) images
    the Q (-loss and -target) network are convolutional neural nets. 

    Implementation details:
        - For stability reasons a Q-target network and Q-loss network are used. The Q-target 
        network is only updated after certain amount of steps given by "q_target_update_steps"
        - The Q-loss network is updated each step using the temporal difference rule (i.e. step size 1)

    Agent only works for environments that have a finite set of actions and the number of actions
    must be the same for each state.
    '''
    def __init__(self, env: gym.Env, epsilon=0.1, learning_rate=0.00001, discount=0.99, buff_size=int(5e5), batch_size=32,
                 save_path="deep_q_agent"):
        self.env = env
        self.epsilon = epsilon
        self.gamma = discount
        self.lr = learning_rate
        
        self.replay_buff = ReplayBuffer(buff_size)

        obs, _ = self.env.reset()
        x = self.preprocess([obs])
        self.input_shape = (3, x.shape[1], x.shape[2]) #pytorch has channels first as default
        
        self.q_net = ConvNet(self.input_shape, env.action_space.n)
        
        self.target_net = copy.deepcopy(self.q_net)
        self.batch_size = batch_size
        self.loss = torch.nn.HuberLoss()
        #self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.save_path = Path(save_path)

        #Debug vals
        self.max_q = 0.0
        self.min_q = 100
        self.target_q_vals = collections.deque(maxlen=10000)


    def learn(self, num_episodes):
        action_dict = {key: 0 for key in range(6)}
        total_steps = 0
        rewards = []
        for i in range(num_episodes):
            obs, _ = self.env.reset()
            state = np.stack([obs] * 3).squeeze()

            done = False
            j = 0
            total_loss = 0
            total_reward = 0
            mean_reward = -21.0
            self.max_q = 0.0
            while not done:
                action = self.get_action(state)
                action_dict[action] += 1
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward         
                
                done = (terminated or truncated)  
                self.replay_buff.add_item(BufferItem(obs, action, reward, next_obs, done))
                
                loss = self.learn_from_buffer()

                if loss is not None:
                    total_loss += loss
                else:
                    total_loss = float('inf')
                j = j + 1
                total_steps += 1

                if total_steps%10000 == 0:
                    #Copy q_net to target net
                    print("Copying to target network")
                    self.target_net = copy.deepcopy(self.q_net)
                    self.target_net.eval()

                state = self.update_state(state, next_obs)
                obs = next_obs
            #self.update_epsilon(i, num_episodes)
            rewards.append(total_reward)

            if len(rewards)>10 and np.mean(rewards[-10:])>mean_reward:
                mean_reward = np.mean(rewards[-10:])
                self.save_model()

            print(f"Episode {i} \nTotal loss of episode {total_loss/j}")
            print(f"Length of episode {j}")
            print(f"Total reward of episode {total_reward}")
            print(f"Max q value seen {self.max_q}")
            print(f"Min q value seen {self.min_q}")
            print(f"Mean q values {np.mean(self.target_q_vals)}")
            print(f"Action dict {action_dict}")
            print(f"epsilon {self.epsilon}")
            print(f"mean_reward {np.mean(rewards[-10:])}")
    
        self.save_model()

    def update_epsilon(self, step, episodes, fraction=1/10.0, thresh=0.1):
        '''Implements scheduler for adapting epsilon over time. "fraction" gives the fraction
        of total time steps during which epsilon is linearly changed from 1.0 to 0.0
        '''
        ratio = 1 - 0.9 * step / (episodes*fraction)
        if ratio>=thresh:
            self.epsilon = ratio  

    def update_state(self, state, next_obs):
        state = np.roll(state, shift=-1, axis=0)
        state[-1, :, :] = next_obs.squeeze()
        return state

    def learn_from_buffer(self):
        try:
            samples = self.replay_buff.draw_random_sample(self.batch_size)
        except ValueError:
            print("Skip learning because buffer is too small.")
            samples = None

        if samples is not None:
            y = torch.tensor([self.get_target(el) for el in samples])
            x = self.preprocess([el.state for el in samples])
        
            actions = torch.tensor([int(el.action) for el in samples])

            y, actions = self.send_to_cuda([y, actions])

            self.optimizer.zero_grad()
            y_pred = self.q_net(x)
            y_pred = y_pred[torch.arange(y_pred.shape[0]), actions]

            loss = self.loss(y_pred, y)
            loss.backward()

            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(self.q_net.parameters(), 100)
            
            self.optimizer.step()

            return loss.item()

    def send_to_cuda(self, arr):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        moved_items = []
        for el in arr:
            item = el.to(device)
            moved_items.append(item)
        return moved_items

    def get_target(self, item: BufferItem):
        #TODO allow for batch processing
        if item.done:
            return float(item.reward)
        else:
            with torch.no_grad():
                x = self.preprocess([item.next_state])
                pred = self.target_net(x)[0]
                q_targ = torch.max(pred)
                for el in pred.cpu().detach().numpy():
                    self.target_q_vals.append(el)
                if q_targ>self.max_q:
                    self.max_q = q_targ
                if q_targ<self.min_q:
                    self.min_q = q_targ
                return (item.reward + self.gamma * q_targ)

    def preprocess(self, x):
        '''preprocesses a batch of images that are in channel_last format
        and pixel values are in the range 0-255, so that the output is a pytorch tensor
        in channel_first format and scaled between 0 and 1
        '''
        x = np.multiply(x, 1.0/255.0)
        x = x[:,:,13:77,:]  #Crop image to relevant information
        x = torch.from_numpy(x)
        x = x.type(torch.FloatTensor)
        x = self.send_to_cuda([x])[0]
        return x

    def get_action(self, state: np.ndarray):
        '''Returns an action for a given state using and epsilon-greedy approach
        '''
        if np.random.random()<self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                x = self.preprocess([state])
                pred = self.q_net(x)[0]
                return torch.argmax(pred).item()

    def save_model(self):
        self.save_path.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.q_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
            }, self.save_path / 'model.pt')
        with open(self.save_path / 'replay_buff.pkl', 'wb') as file:
            pickle.dump(self.replay_buff, file, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved q-net model to {self.save_path / 'q_net.pt'}")


    def load_model(self, path):
        path=Path(path)
        if (path / 'model.pt').exists():
            self.q_net = ConvNet(self.input_shape, self.env.action_space.n)
            self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)

            checkpoint = torch.load(path / 'model.pt')
            self.q_net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.loss = checkpoint["loss"]
            self.q_net.train()

            self.target_net = copy.deepcopy(self.q_net)
            self.target_net.eval()
            print(f"Loaded q-net model from {path / 'pt_model'}")
        if (path / 'replay_buff.pkl').exists():
            with open(path / 'replay_buff.pkl', 'rb') as file:
                self.replay_buff = pickle.load(file)
                print(f"Loaded buffer with size {len(self.replay_buff.buffer)}")
