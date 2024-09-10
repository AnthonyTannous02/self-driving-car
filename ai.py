#   IMPORT LIBRARIES
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


# Create the architecture of the Neural Network
class Network(nn.Module):

    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(
            input_size, 30
        )  # full connx (all connx) between input layer and the hidden layer
        self.fc2 = nn.Linear(30, nb_action)

    def forward(self, state):  # activate forward propagation (nn)
        x = F.relu(self.fc1.forward(state))  # hidden neurons
        q_values = self.fc2.forward(x)
        return q_values


# IMPLEMENTING EXPERIENCE REPLAY (TAKING MANY STATES IN THE PAST INTO CONSIDERATION FOR LEARNING (lONG TERM MEMORY))
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        # if list = ((1, 2, 3), (4, 5, 6)), then zip(*list) = ((1, 4), (2, 5), (3, 6))
        # #((state, action, reward)) => ((states batch), (actions batch), (rewards batch))
        # To differenciate between the Tensors and Gradiants
        samples = zip(*random.sample(self.memory, batch_size))
        return map(
            lambda x: Variable(torch.cat(x, 0)), samples
        )  # 0 means this fake dimension should be the first dimension

    def get_size(self):
        return len(self.memory)


# Implementing the Deep Q Learning
class Dqn:

    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100_000)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=0.001
        )  # lr is learning rate, we cant have it too small for the model to have time to update
        self.last_state = torch.Tensor(input_size).unsqueeze(
            0
        )  # 0 means this fake dimension should be the first dimension
        self.last_action = 0  # this can be either 0, 1 or 2 (0, 20, -20) and here we initialize it to 0
        self.last_reward = 0  # same

    def select_action(self, state):
        T = 100  # temperature, the higher, the more certain we are of  the action we are going to take
        probs = F.softmax(
            self.model.forward(Variable(state, volatile=True)) * T
        )  # volatile = True will remove the gradient, therefore improving performance
        # softmax([1, 2, 3] * 1) => [0.09, 0.11, 0.8], softmax([1, 2, 3] * 3) => [0, 0.02, 0.98]
        action = probs.multinomial(num_samples=1)
        return action.data[0, 0]

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = (
            self.model.forward(batch_state)
            .gather(1, batch_action.unsqueeze(1))
            .squeeze(1)
        )
        next_outputs = self.model.forward(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()  # makes the optimizer rinitialize at each 'loop'
        td_loss.backward(
            retain_graph=True
        )  # should be retain_variables i think (to free up memory)
        self.optimizer.step()  # updates the weights

    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push(
            (
                self.last_state,
                new_state,
                torch.LongTensor([int(self.last_action)]),
                torch.Tensor([self.last_reward]),
            )
        )
        action = self.select_action(new_state)
        if self.memory.get_size() > 100:
            batch_state, batch_next_state, batch_action, batch_reward = (
                self.memory.sample(100)
            )
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]

        return action

    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1)

    def save(self):
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            "last_brain.pth",
        )

    def load(self):
        if os.path.isfile("last_brain.pth"):
            print("=> Loading CheckPoint...")
            checkpoint = torch.load("last_brain.pth")
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            print("Done !")
        else:
            print("No Checkpoint Found !")
