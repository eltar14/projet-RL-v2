import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from DQN_model import DQN


class DQNAgent:
    def __init__(self, state_shape, action_size, n_frames=4, hidden_dim=128, gamma=0.99, lr=1e-4,
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.999995,
                 memory_size=10000, batch_size=64):

        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.n_frames = n_frames

        self.memory = deque(maxlen=memory_size)
        self.state_stack = deque(maxlen=n_frames)

        stacked_state_shape = (n_frames, *state_shape)


        self.model = DQN(state_shape, action_size).to(self.device) # ========= MODEL DEFINITION


        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        #self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.SmoothL1Loss() #TODO fonction de loss

    def init_stack(self, state):
        """Initialise la stack avec le même état n_frames fois"""
        for _ in range(self.n_frames):
            self.state_stack.append(state)

    def update_stack(self, new_state):
        """Ajoute une nouvelle frame à la stack"""
        self.state_stack.append(new_state)

    def get_stacked_state(self):
        """Retourne la stack comme un np.array (n_frames, H, W)"""
        return np.stack(self.state_stack, axis=0)

    def get_action(self, stacked_state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        state_tensor = torch.FloatTensor(stacked_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return None, None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.model(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q_values

        #q_values = torch.clamp(q_values, -1e3, 1e3)
        #target_q = torch.clamp(target_q, -1e3, 1e3)
        assert q_values.abs().max().item() < 1e4, "Q-values are exploding!"

        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        avg_q = q_values.mean().item()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        #print(len(self.memory))
        return loss.item(), avg_q
