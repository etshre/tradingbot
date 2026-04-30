import torch
import torch.nn as nn
import torch.nn.functional as F
import ptan

class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DuelingDQN, self).__init__()

        # Shared feature extractor (e.g., CNN or Dense)
        self.conv = nn.Sequential(
            nn.Linear(input_shape[0], 128),
            nn.ReLU()
        )

        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

        # Value stream
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        advantage = self.advantage(x)
        value = self.value(x)
        # Dueling formula: Q = V + A - mean(A)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

# Assuming env is initialized (e.g., Gym CartPole)
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

# 1. Create main net and target net
net = DuelingDQN(obs_size, n_actions)
target_net = ptan.agent.TargetNet(net)

# 2. Selector for epsilon-greedy policy
selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=1.0, selector=ptan.actions.ArgmaxActionSelector())

# 3. PTAN DQNAgent
agent = ptan.agent.DQNAgent(net, selector, preprocessor=ptan.agent.float32_preprocessor)

import torch.optim as optim

# 1. Experience Source
exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=0.99, steps_count=1)

# 2. Experience Buffer
buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=100000)

optimizer = optim.Adam(net.parameters(), lr=1e-3)

# 3. Training Step (inside your main loop)
def training_step():
    buffer.populate(1) # Populate buffer with 1 new step
    if len(buffer) < 1000: # Min experience before training
        return
    
    batch = buffer.sample(32)
    # Convert batch to tensors...
    
    # --- DDQN Update ---
    # 1. Get action from main net, but value from target net
    # 2. Loss = MSE(Q_main, Reward + Gamma * Q_target(next_state, BestAction_Main))
    
    # --- PTAN TargetNet Sync ---
    target_net.sync()
