import random

import Blackjack_DQL as bj
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from pandas import DataFrame

''' Learning parameters'''
EPS = 0.1  # epsilon-greedy
GAMMA = .99
TAU = 0.01

''' Experience replay parameters'''
BATCH_SIZE = 256
BUFFER_SIZE = 20000

''' Environment parameters'''
COUNT_CARDS = False
if COUNT_CARDS:
    STATE_SIZE = 4
else:
    STATE_SIZE = 3
ACTION_SIZE = 2

NB_EPOCH = 10
NB_TRAIN_GAMES = 10000


class ValueNetwork(nn.Module):

    def __init__(self):
        super(ValueNetwork, self).__init__()

        self.l1 = nn.Linear(in_features=STATE_SIZE, out_features=39)
        self.b1 = nn.BatchNorm1d(num_features=39)

        self.l2 = nn.Linear(in_features=39, out_features=9)
        self.b2 = nn.BatchNorm1d(num_features=9)

        self.l3 = nn.Linear(in_features=9, out_features=1)

    def forward(self, state):
        x = torch.tensor(state).clone().detach().float()
        if x.ndimension() == 1:
            x = torch.tensor([state]).clone().detach().float()

        x = self.l1(x)
        x = self.b1(x)
        x = F.relu(x)

        x = self.l2(x)
        x = self.b2(x)
        x = F.relu(x)

        x = self.l3(x)

        return x


class SoftQNetwork(nn.Module):

    def __init__(self):
        super(SoftQNetwork, self).__init__()

        self.l1 = nn.Linear(in_features=STATE_SIZE + ACTION_SIZE, out_features=50)
        self.b1 = nn.BatchNorm1d(num_features=50)

        self.l2 = nn.Linear(in_features=50, out_features=20)
        self.b2 = nn.BatchNorm1d(num_features=20)

        self.l3 = nn.Linear(in_features=20, out_features=1)

    def forward(self, state, action):
        x = torch.cat([state.long(), action.long()], 1).float()

        x = self.l1(x)
        x = self.b1(x)
        x = F.relu(x)

        x = self.l2(x)
        x = self.b2(x)
        x = F.relu(x)

        x = self.l3(x)

        return x


class PolicyNetwork(nn.Module):

    def __init__(self, log_std_min=20, log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(STATE_SIZE, 50)
        self.linear2 = nn.Linear(50, 20)

        self.mean_linear = nn.Linear(20, ACTION_SIZE)

        self.log_std_linear = nn.Linear(20, ACTION_SIZE)

    def forward(self, state):
        x = F.relu(self.linear1(state.float()))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std * z)
        log_prob = Normal(mean, std).log_prob(mean + std * z) - torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std * z)

        return action[0]


VALUE = ValueNetwork()
TARGET_VALUE = ValueNetwork()

SOFT_Q1 = SoftQNetwork()
SOFT_Q2 = SoftQNetwork()

POLICY = PolicyNetwork()

for target_param, param in zip(TARGET_VALUE.parameters(), VALUE.parameters()):
    target_param.data.copy_(param.data)

VALUE_LOSS = nn.MSELoss()
SQ1_LOSS = nn.MSELoss()
SQ2_LOSS = nn.MSELoss()

VALUE_OPT = optim.Adam(VALUE.parameters())
SQ1_OPT = optim.Adam(SOFT_Q1.parameters())
SQ2_OPT = optim.Adam(SOFT_Q2.parameters())
POLICY_OPT = optim.Adam(POLICY.parameters())


class ExperienceReplay:
    """ From https://medium.com/@qempsil0914/deep-q-learning-part2-double-deep-q-network-double-dqn-b8fc9212bbb2 """

    def __init__(self):
        self.exp = {'state': [], 'action': [], 'reward': [], 'next_state': [],
                    'done': []}  # total experiences the Agent stored

    def get_num(self):
        """return the current number of experiences"""
        return len(self.exp['state'])

    def get_batch(self):
        """random choose a batch of experiences for training"""
        idx = np.random.choice(self.get_num(), size=BATCH_SIZE, replace=False)
        state = torch.tensor([self.exp['state'][i] for i in idx])
        action = torch.tensor([self.exp['action'][i] for i in idx])
        reward = torch.tensor([self.exp['reward'][i] for i in idx])
        next_state = torch.tensor([self.exp['next_state'][i] for i in idx])
        done = torch.tensor([self.exp['done'][i] for i in idx])
        return state, action, reward, next_state, done

    def add(self, state, action, reward, next_state, done):
        """remove the oldest experience if the memory is full"""
        if self.get_num() > BUFFER_SIZE:
            del self.exp['state'][0]
            del self.exp['action'][0]
            del self.exp['reward'][0]
            del self.exp['next_state'][0]
            del self.exp['done'][0]
        """add single experience"""
        self.exp['state'].append(state)
        self.exp['action'].append(action)
        self.exp['reward'].append(reward)
        self.exp['next_state'].append(next_state)
        self.exp['done'].append(done)


def optimize_model(replay_buffer):

    if replay_buffer.get_num() < BATCH_SIZE:
        return  # Not enough experiences yet

    states, actions, rewards, next_states, dones = replay_buffer.get_batch()
    states = states.reshape(BATCH_SIZE, STATE_SIZE)
    next_states = next_states.reshape(BATCH_SIZE, STATE_SIZE)

    predicted_q_value1 = SOFT_Q1(states, actions)
    predicted_q_value2 = SOFT_Q2(states, actions)
    predicted_value = VALUE(states)
    new_actions, log_probs, epsilons, means, log_stds = POLICY.evaluate(states)

    # Training Q Function
    target_value = TARGET_VALUE(next_states)
    target_q_value = rewards + ~dones * GAMMA * target_value
    q_value_loss1 = SQ1_LOSS(predicted_q_value1, target_q_value.detach())
    q_value_loss2 = SQ2_LOSS(predicted_q_value2, target_q_value.detach())
    print("Q Loss:", q_value_loss1)
    SQ1_OPT.zero_grad()
    q_value_loss1.backward()
    SQ1_OPT.step()
    SQ2_OPT.zero_grad()
    q_value_loss2.backward()
    SQ2_OPT.step()

    # Training Value Function
    predicted_new_q_value = torch.min(SOFT_Q1(states, new_actions), SOFT_Q2(states, new_actions))
    target_value_func = predicted_new_q_value - log_probs
    value_loss = VALUE_LOSS(predicted_value, target_value_func.detach())
    print("V Loss:", value_loss)
    VALUE_OPT.zero_grad()
    value_loss.backward()
    VALUE_OPT.step()

    # Training Policy Function
    policy_loss = (log_probs - predicted_new_q_value).mean()
    POLICY_OPT.zero_grad()
    policy_loss.backward()
    POLICY_OPT.step()

    for target_param, param in zip(TARGET_VALUE.parameters(), VALUE.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)


def training():
    env = bj.BlackjackEnv(count_cards=COUNT_CARDS)

    episode_reward = 0
    replay_buffer = ExperienceReplay()

    for i in range(NB_TRAIN_GAMES):

        state = env.new_game()

        round_not_over = True

        while round_not_over:

            action = POLICY.get_action(state).detach().numpy()
            action_env = np.argmax(action)

            next_state, reward, done, _ = env.step(action_env)
            next_state = next_state
            if done:
                round_not_over = False

            replay_buffer.add(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

    optimize_model(replay_buffer)

    env.close()


def print_q_table():


    cards = [i for i in range(2, 11)]
    cards.append(1)

    dealer_first_cards = [str(i) + "" for i in cards]
    dealer_first_cards[-2] += " or any face"
    dealer_first_cards[-1] = "ace"
    player_cards = [str(i) for i in range(4, 21)]

    combinations = []
    for player_hand in range(4, 21):
        lst = []
        for dealer_hand in cards:
            pair = []

            ''' Soft ace '''
            state = torch.tensor([player_hand] + [dealer_hand] + [0]).float()
            output = POLICY.evaluate(state)[0]
            action = torch.argmax(output).item()
            if action == 1:
                pair.append("HIT")
            else:
                pair.append("STAND")

            ''' Hard ace '''
            state = torch.tensor([player_hand] + [dealer_hand] + [1]).float()
            output = POLICY.evaluate(state)[0]
            action = torch.argmax(output).item()
            if action == 1:
                pair.append("HIT")
            else:
                pair.append("STAND")
            lst.append(pair)
        combinations.append(lst)

    combinations = np.array(combinations)
    print("With a soft ace")
    print(DataFrame(np.array(combinations[:, :, 0]), player_cards, dealer_first_cards))  # begins at 13
    print("With a hard ace")
    print(DataFrame(np.array(combinations[:, :, 1]), player_cards, dealer_first_cards))  # begins at 12


''' Training '''
for epoch in range(1, NB_EPOCH + 1):
    print("Epoch", epoch, ":")
    training()
    print(" ")

print_q_table()
