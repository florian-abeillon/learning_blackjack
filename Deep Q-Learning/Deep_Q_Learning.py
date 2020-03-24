import random

import Blackjack_DQL as bj
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pandas import DataFrame

''' Learning parameters'''
EPS = 0.1  # epsilon-greedy
GAMMA = .99

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
NB_EVAL_GAMES = 1000


class QNeuralNetwork(nn.Module):

    def __init__(self):
        super(QNeuralNetwork, self).__init__()

        '''
        self.l1 = nn.Linear(in_features=state_size, out_features=250)
        self.l2 = nn.Linear(in_features=250, out_features=250)
        self.l3 = nn.Linear(in_features=250, out_features=action_size)
        '''

        self.l1 = nn.Linear(in_features=STATE_SIZE, out_features=39)
        self.b1 = nn.BatchNorm1d(num_features=39)

        self.l2 = nn.Linear(in_features=39, out_features=9)
        self.b2 = nn.BatchNorm1d(num_features=9)

        self.l3 = nn.Linear(in_features=9, out_features=ACTION_SIZE)

    def forward(self, state):
        x = torch.tensor(state).clone().detach().float()
        if x.ndimension() == 1:
            x = torch.tensor([state]).clone().detach().float()

        x = self.l1(x)
        x = self.b1(x)
        # x = F.dropout(x)
        x = F.relu(x)

        x = self.l2(x)
        x = self.b2(x)
        # x = F.dropout(x)
        x = F.relu(x)

        x = self.l3(x)

        return x


DQN = QNeuralNetwork()
OPT = optim.Adam(DQN.parameters())
LOSS_FN = nn.MSELoss()


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
        state = np.array([self.exp['state'][i] for i in idx])
        action = torch.tensor([self.exp['action'][i] for i in idx])
        reward = torch.tensor([self.exp['reward'][i] for i in idx])
        next_state = np.array([self.exp['next_state'][i] for i in idx])
        done = np.array([self.exp['done'][i] for i in idx])
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


def choose_action(state):
    """epsilon-greedy for trade-off exploration/exploitation"""
    if random.uniform(0, 1) < EPS:
        action = random.randrange(0, ACTION_SIZE)

    else:
        DQN.eval()
        output = DQN(state)
        action = torch.argmax(output).item()
    return action


def optimize_model(replay_buffer):
    DQN.train()
    if replay_buffer.get_num() < BATCH_SIZE:
        return  # Not enough experiences yet

    states, actions, rewards, next_states, dones = replay_buffer.get_batch()
    states = states.reshape(BATCH_SIZE, STATE_SIZE)
    next_states = next_states.reshape(BATCH_SIZE, STATE_SIZE)

    OPT.zero_grad()

    outputs = DQN(states)
    targets = outputs.clone().detach()
    next_states_values = DQN(next_states)

    for i in range(BATCH_SIZE):
        targets[i, actions[i]] = rewards[i] + (1 - dones[i]) * GAMMA * torch.max(next_states_values[i])

    loss = LOSS_FN(outputs, targets)
    loss.backward()
    OPT.step()


def training():
    env = bj.BlackjackEnv(count_cards=COUNT_CARDS)

    replay_buffer = ExperienceReplay()

    for i in range(NB_TRAIN_GAMES):

        init_state = env.new_game()

        round_not_over = True

        while round_not_over:

            action = choose_action(init_state)

            next_state, reward, done, _ = env.step(action)
            next_state = next_state
            if done:
                round_not_over = False

            replay_buffer.add(init_state, action, reward, next_state, done)

            init_state = next_state

            optimize_model(replay_buffer)

    env.close()


def test():
    env = bj.BlackjackEnv(count_cards=COUNT_CARDS)

    DQN.eval()

    nb_victories = 0.
    nb_ties = 0.
    nb_losses = 0.

    for game in range(NB_EVAL_GAMES):
        init_state = env.new_game()
        action = torch.argmax(DQN(init_state)).item()
        next_state, reward, done, _ = env.step(action)
        next_state = next_state
        init_state = next_state

        while not done:
            action = torch.argmax(DQN(init_state)).item()
            next_state, reward, done, _ = env.step(action)
            next_state = next_state
            init_state = next_state

        if reward >= 1:
            nb_victories += 1
        elif reward == 0:
            nb_ties += 1
        elif reward == -1:
            nb_losses += 1

    env.close()
    print("Wins: {:.2%} || Ties: {:.2%} || Losses: {:.2%} ".format(nb_victories / NB_EVAL_GAMES,
                                                                   nb_ties / NB_EVAL_GAMES,
                                                                   nb_losses / NB_EVAL_GAMES))
    print("Expected value:", (nb_victories - nb_losses) / NB_EVAL_GAMES)


def print_q_table():
    DQN.eval()

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
            output = DQN(state)
            action = torch.argmax(output).item()
            if action == 1:
                pair.append("HIT")
            else:
                pair.append("STAND")

            ''' Hard ace '''
            state = torch.tensor([player_hand] + [dealer_hand] + [1]).float()
            output = DQN(state)
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
    test()
    print(" ")

print_q_table()
