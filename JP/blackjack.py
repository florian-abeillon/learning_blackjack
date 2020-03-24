import gym
from gym import spaces
from gym.utils import seeding

import numpy as np


def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]


def deck_init(nb_deck):
    # 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
    one_color_deck = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])
    deck = np.tile(one_color_deck, 4)  # full start deck
    deck = np.tile(deck, nb_deck)
    np.random.shuffle(deck)  # the dealer shuffles the cards
    return deck


def reward(player_hand, dealer_hand, natural):  # returns the reward value
    player_score = score(player_hand)  # score() returns 0 if over 21
    dealer_score = score(dealer_hand)

    if player_score > dealer_score:  # won
        if natural and is_natural(player_hand):
            return 1.5
        else:
            return 1

    elif player_score == dealer_score:  # tie
        return 0

    elif player_score < dealer_score:  # loss
        return -1


class BlackjackEnv(gym.Env):
    """Simple blackjack environment

    Blackjack is a card game where the goal is to obtain cards that sum to as
    near as possible to 21 without going over.  They're playing against a fixed
    dealer.
    Face cards (Jack, Queen, King) have point value 10.
    Aces can either count as 11 or 1, and it's called 'usable' at 11.
    This game is placed with an infinite deck (or with replacement).
    The game starts with each (player and dealer) having one face up and one
    face down card.

    The player can request additional cards (hit=1) until they decide to stop
    (stick=0) or exceed 21 (bust).

    After the player sticks, the dealer reveals their facedown card, and draws
    until their sum is 17 or greater.  If the dealer goes bust the player wins.

    If neither player nor dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.  The reward for winning is +1,
    drawing is 0, and losing is -1.

    The observation of a 3-tuple of: the players current sum,
    the dealer's one showing card (1-10 where 1 is ace),
    and whether or not the player holds a usable ace (0 or 1).

    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto.
    http://incompleteideas.net/book/the-book-2nd.html
    """

    def __init__(self, natural=False, nb_deck=1):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(11),
            spaces.Discrete(2)))
        self.seed()
        self.deck_value = 0

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural

        self.nb_deck = nb_deck
        self.deck = deck_init(self.nb_deck)

    def new_game(self):

        self.dealer = self.draw_hand()
        self.player = self.draw_hand()

        return self.get_obs()

    def step(self, action):
        assert self.action_space.contains(action)
        if action:  # hit: add a card to players hand and return
            self.player.append(self.draw_card())
            if is_bust(self.player):
                done = True
                rwd = -1
            else:
                done = False
                rwd = 0
        else:  # stick: play out the dealers hand, and score
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(self.draw_card())

            rwd = reward(self.player, self.dealer, self.natural)

        return self.get_obs(), rwd, done, {}

    def get_obs(self):
        return self.player, self.dealer[0], usable_ace(self.player)

    def draw_card(self):  # randomly draw a card

        # Check if deck is empty => new shuffled deck
        if len(self.deck) == 0:
            self.deck = deck_init(self.nb_deck)
            self.deck_value = 0

        card = self.deck[0]  # get the first card
        self.deck = self.deck[1:]
        
        if (card in [1,10]) :
            self.deck_value -= 1
        elif (card in [7,8,9]) :
            self.deck_value += 1
        
        return int(card)

    def draw_hand(self):  # randomly draw a hand (two cards)
        return [self.draw_card(), self.draw_card()]

    def get_deck_value(self) :
        return self.deck_value / ( (len(self.deck) // 52) + 1 )


'''
    def step(self, action):

        assert self.action_space.contains(action)

        if action:  # hit: add a card to players hand and return
            self.player.append(self.draw_card())
            if is_bust(self.player):
                done = True
                rew = -1
            else:
                done = False
                rew = 0

        else:  # stick: play out the dealers hand, and score
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(self.draw_card())

            rew = reward(self.player, self.dealer, self.natural)

        return self.get_obs(), rew, done, {}
'''
