import numpy as np
import matplotlib.pyplot as plt
import Blackjack as bj

nb_deck = 1
gamma = 0.8
epsilon = 0.01


# random action: winning score is 28% on average
def random_strategy():
    return np.random.randint(2)


# semi-random action: winning score is 37% on average
def semi_random_strategy(player_hand, count_cards, nb_cards_out):
    player_score = bj.sum_hand(player_hand)
    card_sup = 21 - player_score
    if card_sup >= 10:
        return 1

    proba = 4 * card_sup
    for i in range(card_sup):
        proba -= count_cards[i]
    proba /= (52 - nb_cards_out)

    if proba < 0.5:
        return 0
    return 1


# strategic action: winning score is 42% on average for sup=14
def strategic_action(player_hand, sup=18):
    player_total = bj.sum_hand(player_hand)
    if player_total < sup:
        return 1
    return 0


# computes the probability of having "end" if we draw a card on top of "start"
def proba(start, end, player=np.array([]), dealer=-1):
    card = end - start
    if not 1 <= card <= 11:
        return 0

    # 32 cards in the beginning, minus the visible cards
    # (those from the player's hand plus the one from the dealer's hand)
    size_deck = 32 - player.size - (dealer != -1) * 1

    # making the probability up upon our current knowledge
    count = 4 - list(player).count(card) - (dealer == card) * 1
    return count / size_deck


def basic_strategy(player_hand, dealer_value, soft):
    """ This is a simple implementation of Blackjack's
        basic strategy. It is used to recommend actions
        for the player. """

    player_total = bj.sum_hand(player_hand)

    if 4 <= player_total <= 11:
        return 1

    elif soft:
        # we only double soft 12 because there's no splitting
        if 12 <= player_total <= 18:
            return 1
        if player_total == 18:
            if dealer_value in [2, 7, 8]:
                return 0
            else:
                return 1
        if player_total >= 19:
            return 0

    else:
        if player_total == 12:

            if dealer_value in [1, 2, 3, 7, 8, 9, 10]:
                return 1
            else:
                return 0

        if 13 <= player_total <= 16:
            if 2 <= dealer_value <= 6:
                return 0
            else:
                return 1

        if player_total >= 17:
            return 0


def action(algo, player_hand, count_cards, sup, dealer_first_card, usable_ace, t):
    # if we choose the 'random' strategy
    if algo == 'random':
        return random_strategy()

    # if we choose the 'semi-random' strategy
    elif algo == 'semi-random':
        # for the first iteration, we need to sum the hand (it will be done in the loop later on)
        return semi_random_strategy(player_hand, count_cards, 3 + t)

    # if we choose the 'strategic' strategy
    elif algo == 'strategic':
        return strategic_action(player_hand, sup)

    elif algo == "basic strategy":
        return basic_strategy(player_hand, dealer_first_card, usable_ace)


# sup is the maximum hand score in the deterministic action
def main(algo, Verbose=False, sup=18):
    win = 0
    tie = 0
    loss = 0
    nb_games = 999999

    # initialization of the game


    # for loop to run nb_games blackjack games
    for i_game in range(nb_games):

        player_hand, dealer_first_card, usable_ace = env.new_game()

        # Counting cards
        # (we make no difference between 10/J/Q/K)
        count_cards = np.zeros(10)

        if sum(count_cards >= 52 * nb_deck): # Reset when every card has been played
            count_cards = np.zeros(10)

        for card in player_hand:
            count_cards[card - 1] -= 1
        count_cards[dealer_first_card - 1] -= 1

        # in the official rules, there can only be a maximum of 2 passes, but we will not implement this rule
        # (not relevant)
        # theoretically ,there cannot be more than 11 passes (4*aces, 4*two, 3*three)
        for t in range(11):

            act = action(algo, player_hand, count_cards, sup, dealer_first_card, usable_ace, t)

            observation, reward, done, info = env.step(act)
            player_hand, dealer_first_card, usable_ace = observation

            if done:
                # if the player won the game
                if reward >= 1:
                    win += 1

                elif reward == 0:
                    tie += 1

                elif reward == -1:
                    loss += 1

                break

    env.close()

    # percentage of winning games
    return 100 * win / nb_games, 100 * tie / nb_games, 100 * loss/nb_games, (win - loss) / nb_games


# TEST

# basic strategy action
env = bj.BlackjackEnv(nb_deck)
results = main(algo='basic strategy')
print("Results with basic strategy action:")
print("Wins:", results[0], "% || Ties:", results[1], "%  ||  Losses:", results[2], "%")
print("Expected value:", results[3])

# Random action
env = bj.BlackjackEnv(nb_deck)
results = main(algo='random')
print("Results with random action:",)
print("Wins:", results[0], "% || Ties:", results[1], "%  ||  Losses:", results[2], "%")
print("Expected value:", results[3])

# Semi-random action
env = bj.BlackjackEnv(nb_deck)
results = main(algo='semi-random')
print("Results with semi-random action:",)
print("Wins:", results[0], "% || Ties:", results[1], "%  ||  Losses:", results[2], "%")
print("Expected value:", results[3])


# TEST
# Strategic action
env = bj.BlackjackEnv(nb_deck)
x = np.arange(22)
y = []
for i in x:
    # We try for every maximum hand score
    y.append(main('strategic', sup=i))
y = np.array(y)
#print(y)
#print(y[:, 0])
best_i = np.argmax(y, axis=0)[0]
print("Best winning score of", y[best_i][0], "% was obtained with maximum hand score of", best_i)
print("Wins:", y[best_i][0], "% || Ties:", y[best_i][1], "%  ||  Losses:", y[best_i][2], "%")
print("Expected value:", y[best_i][3])

# And we plot the results
plt.plot(x, y[:, 0])
plt.ylim((0, 100))
plt.xlabel("Maximum hand score before quitting", fontsize=16)
plt.ylabel("Average winning score", fontsize=16)
plt.show()
