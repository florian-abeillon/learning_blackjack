import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gym
import blackjack as bj  # no joke


env = bj.BlackjackEnv(nb_deck=1)
gamma = 0.8
epsilon = 0.01
Verbose = False


# random action: winning score is 28% on average
def random_action():
    return env.action_space.sample()


# deterministic action: winning score is 42% on average for sup=14
def deterministic_action(player_hand, sup=18):
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


# sup is the maximum hand score in the deterministic action
def main(algo='basic strategy', sup=18, Verbose=False):
    avg_win = 0
    nb_games = 10000

    # for loop to run nb_games blackjack games
    for i_game in range(nb_games):
        player_hand, dealer_first_card, usable_ace = env.reset()
        reward = 0.
        if Verbose:
            print("Pass 0 - Player's score:", bj.sum_hand(player_hand))

        # in the official rules, there can only be a maximum of 2 passes, but we will not implement this rule
        # (not relevant)
        # theoretically ,there cannot be more than 11 passes (4*aces, 4*two, 3*three)
        for t in range(11):
            # if we choose the 'random' strategy
            if algo == 'random':
                action = random_action()

                # if we choose the 'deterministic' strategy
            if algo == 'deterministic':
                # for the first iteration, we need to sum the hand (it will be done in the loop later on)
                action = deterministic_action(player_hand, sup)

            if algo == "basic strategy":
                action = basic_strategy(player_hand, dealer_first_card, usable_ace)

            '''
            #if we choose the 'value iteration' strategy
            if algo == 'value_iteration':
                #building the probability matrix
                M = np.zeros(18, 18)
                for i in range(4, 22):
                    for j in range(4, 22):
                        M[i - 4, j - 4] = proba(i, j, player_hand, dealer_first_card)
                v2 = np.zeros(18)

                while delta < epsilon * (1 - gamma) / gamma:
                    v1 = np.copy(v2)
                    delta = 0

                    #loop on every state possible
                    for i in range(4, 22):
                        if i == 21:
                            v2[-1] = reward[-1]
                        else:
                            if_draw = M.dot(v1)[i - 4]
                            if_not_draw = v1[i - 4]
                            v2[i - 4] = reward[i - 4] + gamma * max(if_draw, if_not_draw)
                        
                        diff = abs(v2[s] - v1[s])
                        if diff > delta:
                            delta = diff    
            '''

            observation, reward, done, info = env.step(action)
            player_hand, dealer_first_card, usable_ace = observation
            #player_hand = bj.sum_hand(player_hand)
            if Verbose:
                print("Pass {} - Player's score:".format(i_game + 1), player_hand)
                if player_hand > 21:
                    print("Player has been busted.")

            if done:
                if Verbose:
                    dealer_hand = bj.score(env.dealer)
                    print("Dealer's score:", dealer_hand)
                    if dealer_hand > 21:
                        print("Dealer has been busted.")

                # if the player won the game
                if reward == 1.:
                    avg_win += 1
                    if Verbose:
                        print("GAME WON")
                        print()
                # if the player lost the game
                else:
                    if Verbose:
                        print("GAME LOST")
                        print()
                break

    env.close()

    # percentage of winning games
    return avg_win / nb_games


# TEST
# Random action
print("Average winning score with random action:", main(algo='random'), "%")

#basic strategy action
print("Average winning score with basic strategy action:", main(algo='basic strategy'), "%")

# TEST
# Deterministic action
x = np.arange(22)
y = []
for i in x:
    # We try for every maximum hand score
    y.append(main(algo='deterministic', sup=i))
y = np.array(y)
best_i = np.argmax(y)
print("Best winning score of", y[best_i], "was obtained with maximum hand score of", best_i)
# And we plot the results
plt.plot(x, y)
plt.ylim((0, 1.0))
plt.xlabel("Maximum hand score before quitting", fontsize=16)
plt.ylabel("Average winning score", fontsize=16)
plt.show()
