import numpy as np
import matplotlib.pyplot as plt
import blackjackReplacement as bjr
import blackjack as bj

import dqlearning as dq

nb_deck = 1

# sup is the maximum hand score in the deterministic action
def main(algo, Verbose=False, sup=18):
    win = 0
    tie = 0
    loss = 0
    nb_games = 30000

    # initialization of the game
    player = dq.DeepQLearner()

    # for loop to run nb_games blackjack games
    for i_game in range(nb_games):

        player_hand, dealer_first_card, usable_ace = env.new_game()
        state = [bj.sum_hand(player_hand)] + [dealer_first_card] + [usable_ace]

        # in the official rules, there can only be a maximum of 2 passes, but we will not implement this rule
        # (not relevant)
        # theoretically ,there cannot be more than 11 passes (4*aces, 4*two, 3*three)
        for t in range(11):

            act = player.getAction(state)

            observation, reward, done, info = env.step(act)
            player_hand, dealer_first_card, usable_ace = observation

            state = [bj.sum_hand(player_hand)] + [dealer_first_card] + [usable_ace]

            if done:
                # if the player won the game
                if reward >= 1:
                    player.update(state,1)
                    if (i_game > nb_games-3000) :
                        win += 1

                elif reward == 0:
                    player.update(state,0)
                    if (i_game > nb_games-3000) :
                        tie += 1

                elif reward == -1:
                    player.update(state,-1)
                    if (i_game > nb_games-3000) :
                        loss += 1

                break

    env.close()

    # percentage of winning games
    return 100 * win / 3000, 100 * tie / 3000, 100 * loss/ 3000


# TEST

# basic strategy action
print("WITH REPLACEMENT")
env = bj.BlackjackEnv(1000000)
results = main(algo='basic strategy')
print("Results with DQ action:")
print("Wins:", results[0], "% || Ties:", results[1], "%  ||  Losses:", results[2], "%")
print("Espérance : " + str(results[0] - results[2]))
print("WITHOUT REPLACEMENT")
env = bj.BlackjackEnv(nb_deck)
results = main(algo='basic strategy')
print("Results with DQ action:")
print("Wins:", results[0], "% || Ties:", results[1], "%  ||  Losses:", results[2], "%")
print("Espérance : " + str(results[0] - results[2]))
