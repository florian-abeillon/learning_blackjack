import numpy as np
import matplotlib.pyplot as plt
import blackjack as bj

import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop

nb_deck = 1

df = pd.DataFrame(columns = ['dealer_first_card', 'player_hand_value', 'usable_ace', 'deck_value' , 'action'])

# random action: winning score is 28% on average
def random_strategy():
    return np.random.randint(2)


def action(algo, player_hand, count_cards, sup, dealer_first_card, usable_ace, t):
    # if we choose the 'random' strategy
    if algo == 'random':
        return random_strategy()

# sup is the maximum hand score in the deterministic action
def main(algo, nb_games_to_win, txt_to_read, n_test, Verbose=False, sup=18, new_winning_dataset = True, txt = 'how_to_win.csv'): #new_winning_dataset = True if the dataset has not been generated yet
                                                                                                                          #txt to name the output csv, txt_tu_read to get the name of the csv to get
    win = 0
    tie = 0
    loss = 0
    nb_games_won = 0

    # initialization of the game
    df = pd.DataFrame(columns = ['dealer_first_card', 'player_hand_value', 'usable_ace', 'deck_value' , 'action'])

    # while loop to run enough games to store nb_games_won
    if new_winning_dataset :
        while (nb_games_won < nb_games_to_win) :
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
                deck_value = env.get_deck_value()

                if (t == 0) :
                    state = [bj.sum_hand(player_hand), dealer_first_card, usable_ace*1, deck_value, act]

                observation, reward, done, info = env.step(act)
                player_hand, dealer_first_card, usable_ace = observation

                if done:
                    # if the player won the game, we store the game
                    if reward >= 1:
                        df.loc[nb_games_won] = state
                        nb_games_won += 1
                        if (nb_games_won % (nb_games_to_win//10) == 0) :
                            print('|', end='')

                    break

        env.close()
        df.to_csv(txt,index = False)

    else :
        df = pd.read_csv(txt_to_read)

    # Train the model with the dataset
    model = Sequential()
    model.add(Dense(2, kernel_initializer='lecun_uniform'))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Dense(8, kernel_initializer='lecun_uniform'))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
        
    model.add(Dense(1, kernel_initializer='lecun_uniform'))
    model.add(Activation('sigmoid'))

    model.compile(loss='mse', optimizer='adam')

    X_train = np.array(df[['dealer_first_card', 'player_hand_value', 'usable_ace','deck_value']])
    y_train = np.array(df['action']).reshape(-1,1)
    model.fit( X_train, y_train, epochs=4)

    # Use the model to predict
    for i_game in range(n_test):

        player_hand, dealer_first_card, usable_ace = env.new_game()

        # in the official rules, there can only be a maximum of 2 passes, but we will not implement this rule
        # (not relevant)
        # theoretically ,there cannot be more than 11 passes (4*aces, 4*two, 3*three)
        for t in range(11):
            deck_value = env.get_deck_value()
            state = [[bj.sum_hand(player_hand), dealer_first_card, usable_ace*1, deck_value]]

            # choosing the next action and help the algorithm if the value of the player's hand is inferior to 14
            act = 0
            p = model.predict(np.array(state)[0:1], batch_size=1)
            if ( (p > 0.55) | (bj.sum_hand(player_hand)<14)) :
                act = 1
            
            observation, reward, done, info = env.step(act)
            player_hand, dealer_first_card, usable_ace = observation

            if done:
                if reward >= 1:
                    win += 1

                elif reward == 0:
                    tie += 1

                elif reward == -1:
                    loss += 1

                break

    # percentage of winning games
    return 100 * win / n_test, 100 * tie / n_test, 100 * loss/n_test


# TEST

# Results
print("WITH REPLACEMENT")
env = bj.BlackjackEnv(1000000)
results = main(algo='random', nb_games_to_win = 20000, txt = 'how_to_win_replacement.csv', txt_to_read = 'how_to_win_replacement.csv', n_test = 2000)
print("Wins:", results[0], "% || Ties:", results[1], "%  ||  Losses:", results[2], "%")
print("Espérance : " + str(results[0] - results[2]))
print("WITHOUT REPLACEMENT")
env = bj.BlackjackEnv(nb_deck)
results = main(algo='random', nb_games_to_win = 200000, txt = 'how_to_win_no_replacement.csv', txt_to_read = 'how_to_win_no_replacement.csv', n_test = 20000)
print("Wins:", results[0], "% || Ties:", results[1], "%  ||  Losses:", results[2], "%")
print("Espérance : " + str(results[0] - results[2]))