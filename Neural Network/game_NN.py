import numpy as np
import matplotlib.pyplot as plt
import blackjack as bj

import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop

import sklearn.metrics as metrics

nb_deck = 1

# random action: winning score is 28% on average
def random_strategy():
    return np.random.randint(2)


def action(algo, player_hand, sup, dealer_first_card, usable_ace, t):
    # if we choose the 'random' strategy
    if algo == 'random':
        return random_strategy()
    else :
        if (bj.sum_hand(player_hand)) <= 11 :
            return 1
        else :
            return 0

# sup is the maximum hand score in the deterministic action
def main(algo, nb_games, txt_to_read, n_test, Verbose=False, sup=18, new_winning_dataset = False, txt = 'test.csv'): #new_winning_dataset = True if the dataset has not been generated yet
                                                                                                                          #txt to name the output csv, txt_to_read to get the name of the csv to get
    win = 0
    tie = 0
    loss = 0
    nb_games_played = 0

    # initialization of the game
    df = pd.DataFrame(columns = ['player_hand_value','dealer_first_card', 'usable_ace', 'deck_value' , 'action',1,2,3,4,5,6,7,8,9,10])

    # while loop to run enough games to store nb_games
    if new_winning_dataset :
        while (nb_games_played < nb_games) :
            player_hand, dealer_first_card, usable_ace = env.new_game()

            # in the official rules, there can only be a maximum of 2 passes, but we will not implement this rule
            # (not relevant)
            # theoretically ,there cannot be more than 11 passes (4*aces, 4*two, 3*three)
            for t in range(11):
                act = action(algo, player_hand, sup, dealer_first_card, usable_ace, t)
                # If the player has a blackjack, he automatically stands
                if sum(player_hand) == 21 :
                    act = 0

                # features enabling to count the cards
                deck_value = env.get_deck_value()
                deck_counting_cards = env.get_deck_cards_counting()

                if (t == 0) :
                    state = [bj.sum_hand(player_hand), dealer_first_card, bj.has_ace(player_hand)*1, deck_value,act] + list(deck_counting_cards)
                    opposite_state = [bj.sum_hand(player_hand), dealer_first_card, bj.has_ace(player_hand)*1, deck_value, 1 - act] + list(deck_counting_cards)

                observation, reward, done, info = env.step(act)
                player_hand, dealer_first_card, usable_ace = observation

                if done:
                    # if the player won the game, we store the game
                    if reward >= 0:
                        df.loc[nb_games_won] = state
                    else :
                        df.loc[nb_games_won] = opposite_state
                    nb_games_played += 1
                    if (nb_games_played % (nb_games//10) == 0) :
                        print('|', end='')
                    break

        env.close()
        df.to_csv(txt,index = False)

    else :
        df = pd.read_csv(txt_to_read)

    # Train the model with the dataset
    model = Sequential()
    model.add(Dense(16, kernel_initializer='lecun_uniform'))
    model.add(Activation('relu'))

    model.add(Dense(256, kernel_initializer='lecun_uniform'))
    model.add(Activation('relu'))

    model.add(Dense(128, kernel_initializer='lecun_uniform'))
    model.add(Activation('relu'))

    model.add(Dense(32, kernel_initializer='lecun_uniform'))
    model.add(Activation('relu'))

    model.add(Dense(8, kernel_initializer='lecun_uniform'))
    model.add(Activation('relu'))
        
    model.add(Dense(1, kernel_initializer='lecun_uniform'))
    model.add(Activation('sigmoid'))

    model.compile(loss='mse', optimizer='adam')

    X_train = np.array(df[['dealer_first_card','player_hand_value', 'usable_ace','deck_value']])
    y_train = np.array(df['action']).reshape(-1,1)
    model.fit( X_train, y_train, epochs=20)

    #ROC CURVE : play 10000 games and test predict them
    actuals = []
    pred_Y_train = []

    for i in range(10000) :
            player_hand, dealer_first_card, usable_ace = env.new_game()
            for t in range(11):
                act = action(algo, player_hand, sup, dealer_first_card, usable_ace, t)
                if sum(player_hand) == 21 :
                    act = 0
                deck_value = env.get_deck_value()
                deck_counting_cards = env.get_deck_cards_counting()

                if (t == 0) :
                    state = [bj.sum_hand(player_hand), dealer_first_card, bj.has_ace(player_hand)*1, deck_value, act] 
                    opposite_state = [bj.sum_hand(player_hand), dealer_first_card, bj.has_ace(player_hand)*1, deck_value, 1 - act] 

                    cur_state = [[bj.sum_hand(player_hand), dealer_first_card, bj.has_ace(player_hand)*1, deck_value]]
                    p = model.predict(np.array(cur_state)[0:1], batch_size=1)            
                    pred_Y_train.append(p[0][0])

                observation, reward, done, info = env.step(act)
                player_hand, dealer_first_card, usable_ace = observation

                if done:
                    if reward >= 0:
                        actuals.append(state[4])
                    else :
                        actuals.append(opposite_state[4])
                    break

    env.close()
    
    fpr, tpr, threshold = metrics.roc_curve(actuals, pred_Y_train)
    roc_auc = metrics.auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(10,8))
    plt.plot(fpr, tpr, label = ('ROC AUC = %0.3f' % roc_auc))

    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    ax.set_xlabel("False Positive Rate",fontsize=12)
    ax.set_ylabel("True Positive Rate",fontsize=12)
    plt.setp(ax.get_legend().get_texts(), fontsize=12)
    plt.show()



    # Use the model to predict
    for i_game in range(n_test):

        player_hand, dealer_first_card, usable_ace = env.new_game()

        # in the official rules, there can only be a maximum of 2 passes, but we will not implement this rule
        # (not relevant)
        # theoretically ,there cannot be more than 11 passes (4*aces, 4*two, 3*three)
        for t in range(11):
            # Counting the cards
            deck_value = env.get_deck_value()
            deck_counting_cards = env.get_deck_cards_counting()

            # Choosing the next action
            state = [[bj.sum_hand(player_hand),dealer_first_card, bj.has_ace(player_hand)*1,deck_value]] 
            act = 0
            p = model.predict(np.array(state)[0:1], batch_size=1)
            if ( (p > 0.5)) :
                act = 1
            
            # Playing
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
results = main(algo='random', nb_games = 50000, txt = 'replacement.csv', txt_to_read = 'replacement.csv', n_test = 10000)
print("Wins:", results[0], "% || Ties:", results[1], "%  ||  Losses:", results[2], "%")
print("Espérance : " + str(results[0] - results[2]))
print("WITHOUT REPLACEMENT")
env = bj.BlackjackEnv(nb_deck)
results = main(algo='random', nb_games = 50000, txt = 'no_replacement.csv', txt_to_read = 'no_replacement.csv', n_test = 10000)
print("Wins:", results[0], "% || Ties:", results[1], "%  ||  Losses:", results[2], "%")
print("Espérance : " + str(results[0] - results[2]))
