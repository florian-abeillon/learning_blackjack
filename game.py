import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gym
import blackjack as bj  #no joke



env = bj.BlackjackEnv()
Verbose = False
algo = ['random', 'semi-random', 'strategic']



#random action: winning score is 28% on average
def random_action():
    return env.action_space.sample()


#semi-random action: winning score is 37% on average
def semi_random_action(player_score, count_cards, nb_cards_out):
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


#strategic action: winning score is 42% on average for sup=14
def strategic_action(player_hand, sup=18):
    if player_hand < sup:
        return 1
    return 0






#sup is the maximum hand score in the strategic action
def main(algo='random', sup=18, Verbose=False):
    avg_win = 0
    avg_tie = 0
    nb_games = 10000


    #for loop to run nb_games blackjack games
    for i_game in range(nb_games):
        count_cards = [0 for _ in range(10)]         #we make no difference between 10/J/Q/K

        player_hand, dealer_first_card, usable_ace = env.reset()
        for card in player_hand:
            index = bj.deck.index(card)
            count_cards[index] -= 1
        index = bj.deck.index(dealer_first_card)
        count_cards[index] -=1

        reward = 0.

        if Verbose:
            print("Pass 0 - Player's score:", bj.sum_hand(player_hand))


        #theoretically ,there cannot be more than 11 passes (4*aces, 4*two, 3*three)
        for t in range(11):
            #if we choose the 'random' strategy
            if algo == 'random':
                action = random_action()  

            #if we choose the 'semi-random' strategy
            if algo =='semi-random':
                #for the first iteration, we need to sum the hand (it will be done in the loop later on)
                if t == 0:
                    player_hand = bj.sum_hand(player_hand)
                action = semi_random_action(player_hand, count_cards, 3 + t)
                
            #if we choose the 'strategic' strategy
            if algo == 'strategic':
                #for the first iteration, we need to sum the hand (it will be done in the loop later on)
                if t == 0:
                    player_hand = bj.sum_hand(player_hand)
                action = strategic_action(player_hand, sup)
                   


            observation, reward, done, info = env.step(action)
            player_hand, dealer_first_card, usable_ace = observation
            player_hand = bj.sum_hand(player_hand)
            if Verbose:
                print("Pass {} - Player's score:".format(i_game+1), player_hand)
                if player_hand > 21:
                    print("Player has been busted.")

            if done:
                if Verbose:
                    dealer_hand = bj.score(env.dealer)
                    print("Dealer's score:", dealer_hand)
                    if dealer_hand > 21:
                        print("Dealer has been busted.")
                
                #if the player won the game
                if reward == 1.:
                    avg_win += 1
                    if Verbose:
                        print("GAME WON")
                        print()
                #if there has been a draw
                elif reward == 0:
                    avg_tie += 1
                    if Verbose:
                        print("TIE")
                        print()
                #if the player lost the game
                else:
                    if Verbose:
                        print("GAME LOST")
                        print()
                break


    if Verbose:
        print("Average winning score with", algo, "action:", 100 * avg_win / nb_games, "%")
        print("Ties:", 100 * avg_tie / nb_games, "%  ||  Losses:", 100 * (1 - (avg_win + avg_tie) / nb_games), "%")
        if algo == 'strategic':
            print("( sup =", sup, ")")

    env.close()

    #percentage of winning games
    return 100 * avg_win / nb_games, 100 * avg_tie / nb_games





#TEST
#Random action
random_algo = main()
print("Average winning score with random action:", random_algo[0], "%")
print("Ties:", random_algo[1], "%  ||  Losses:", 100 - (random_algo[0] + random_algo[1]), "%")
#Semi-random action
semi_random_algo = main('semi-random')
print("Average winning score with semi-random action:", semi_random_algo[0], "%")
print("Ties:", semi_random_algo[1], "%  ||  Losses:", 100 - (semi_random_algo[0] + semi_random_algo[1]), "%")


#TEST
#Strategic action
x = np.arange(22)
y = []
for i in x:
    #We try for every maximum hand score
    y.append(main('strategic', sup=i))
y = np.array(y)
print(y)
print(y[:,0])
best_i = np.argmax(y, axis=0)[0]
print("Best winning score of", y[best_i][0], "% was obtained with maximum hand score of", best_i)
print("Ties:", y[best_i][1], "%  ||  Losses:", 100 - (y[best_i][0] + y[best_i][1]), "%")
#And we plot the results
plt.plot(x,y[:,0])
plt.ylim((0, 100))
plt.xlabel("Maximum hand score before quitting", fontsize=16)
plt.ylabel("Average winning score", fontsize=16)
plt.show()