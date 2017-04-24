"""
Author: Daniel Eynis
HW 6: Reinforcement Learning

Instruction to run:
Create an instance of the the class as follows and then run the learn() and test() methods

rf = Reinforcement.ReinforcementLearning(trainfn='train', testfn='test')
rf.learn()
rf.test()

These functions calls will train and test robby. Certain parameters can be changed when the class instance is
being initialized.
"""

import numpy as np
from itertools import product
from random import randint
from enum import Enum, unique
import pandas as pd


@unique
class Action(Enum):  # Enum for the 5 possible moved the robot can make
    MoveN = 0
    MoveS = 1
    MoveE = 2
    MoveW = 3
    PickUpCan = 4


class ReinforcementLearning:
    """
    eps: number of episodes to train and test
    steps: number of steps in each episode
    lrate: learning rate
    dfactor: the discount factor for the equation
    epsilon: starting probability for random action selection
    trainfn: the name of the file where training data will be stored
    testfn: name of the file where testing data will be stored
    constant_epsilon: switch whether to use a constant epsilon in training
    action_tax: switch to set action tax of 0.5 on all actions
    """
    def __init__(self, eps=5000, steps=200, lrate=0.2, dfactor=0.9, epsilon=1, trainfn='traindata', testfn='testdata', constant_epsilon=False, action_tax=False):
        self.action_tax = action_tax
        self.constant_epsilon = constant_epsilon
        self.trainfn = trainfn + '.csv'
        self.testfn = testfn + '.csv'
        self.eps = eps
        self.steps = steps
        self.lrate = lrate
        self.dfactor = dfactor
        self.epsilon = epsilon
        self.cumreward = 0
        # generate q matrix where a state maps to an array of actions
        self.q_matrix = {''.join(s): [0.0] * 5 for s in product('ECW', repeat=5)}
        # generate a 10x10 game grid where each grid square has 1/2 chance to contain a can
        self.game_grid = np.random.choice([0, 1], size=(10, 10), replace=True)
        # generate random initial starting point
        self.cur_point = [randint(0, 9) for _ in range(2)]
        self.train_reward_history = []
        self.test_reward_history = []
        # dictionary to map action enum to a function that executes that action and returns a reward
        self.move_actions = {
            Action.PickUpCan: self.pick_up_can,
            Action.MoveN: lambda: self.perform_action([self.cur_point[0]-1, self.cur_point[1]]),
            Action.MoveS: lambda: self.perform_action([self.cur_point[0]+1, self.cur_point[1]]),
            Action.MoveE: lambda: self.perform_action([self.cur_point[0], self.cur_point[1]+1]),
            Action.MoveW: lambda: self.perform_action([self.cur_point[0], self.cur_point[1]-1]),
        }

    """
    Learning function that implements Q learning using parameters upon initialization
    """
    def learn(self):
        for i in range(1, self.eps+1):
            # subtract 0.01 from epsilon every 50 episodes
            if self.constant_epsilon is False and i % 50 == 0 and self.epsilon > 0.1:
                self.epsilon -= 0.01
            for j in range(self.steps):
                prev_state = self.get_cur_state()
                a_chosen = self.choose_action()
                reward = self.move_actions[Action(a_chosen)]()
                if self.action_tax is True:
                    reward -= 0.5
                self.cumreward += reward
                cur_state = self.get_cur_state()
                cur_rv = self.q_matrix[prev_state][a_chosen]
                # update the state-action value in the q matrix using action performed and new state
                self.q_matrix[prev_state][a_chosen] = cur_rv + self.lrate * (reward + self.dfactor * max(self.q_matrix[cur_state]) - cur_rv)
            if i % 100 == 0:  # log cumulative reward for episode every 100 episodes
                print(i, self.epsilon, self.cumreward)
                self.train_reward_history.append([i, self.cumreward])
            self.reset()  # reset the game grid, starting location, and cumulative reward
        # save training information to file
        pd.DataFrame(self.train_reward_history).to_csv(str(self.trainfn), index=False, header=False)

    """
    Run testing episodes with the q matrix that was formed during learning. Same steps as in learning but
    eliminate the need to update the q matrix
    """
    def test(self):
        self.epsilon = 0.01
        for i in range(1, self.eps + 1):
            for j in range(self.steps):
                a_chosen = self.choose_action()
                reward = self.move_actions[Action(a_chosen)]()
                self.cumreward += reward
            if i % 100 == 0:
                print(i, self.cumreward)
                self.test_reward_history.append([i, self.cumreward])
            self.reset()
        pd.DataFrame(self.test_reward_history).to_csv(str(self.testfn), index=False, header=False)

    """
    Reset the game board to a new distribution of cans.
    Generate a new starting point for the robot
    Reset the cumulative reward generated during the episode
    """
    def reset(self):
        self.game_grid = np.random.choice([0, 1], size=(10, 10), replace=True)
        self.cur_point = [randint(0, 9) for _ in range(2)]
        self.cumreward = 0

    """
    Looks at the current grid point the robot is in and if it has a can in it it will pick it up by setting
    that grid point to zero. Then will return a reward of 10. If grid was empty gives reward of -1
    """
    def pick_up_can(self):
        if self.get_cur_state()[0] == 'C':
            self.game_grid[self.cur_point[0]][self.cur_point[1]] = 0
            return 10
        return -1

    """
    Performs a move action up, down, left, or right by setting the current position of the given point.
    If the point to move to has a wall a reward of -5 will be returned, otherwise reward of 0.
    """
    def perform_action(self, new_state):
        if self.state_of_point(*new_state) == 'W':
            return -5
        self.cur_point = new_state
        return 0

    """
    Chooses an action based on epsilon values. The probability of choosing a random action is epsilon while the
    probability of choosing the action with the most points is 1-epsilon. Returns the chosen action.
    """
    def choose_action(self):
        a_type = np.random.choice([0, 1], p=[self.epsilon, 1-self.epsilon])
        if a_type:
            return np.argmax(self.q_matrix[self.get_cur_state()])
        return randint(0, 4)

    """
    Gets the state of the current point where the robot is on the grid by analyzing the squares around that point.
    Returns a string of length 5 representing the state of position Here, North, South, East, West.
    For example: 'CWEEW', 'EEEEC', ...
    """
    def get_cur_state(self):
        x = self.cur_point[0]
        y = self.cur_point[1]
        sensors = [(x, y), (x-1, y), (x+1, y), (x, y+1), (x, y-1)]
        state = [self.state_of_point(*pt) for pt in sensors]
        return ''.join(state)

    """
    Gets the state of the point provided saying where that point has a can (C), is empty (E) or is a wall (W)
    """
    def state_of_point(self, x, y):
        if x < 0 or x > 9 or y < 0 or y > 9:
            return 'W'
        if self.game_grid[x][y]:
            return 'C'
        return 'E'

