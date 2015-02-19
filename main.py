#!/usr/bin/python
"""Collaborative project with Alex"""
import numpy as np
import pandas as pd #Panda library used to handle Alex Data
import os
import h5py
import matplotlib.pyplot as plt
import unittest
from traits.api import HasTraits, Float, Function, Array, List, Int, Dict

project_name = "Berdi2014"

def import_data_berdi():
    path = "~/Data/" + project_name + "/"
    os.chdir(path) #Change of directory
    for fname in os.dirlist(path):
        with open(fname) as f:
            pd.read_csv(f)

def stimulus(duration, probability):
    """Generate a stimulus of length duration"""
    return np.random.binomial(1, probability, (duration,))

class TestStimulus(unittest.TestCase):
    """Testing the generation of stimulus"""
    def setUp(self):
        self.duration = 100000
        self.probability = 0.5

    def test_stimulus(self):
        """Test the generation of stimulus"""
        #Given
        duration = 10000
        probability = 0.5
        #When the stimulus generation is simple
        test_stim = stimulus(duration, self.probability)
        #Then
        self.assertAlmostEqual(np.mean(test_stim), probability, 1)

def analysis(L, G, n_chunks=10):
    """Split the vectors into n_chunks"""
    G_split = np.array_split(G, n_chunks)
    L_split = np.array_split(L, n_chunks)
    spe = np.zeros(n_chunks)
    sen = np.zeros(n_chunks)

    for i in range(n_chunks):
        spe[i], sen[i] = spe_sen(G_split[i], L_split[i])
    return spe, sen

def motive_curve(init_val, n_trials):
    """Define the motivation curve as a linear curve with 0 in the middle"""
    slope = init_val/float((n_trials-1)/2.)
    return [init_val - i*slope for i in range(n_trials)]

def set_qnext_motiv(q, motivation):
    """Modify the Q-values depending on the motivation of the agent"""
    q_motiv = np.zeros(q.shape)
    for action in range(2):
        for state in range(2):
            if action == 1:
                q_motiv[state, action] = q[state, action] + motivation
            else:
                q_motiv[state, action] = q[state, action] - motivation
    return q_motiv

def spe_sen(target, actual):
    """Compute the (specificity,sensitivity) couple and the Matthews correlation coefficient
    for a desired Boolean function called ftar for a neuron implementing the Boolean function f.

    parameters
    ----------
    target : Boolean array
        actions taken
    actual : Boolean array
        actions expected

    returns
    -------
    spe : a float between 0 and 1
    sen : a float between 0 and 1
    """
    #Use the binary of the vector to see the difference between actual and target
    tp = np.array(target)*2 - actual
    TN = len(np.repeat(tp,tp==0))
    FN = len(np.repeat(tp,tp==2))
    TP = len(np.repeat(tp,tp==1))
    FP = len(np.repeat(tp,tp==-1))

    spe = float(TN)/(TN+FP)
    sen = float(TP)/(TP+FN)

    return spe, sen

def set_qnext(qprev, reward, alpha=0.1):
    """
    Update a Q value given a distinct learning rate for reward and punishment

    PARAMETERS
    ----------
    qprev: float
        q-value to be changed
    reward: integer
        reward value
    alpha: float
        learning rate for reward

    RETURNS
    -------
    Qnext: float
        estimation of the Q-value at the next time step
    """
    return qprev + alpha * ( reward - qprev )

def softmax(qval=[0.5,0.9], temp=1):
    """
    Generate a softmax choice given a Q-value and a temperature parameter

    PARAMETERS
    ----------
    qval: list of floats
        Q-values corresponding to the different arm
    temp: float
        temperature parameter

    RETURNS
    -------
    choice : an integer, the index of the chosen action
    """
    qval = np.array(qval)
    gibbs = [np.exp(qval[qind]/temp)/np.sum(np.exp(qval/temp)) for qind in range(len(qval))]
    rand = np.random.random()
    choice = 0
    partsum = 0
    for i in gibbs:
        partsum += i
        if partsum >= rand:
            return choice
        choice += 1

def testbed(states, q_init = np.zeros((2,2)), learning=True, init_motiv=1, rew_motiv=False):
    """
    Launch a testbed determined by the obs array for an agent with one or two fix or plastic learning rates

    PARAMETERS
    -----
    states: array
        observation correponding to states

    RETURNS
    -------
    rec_choice: array
         recording the choice of the agent for each episode and each iteration
    rec_q: array
         recording the internal q-values of the agent
    rec_reward: array
         recording the learning rates
    """
    n_trials = len(states)
    # Generate the evolution of the first given the number of positive stimulus
    thirst = motive_curve(init_motiv, np.sum(states))

    rec_q = np.zeros((n_trials, 2, 2), np.float)
    rec_action = np.zeros(n_trials, np.int)
    rec_reward = np.zeros(n_trials, np.int)

    q_est = q_init
    th_evol = 0
    for i in range(n_trials):
        #Modify the Q estimates given the motivation
        q_est = set_qnext_motiv(q_init, thirst[th_evol])
        #Choose given the Q estimates and the state
        action = softmax(q_est[states[i]])
        #Record the choice
        rec_action[i] = action
        if states[i] == 1:
            if action == 1:
                #Diminish the thirst at each lick
                th_evol += 1
                if rew_motiv:
                    reward = thirst[th_evol]
                else:
                    reward = 1
            else:
                reward = 0
        else:
            reward = 0
        rec_reward[i] = reward
        #Update the q_values given the state, action and reward
        if learning:
            q_est[states[i], action] = set_qnext(q_est[states[i], action], reward)

        #Record Q estimate
        rec_q[i] = q_est
        #import pdb; pdb.set_trace()

    return rec_q, rec_action, rec_reward

class TestTestbed(unittest.TestCase):
    """Testing the testbed"""
    def setUp(self):
        prob = 0.5
        duration = 100
        obs = np.array([stimulus(prob, duration),
                        stimulus(prob, duration)])
        rec_q, rec_choice, rec_rewad = testbed(obs)

def motiv_expe(itsit, learning, motivation):
    """An experiment where the motivation matters
    #Define the intensity of licking
    itsit = 1"""
    #For action 0 (no-go) then there is no-lick (0)
    #And for action
    q_init = np.array([[itsit,-itsit],
                       [-itsit,itsit]])

    states = stimulus(200, 0.5)
    rec_q, rec_action, rec_reward = testbed(states, q_init, learning=learning, init_motiv=motivation)

    return states, rec_action


def sliding_estimate(target, actual, window_s=20):
    """Compute the specificity and sensitivity given a sliding window"""
    n_windows = len(target) - window_s
    spe_traj = range(n_windows)
    sen_traj = range(n_windows)
    for i in xrange(n_windows):
        spe, sen, matt = spe_sen(target[i:i + window_s],
                                 actual[i:i + window_s])
        spe_traj[i] = spe
        sen_traj[i] = sen
    return np.array(spe_traj), np.array(sen_traj)


if __name__=="__main__":
    repetition = 10
    n_c = 3
    spe = np.zeros((repetition, n_c))
    sen = np.zeros((repetition, n_c))
    itsit = 1
    motiv = 2
    learning = False
    for i in range(repetition):
        G, L = motiv_expe(itsit, learning, motiv)
        spe[i], sen[i] = analysis(L, G, n_c)
    plt.plot(1 - np.mean(spe, axis=0), color='r')
    plt.plot(np.mean(sen, axis=0), color='g')
    plt.savefig('fig3.png')
    plt.show()

