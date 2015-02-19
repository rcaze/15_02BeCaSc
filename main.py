#!/usr/bin/python
"""Collaborative project with Alex"""
import sys
sys.path.append("/home/rcaze/Documents/Scripts/PYTHON/lib/")
import numpy as np
import pandas as pd #Panda library used to handle Alex Data
import os
import h5py
import matplotlib.pyplot as plt
import unittest
from plot_data import adjust_spines

project_name = "Berdi2014"
folder = "/home/rcaze/Documents/Articles/15_02BeCaSc/Figs/"

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

def softmax(qval=[0.5,0.9], temp=2):
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

def analysis(L, G, n_chunks=10):
    """Split the vectors into n_chunks and analyse them

    Parameters
    ----------
    L: 1D binary array
        The licks, actions taken by the agent
    G: 1D binary array
        The stimulus, or goal (0 negative and 1 positive)
    """
    G_split = np.array_split(G, n_chunks)
    L_split = np.array_split(L, n_chunks)
    spe = np.zeros(n_chunks)
    sen = np.zeros(n_chunks)

    for i in range(n_chunks):
        spe[i], sen[i] = spe_sen(G_split[i], L_split[i])
    return spe, sen

def sliding_estimate(target, actual, window_s=20):
    """Compute the specificity and sensitivity iven a sliding window"""
    n_windows = len(target) - window_s
    spe_traj = range(n_windows)
    sen_traj = range(n_windows)
    for i in xrange(n_windows):
        spe, sen, matt = spe_sen(target[i:i + window_s],
                                 actual[i:i + window_s])
        spe_traj[i] = spe
        sen_traj[i] = sen
    return np.array(spe_traj), np.array(sen_traj)

def fig_gen(spe, sen, fname='fig_model.png'):
    """Plot the specificity for the early, middle and end section"""
    fig, ax = plt.subplots()
    adjust_spines(ax, ['left', 'bottom'])
    fa_rate = 1 - np.mean(spe, axis=0)
    fa_err = np.var(spe, axis=0)
    hits_rate = np.mean(sen, axis=0)
    hits_err = np.var(sen, axis=0)
    plt.plot(np.arange(1,4), hits_rate, color='#41b93c', linewidth=4)
    plt.plot(np.arange(1,4), fa_rate, color='#ec1d27', linewidth=4)
    width = 0.5
    ax.bar(np.arange(1,4)-width/2., hits_rate, width, yerr=hits_err, color='#adde76', label='HIT rate')
    ax.bar(np.arange(1,4)-width/2., fa_rate, width, yerr=fa_err, color='#f46f80', label='FA rate')
    plt.xlim(0,4)
    plt.ylabel("Response Rate")
    plt.xlabel(r'Session Start $\rightarrow$ Session End')
    plt.xticks(range(1,4), ["Initial", "Middle", "Final"])
    ax.legend()
    plt.savefig(folder + fname)

itsit = 1
q_init = np.array([[itsit,-itsit],
                   [-itsit,itsit]])
stim = stimulus(200, 0.5)

if __name__=="__main__":
    repetition = 10
    n_c = 3
    spe = np.zeros((repetition, n_c))
    sen = np.zeros((repetition, n_c))
    init_motiv = 5
    learning = False
    for i in range(repetition):
        rec_q, rec_action, rec_reward = testbed(stim, q_init, learning, init_motiv)
        spe[i], sen[i] = analysis(rec_action, stim, n_c)
    fig_gen(spe, sen, "fig_model2.png")

