#!/usr/bin/python
"""Collaborative project with Alex"""
import numpy as np
import matplotlib.pyplot as plt
import unittest
import copy

project_name = "15_02BeCaSc"
folder = "/home/rcaze/Documents/Articles/15_02BeCaSc/Figs/"

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
    q_motiv = np.zeros(q.shape, np.float)
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
    rec_thirst = np.zeros(n_trials, np.float)

    q_est = copy.deepcopy(q_init)
    q_lea = copy.deepcopy(q_init)
    th_evol = 0
    for i in range(n_trials):
        #Modify the Q estimates given the motivation
        q_est = copy.deepcopy(q_lea)
        if not rew_motiv:
            q_est = set_qnext_motiv(q_lea, thirst[th_evol])

        #Choose given the Q estimates and the state
        action = softmax(q_est[states[i]])
        #Record the choice
        rec_action[i] = action
        if states[i] == 1:
            if action == 1:
                #Diminish the thirst at each lick
                if rew_motiv:
                    reward = thirst[th_evol]
                else:
                    reward = 1
                th_evol += 1
            else:
                reward = 0
        else:
            if action == 1:
                reward = -1
            else:
                reward = 0
        rec_reward[i] = reward


        #print q_lea

        #Update the q_values given the state, action and reward
        if learning:
            q_lea[states[i], action] = set_qnext(q_lea[states[i], action], reward)
        """
        print '\n'
        print states[i], action, reward
        print '\n'
        print q_lea
        import pdb; pdb.set_trace()
        #Record Q estimate
        """
        rec_q[i] = q_est

        #Record the thirst variable
        rec_thirst[i] = thirst[th_evol]



    return rec_q, rec_action, rec_reward, rec_thirst

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

def fig1_gen(spe, sen, fname='fig_model.png'):
    """Plot the specificity for the early, middle and end section"""
    fig, ax = plt.subplots()
    #adjust_spines(ax, ['left', 'bottom'])
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

def fig2_gen(thirst, fname='fig_thirst.png'):
    """Plot the values at different interval in a ROC plot"""
    fig, ax = plt.subplots()
    #adjust_spines(ax, ['left', 'bottom'])
    for i, th_c in enumerate(thirst):
        plt.plot(th_c, linewidth=4, color='black')
    plt.xlim(0,n_trials)
    plt.ylabel("Motivation variable")
    plt.xlabel("Time (bins)")
    plt.savefig(folder + fname)

def fig3_gen(spe, sen, fname='fig_roc.png'):
    """Plot the specificity for the early, middle and end section"""
    fig, ax = plt.subplots()
    fa_rate = 1 - spe
    hits_rate = sen
    colors = ('#ec1f26','#f79d0e','#a6d71e')
    for i in range(3):
        plt.scatter(fa_rate[:,i],hits_rate[:,i], c=colors[i], s=120)
    ax.plot(np.arange(0,1.1,0.1), np.arange(0,1.1,0.1), color='black', linestyle="--")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel("FA rate")
    plt.ylabel("Hit rate")
    plt.savefig(folder + fname)

if __name__=="__main__":
    itsit = 1
    q_init = np.array([[itsit,-itsit],
                       [-itsit,itsit]], dtype=np.float)
    repetition = 15
    n_c = 3
    n_trials = 150
    spe = np.zeros((repetition, n_c))
    sen = np.zeros((repetition, n_c))
    rec_thirst = np.zeros((repetition, n_trials))
    rew_motiv = False
    init_motiv = 0
    learning = True
    for i in range(repetition):
        stim = stimulus(n_trials, 0.5)
        rec_q, rec_action, rec_reward, rec_thirst[i] = testbed(stim, q_init, learning, init_motiv, rew_motiv)
        spe[i], sen[i] = analysis(rec_action, stim, n_c)
    fig1_gen(spe, sen, "fig_model2.png")
    fig2_gen(rec_thirst, "fig_thirst2.png")
    fig3_gen(spe, sen, "fig_roc2.png")
    #print rec_q[-1]
    #plt.show()
