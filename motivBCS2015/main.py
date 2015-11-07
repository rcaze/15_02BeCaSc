#!/usr/bin/python
"""Code for the generation of figure 3 in our work with Alex"""
import numpy as np
import matplotlib.pyplot as plt
import copy

project_name = "15_02BeCaSc"


def stimulus(duration, probability):
    """Generate a stimulus of length duration"""
    return np.random.binomial(1, probability, (duration,))


def motive_curve(init_val, n_trials, div=15.):
    """Define the motivation curve as a linear curve with 0 in the middle.
    This is to scale the evolution of M given the number of trials."""
    if not init_val:
        return [0 for i in range(n_trials)]

    # If there is no slope precised
    if div is None:
        div = float((n_trials-1)/2.)
    slope = init_val/div
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
    """Compute the (specificity,sensitivity) couple and the Matthews correlation
    coefficient for a desired Boolean function called ftar for a neuron
    implementing the Boolean function f.

    Parameters
    ----------
    target : array Bool
        actions taken
    actual : array Bool
        actions expected

    Returns
    -------
    spe : float between 0 and 1
        specificity of the response
    sen : float between 0 and 1
        sensitivity of the response
    """
    # Use the binary of the vector to see the difference between actual and
    # target
    tp = np.array(target)*2 - actual
    TN = len(np.repeat(tp, tp == 0))
    FN = len(np.repeat(tp, tp == 2))
    TP = len(np.repeat(tp, tp == 1))
    FP = len(np.repeat(tp, tp == -1))

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
    return qprev + alpha * (reward - qprev)


def softmax(qval=[0.5, 0.9], temp=1):
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
    gibbs = [np.exp(qval[qind]/temp)/np.sum(np.exp(qval/temp))
             for qind in range(len(qval))]
    rand = np.random.random()
    choice = 0
    partsum = 0
    for i in gibbs:
        partsum += i
        if partsum >= rand:
            return choice
        choice += 1


def testbed(states, q_init=np.zeros((2, 2)), learning=True,
            init_motiv=1, rew_motiv=False):
    """
    Launch a testbed determined by the obs array for an agent with one or two
    fix or plastic learning rates

    Parameters
    ----------
    states: array
        observation correponding to states

    Returns
    -------
    rec_choice: array
         recording the choice of the agent for each episode and each iteration
    rec_q: array
         recording the internal q-values of the agent
    rec_reward: array
         recording the learning rates
    """
    n_trials = len(states)
    # Generate the evolution of the first given the number of pos stimulus
    thirst = motive_curve(init_motiv, np.sum(states))

    rec_q = np.zeros((n_trials, 2, 2), np.float)
    rec_action = np.zeros(n_trials, np.int)
    rec_reward = np.zeros(n_trials, np.int)
    rec_thirst = np.zeros(n_trials, np.float)

    q_est = copy.deepcopy(q_init)
    q_lea = copy.deepcopy(q_init)
    th_evol = 0
    for i in range(n_trials):
        # Modify the Q estimates given the motivation
        q_est = copy.deepcopy(q_lea)
        if not rew_motiv:
            q_est = set_qnext_motiv(q_lea, thirst[th_evol])

        # Choose given the Q estimates and the state
        action = softmax(q_est[states[i]])
        # Record the choice
        rec_action[i] = action
        if states[i] == 1:
            if action == 1:
                # Diminish the thirst at each lick
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

        # Update the q_values given the state, action and reward
        if learning:
            q_lea[states[i], action] = set_qnext(q_lea[states[i], action],
                                                 reward)

        # Record Q estimate
        rec_q[i] = q_est

        # Record the thirst variable
        rec_thirst[i] = thirst[th_evol]

    return rec_q, rec_action, rec_reward, rec_thirst


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


def fig_spesen(spe, sen, fname='fig_model.png'):
    """Plot the specificity for the early, middle and end section"""
    fig, ax = plt.subplots()
    fa_rate = 1 - np.mean(spe, axis=0)
    fa_err = np.var(spe, axis=0)
    hits_rate = np.mean(sen, axis=0)
    hits_err = np.var(sen, axis=0)
    ax.plot(np.arange(1, 4), hits_rate, color='#41b93c', linewidth=4)
    ax.plot(np.arange(1, 4), fa_rate, color='#ec1d27', linewidth=4)
    width = 0.5
    ax.bar(np.arange(1, 4) - width/2., hits_rate,
           width, yerr=hits_err, color='#adde76', label='HIT rate')
    ax.bar(np.arange(1, 4) - width/2., fa_rate,
           width, yerr=fa_err, color='#f46f80', label='FA rate')
    plt.xlim(0, 4)
    plt.ylim(0, 0.8)
    plt.ylabel("Response Rate")
    plt.xlabel(r'Session Start $\rightarrow$ Session End')
    ax.set_xticks(range(1, 4), ["Initial", "Middle", "Final"])
    ax.legend()
    plt.savefig(fname)


def fig_roc(spe, sen, fname='fig_roc.png'):
    """Plot the specificity for the early, middle and end section"""
    fig, ax = plt.subplots()
    fa_rate = 1 - spe
    hits_rate = sen
    colors = ('#ec1f26', '#f79d0e', '#a6d71e')
    for i in range(3):
        plt.scatter(fa_rate[:, i], hits_rate[:, i], c=colors[i], s=120)
    ax.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1),
            color='black', linestyle="--")
    ax.set_aspect('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("FA rate")
    plt.ylabel("Hit rate")
    plt.savefig(fname)


def fig_thirst(thirst, ax=None, color=None, fname='fig_thirst.png'):
    """Plot the values at different interval in a ROC plot"""
    if not ax:
        fig, ax = plt.subplots()
    for i, th_c in enumerate(thirst):
        if not color:
            plt.plot(th_c, linewidth=4, color='black')
        else:
            plt.plot(th_c, linewidth=4, color=color)
    plt.xlim(0, len(thirst))
    plt.ylabel("Motivation variable")
    plt.xlabel("Time (bins)")
    plt.savefig(fname)
    return ax


def figs(folder="", Qinit=1, nrep=15):
    """Generate all the subplots necessary for to draw figure 3,
    except the experimental data"""
    plt.close('all')
    q_init = np.array([[0, -Qinit],
                       [0, Qinit]], dtype=np.float)
    n_c = 3
    n_trials = 100
    learning = True
    # Choose the figures format
    suf = ".png"

    # Generate the date using Alex data
    spe_d, sen_d = (np.array([[0.67715263,  0.49399997,  0.08506709],
                              [0.46347041,  0.13583542,  0.01],
                              [0.45477764,  0.1491837,  0.0196239],
                              [0.53858297,  0.01934524,  0.03432847],
                              [0.71186415,  0.26831071,  0.09960139],
                              [0.4409525,  0.04170245,  0.01],
                              [0.60690629,  0.15216989,  0.01125],
                              [0.61278915,  0.23771008,  0.01325445],
                              [0.8452229,  0.40754238,  0.02766234],
                              [0.50867757,  0.09043924,  0.01],
                              [0.66107535,  0.30309348,  0.02191489],
                              [0.7353784,  0.22671009,  0.03066106],
                              [0.59259021,  0.52808894,  0.1734353],
                              [0.56883472,  0.08478361,  0.06293826]]),
                    np.array([[0.9002687,  0.87589243,  0.42867166],
                              [0.66978751,  0.4384674,  0.32985439],
                              [0.5532464,  0.36577512,  0.24859649],
                              [0.76743038,  0.6130325,  0.27673233],
                              [0.90162076,  0.8270338,  0.33253254],
                              [0.80777079,  0.2814785,  0.18760082],
                              [0.78770642,  0.49981493,  0.25967387],
                              [0.86687442,  0.60804062,  0.08551724],
                              [0.92686771,  0.75689618,  0.40398841],
                              [0.66570871,  0.66018232,  0.40683147],
                              [0.67863222,  0.72087083,  0.09833542],
                              [0.77517302,  0.71261951,  0.45038869],
                              [0.7926511,  0.64479712,  0.74534648],
                              [0.81812082,  0.76501369,  0.06303619]]))
    fig_spesen(1-spe_d, sen_d, "fig_data" + suf)
    fig_roc(1-spe_d, sen_d, "fig_data_roc" + suf)

    repetition = spe_d.shape[0]
    spe = np.zeros((repetition, n_c))
    sen = np.zeros((repetition, n_c))
    rec_thirst = np.zeros((repetition, n_trials))

    m_spe_d = np.mean(spe_d, axis=0)
    m_sen_d = np.mean(sen_d, axis=0)

    # Generate the subplot of the models
    init_motiv = [0, 1]
    rew_motiv = [False for i in init_motiv]
    for i, c_motiv in enumerate(init_motiv):
        for j in range(repetition):
            stim = stimulus(n_trials, 0.5)
            rec_q, rec_action, rec_reward, rec_thirst[j] = testbed(stim, q_init,
                                                                   learning,
                                                                   c_motiv,
                                                                   rew_motiv[i])
            spe[j], sen[j] = analysis(rec_action, stim, n_c)
        spe_mod = np.mean(spe, axis=0)
        sen_mod = np.mean(sen, axis=0)
        sen_sc = np.linalg.norm(m_spe_d - (1-spe_mod))
        spe_sc = np.linalg.norm(m_sen_d - sen_mod)
        print sen_sc + spe_sc

        fig_spesen(spe, sen,
                   folder + "fig_mod_%d_Qinit%d%s" % (c_motiv, Qinit, suf))
        fig_roc(spe, sen,
                folder + "fig_roc_%d_Qinit%d%s" % (c_motiv, Qinit, suf))
    plt.close("all")

if __name__ == "__main__":
    figs()
