"""Project on the collaboration with Alex"""
import numpy as np
import matplotlib.pyplot as plt
import unittest
from traits.api import HasTraits, Float, Function, Array, List, Int, Dict

def stimulus(duration, probability):
    """Generate a stimulus of length duration"""
    return np.random.binomial(1, probability, (duration,))


class TestGeneratingStimulus(unittest.TestCase):
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

#Working but not used only when required KIS
class Agent(HasTraits):
    """Define an reinforcement learning agent"""
    alpha = Float(0.1)
    decision_function = Function(softmax())
    q_values = Array(dtype=np.float, value=np.zeros((2,2)))
    q_rec = List()
    temperature = Float(0.3)

    def choose(self, state):
        """Produce an action given the q-values and the state"""
        return self.decision_function(self.q_values[state], self.temperature)

    def update(self, state, action, reward):
        """Update its q-values given the reward and the state"""
        qnext = set_qnext(self.q_values[state, action], reward)
        self.q_values[state, action] = qnext
        self.q_rec.append(qnext)

#Not used now, only when required KIS principle
class Task(HasTraits):
    """Define a task"""
    duration = Int(100)
    states = Array(value=np.random.binomial(1,0.5,100))
    actions = Dict({0:0, 1:1})
    expected_action = Array()
    reward = Array()
    agents = List(Agent)

    def __init__(self):
        #Setting the reward like in Alex's task
        self.expected_action = self.states
        if len(self.agents) >= 1:
            self.actions = np.zeros((self.duration, len(self.agents)))
        else:
            self.actions = np.zeros(self.duration)

    def one_step(self, state):
        """compute one simulation step"""
        action = self.agents[0].choose(state)
        if action == self.expected_action(state):
            reward = 1
        else:
            reward = -1
        self.agents[0].update(state, action, reward)


def testbed(states, q_init = np.zeros((2,2))):
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
    expected_actions = states

    rec_q = np.zeros((n_trials, 2, 2), np.float)
    rec_action = np.zeros(n_trials, np.int)
    rec_reward = np.zeros(n_trials, np.int)

    q_est = q_init
    for i in range(n_trials):
        #Choose given the Q estimates and the state
        action = softmax(q_est[states[i]])
        #Record the choice
        rec_action[i] = action
        if action == expected_actions[i]:
            reward = 1
        else:
            reward = -1
        rec_reward[i] = reward
        #Update the q_values gove the state, action and reward
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

def spe_sen(target,actual):
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

    if (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) == 0:
        matt = (TP*TN-FP*FN)
    else:
        matt = (TP*TN-FP*FN)/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**(0.5))

    return spe, sen, matt

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

def ROC(spe_traj, sen_traj, colors = ('red', 'blue', 'yellow', 'green')):
    """Plotting the specificity trajectory and sensitivity trajectory"""
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    false_alarm = 1 - spe_traj
    false_alarm = np.array_split(false_alarm, len(colors))
    sen_traj = np.array_split(sen_traj, len(colors))
    for i, color in enumerate(colors):
        for j in range(len(false_alarm[i])):
            ax.scatter(false_alarm[i][j], sen_traj[i][j], color=color)
    ax.set_xlim(0,1)
    ax.set_xlabel('False Alarm rate')
    ax.set_ylim(0,1)
    ax.set_ylabel('Hit rate')
