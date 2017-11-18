import numpy as np


class Agent():
    def __init__(self, deepnet, policy_name):
        self.policies = {"epsilon-greedy" : self.epsilon_greedy,
                         "policy_right" : self.policy_right}

        self.dqn = deepnet
        self.policy = self.policies[policy_name]



    def policy_right(state):
        if state in [0, 1, 2, 3, 5, 6, 7, 8, 9]:
            a = 0
        else:
            a = 3
        return a

    def epsilon_greedy(Q, epsilon):
        def policy_fn(state):
            nA = len(env.state_actions[state])
            A = np.ones(nA, dtype=float) * epsilon / nA
            best_action = np.argmax(Q[state])
            if state == 1 and best_action == 2:
                best_action = best_action - 1
            elif state == 4 and best_action == 1:
                best_action = best_action - 1
            elif state == 4 and best_action == 3:
                best_action = best_action - 2
            elif state == 5 and best_action == 3:
                best_action = best_action - 1
            elif state == 7 and best_action == 3:
                best_action = best_action - 2
            elif state == 8 and best_action == 2:
                best_action = best_action - 1
            elif state == 9 and (best_action == 2 or best_action == 3):
                best_action = best_action - 1
            elif state == 10 and (best_action == 2 or best_action == 3):
                best_action = best_action - 2
            A[best_action] += (1.0 - epsilon)
            return A
        return policy_fn