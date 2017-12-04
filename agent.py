import numpy as np


class Agent():
    def __init__(self, deepnet, policy_name, agent_parameters):
        self.policies = {"epsilon-greedy" : self.epsilon_greedy,
                         "policy_right" : self.policy_right}

        self.dqn = deepnet
        self.policy = self.policies[policy_name]

        self.epsilon = agent_parameters["epsilon"]

    def epsilon_greedy(self, state, possible_actions):
        num_actions = possible_actions.n
        if np.random.uniform() < self.epsilon :
            action = possible_actions.sample()
        else :
            # TODO : predict state action value
            values = [self.dqn.predict(state=state, action=possible_actions[i])
                      for i in range(num_actions)]
            action = possible_actions[np.argmax(values)]
        return action

    def thompson_sampling(self):
        pass

    def policy_right(state):
        if state in [0, 1, 2, 3, 5, 6, 7, 8, 9]:
            a = 0
        else:
            a = 3
        return a

    def act(self, state, possible_actions):
        return self.policy(state=state,
                    possible_actions=possible_actions)
