import numpy as np


class Agent():
    def __init__(self, deepnet, policy_name, agent_parameters):
        self.policies = {"epsilon-greedy" : self.epsilon_greedy}

        self.dqn = deepnet
        self.policy = self.policies[policy_name]

        self.epsilon = agent_parameters["epsilon"]

    def epsilon_greedy(self, state, possible_actions):
        num_actions = possible_actions.n
        if np.random.uniform() < self.epsilon :
            action = possible_actions.sample()
        else :
            # TODO : predict state action value
            values = [self.dqn.predict(state=state, action=i) for i in range(num_actions)]
            action = np.argmax(values)
        return action

    def thompson_sampling(self):
        pass

    def act(self, state, possible_actions):
        return self.policy(state=state,
                    possible_actions=possible_actions)
