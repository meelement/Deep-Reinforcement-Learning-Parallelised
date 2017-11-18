from deepnet import DQN
from agent import Agent
from parameters import parameters

deep_q_net = DQN(batch_size=parameters['batch_size'],
                 optimizer=parameters["optimizer"])
agents = [Agent(deepnet=deep_q_net, policy_name="epsilon_greedy") for i in range(parameters["nb_agents"])]

# Interaction with the environment
