from agent import Agent
from deepnet import DQN
import gym


class Simulator():

    def __init__(self, parameters):

        self.env = gym.make(parameters["env_name"])
        self.deep_q_net = DQN(batch_size=parameters['batch_size'],
                              optimizer=parameters["optimizer"],
                              input_shape=parameters["input_shape"])
        self.agents = [Agent(deepnet=self.deep_q_net,
                             policy_name="epsilon-greedy",
                             agent_parameters=parameters["agent_parameters"])
                       for _ in range(parameters["nb_agents"])]

    def run_episodes(self, agent, nb_episodes):
        for i in range(nb_episodes):
            episodes_rewards = []
            episodes_obs = []
            episodes_actions = []
            ob = self.env.reset()

            reward = 0
            done = False
            episode_rewards = []
            episode_obs = []

            while not done:
                action = agent.act(state=ob,
                                   possible_actions=self.env.action_space)
                ob, reward, done, _ = self.env.step(action)
                reward += reward
                episode_rewards.append(reward)
                episode_obs.append(ob)
                episodes_actions.append(action)

            episodes_rewards.append(episode_rewards)
            episodes_obs.append(episodes_obs)
            episodes_actions.append(episodes_actions)

        return {"episodes_rewards" : [sum(episode_rewards[i::]) for i in range(len(episode_rewards))],
                "episodes_obs": episodes_obs,
                "episode_actions": episodes_actions}

    def run_agents(self, agents, nb_episodes):
        simulations = []
        for agent in agents :
            simulations.append(self.run_episodes(agent, nb_episodes))
        return simulations

    def run_simulator(self, nb_iterations, nb_episodes):
        for _ in range(nb_iterations):
            simulations = self.run_agents(self.agents,
                                          nb_episodes=nb_episodes)
            self.deep_q_net.learn(simulations)
            # TODO : Complete the DQN learning
