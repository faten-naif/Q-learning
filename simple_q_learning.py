# this is a simple q-learning that uses a Q-table as its policy.
import numpy as np
import gymnasium as gym
from collections import defaultdict
from tqdm import tqdm
import utils
class QLearning():
    def __init__(self, learning_rate:float, epsilon:float, 
                 end_epsilon:float, epsilon_decay:float, discount_factor:float,action_space:object,seed=42):
        self.lr = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.end_epsilon = end_epsilon
        self.discount_factor = discount_factor
        self.action_space = action_space
        self.training_error = []
        self.q_values = defaultdict(lambda: np.zeros(self.action_space.n))

        np.random.seed(seed)
    def get_action(self, obs):
    
        if np.random.random() < self.epsilon:
            # sample a random action (exploration)
            return self.action_space.sample()
        else:
            # select the action with highest q-value (exploitation)
            return int(np.argmax(self.q_values[obs]))
    # update the q-value of the (s,a) tuple.
    # the next observation is what we are in by taking the action.
    def update(self, obs, action, reward, done, next_obs):
        # get the highest q-value we can get in the next observation. 
        future_q_value = (not done) * np.max(self.q_values[next_obs])
        # this is the cummlative reward we can get by following this trajectory of states.
        target = reward + self.discount_factor * future_q_value
        # the difference between the current and cummlative future reward.
        # if we are taking a bad action that lead to smaller future reward the q-value will get subtracted.
        # but if we are making an action that lead us toward a better reward then then q-value will increase.
        # this is to enforce the path that lead us toward the total reward, specially when we use a small epsilon value.
        temporal_difference = target - self.q_values[obs][action]
        # the learning rate controls how much the new value will change the older one.
        self.q_values[obs][action] = self.q_values[obs][action] + self.lr * temporal_difference
        self.training_error.append(temporal_difference)
        
    def decay(self):
        self.epsilon =  max(self.epsilon * self.epsilon_decay, self.end_epsilon)
        
        
if __name__ == "__main__":
    # add MLflow to record these results
    learning_rate = 0.05
    n_epsiodes = 100_000
    epsilon = 1
    end_epsilon = 0.1
    epsilon_decay = 0.9995
    discount_factor = 0.8
    eps = []
    env = gym.make("Blackjack-v1", sab=False)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_epsiodes)
    agent = QLearning(learning_rate, epsilon, end_epsilon, epsilon_decay, discount_factor, env.action_space)

    for episode in tqdm(range(n_epsiodes)):
        # get the initial state of the environment.
        obs, info = env.reset()
        done = False
        while not done:
            # pick an action give the state of the env.
            action = agent.get_action(obs)
            # apply the action on the environment.
            next_obs, reward, terminated, truncated, info = env.step(action)
            # check if the epsiode terminated for any reason.
            done = terminated or truncated
            # update the Q-values.
            agent.update(obs, action,reward, done, next_obs)
            # update the current obs we are in.
            obs = next_obs
        # update epsilon value (action randomness factor)
        agent.decay()
        eps.append(agent.epsilon)
    # get the percentage of wins.
    rewards = env.return_queue
    wins = rewards.count(1)
    win_percentage = wins / len(rewards)
    print(f"Winning: {win_percentage:.2f}%")
    utils.graph(env,agent.training_error)
            
    