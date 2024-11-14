import random

import matplotlib.pyplot as plt
import numpy as np

import utils as utl
from environment import RaceTrack
from td_algos import QLearningAgent, Sarsa, td_control
import time

# from concurrent.futures import ThreadPoolExecutor

# Set seed
seed = 0
np.random.seed(seed)
random.seed(seed)
render_mode = None

num_episodes = 2000
env = RaceTrack(track_map='c', render_mode=render_mode)

info = [{"env": env,
        "step_size": 0.1,
        "epsilon": 0.05,
        "discount": 0.99,
        'seed': i
        } for i in range(5)]

returns = []
agents = []

# st = time.time()
for i in range(5):
        print("Params:", info[i])
        all_returns, agent = td_control(env, agent_class=Sarsa, info=info[i], num_episodes=num_episodes)
        returns.append(all_returns)
        agents.append(agent)
        print("SARSA AGENT done:", i, time.time() - st)
# def run_sarsa_agent(i):
#         print("Params:", info[i])
#         all_returns, agent = td_control(env, agent_class=Sarsa, info=info[i], num_episodes=num_episodes)
#         print("SARSA AGENT done:", i, time.time() - st)
#         return all_returns, agent

# with ThreadPoolExecutor(max_workers=5) as executor:
#         results = list(executor.map(run_sarsa_agent, range(5)))

# returns, agents = zip(*results)
# returns = list(returns)
# agents = list(agents)


with open('data/td_sarsa_returns.npy', 'wb') as f:
        np.save(f, returns)
with open('data/td_sarsa_agents.npy', 'wb') as g:
        np.save(g, agents)

plt.figure(figsize=(15, 7))
plt.grid()
utl.plot_many(returns)
plt.savefig('td_sarsa.png')
print("Time taken for SARSA: ", time.time() - st)
print("Sample sarsa agent: ", agents[0])
print("=====================================")

# q-learning task
returns = []
agents = []
st = time.time()
for i in range(5):
        print("Params:", info[i])
        all_returns, agent = td_control(env, agent_class=QLearningAgent, info=info[i], num_episodes=num_episodes)
        returns.append(all_returns)
        agents.append(agent)
        print("Q-LEARNING AGENT done:", i, time.time() - st)
# def run_qlearning_agent(i):
#         print("Params:", info[i])
#         all_returns, agent = td_control(env, agent_class=QLearningAgent, info=info[i], num_episodes=num_episodes)
#         print("Q-LEARNING AGENT done:", i, time.time() - st)
#         return all_returns, agent

# with ThreadPoolExecutor(max_workers=5) as executor:
#         results = list(executor.map(run_qlearning_agent, range(5)))

# returns, agents = zip(*results)
# returns = list(returns)
# agents = list(agents)

with open('data/td_qlearning_returns.npy', 'wb') as f:
        np.save(f, returns)
with open('data/td_qlearning_agents.npy', 'wb') as g:
        np.save(g, agents)

plt.figure(figsize=(15, 7))
plt.grid()
utl.plot_many(returns)
plt.savefig('td_qlearning.png')
print("Time taken for Q-learning: ", time.time() - st)
print("Sample qlearning agent: ", agents[0])
print("=====================================")

# #question 9 c
# #load sarsa agents
# sarsa_returns = []
# agents=[]
# # with open('data/td_sarsa_agents.npy', 'rb') as f:
# #         sarsa_agents = np.load(f, allow_pickle=True).tolist()
# #         print("SARSA agents: ", len(sarsa_agents))
# sarsa_info = [{"env": env,
#         "step_size": 0.1,
#         "epsilon": 0,
#         "discount": 0.99,
#         'seed': i
#         } for i in range(5)]

# modified td_Control for this exp
# for i in range(5):
#         print("Params:", sarsa_info[i])
#         all_returns, agent = td_control(env, agent_class=Sarsa, info=sarsa_info[i], num_episodes=1, index=i)
#         sarsa_returns.append(all_returns)
#         agents.append(agent)
# print("AVerage sarsa return: ", sarsa_returns, np.mean(sarsa_returns))



# #question 11 c
# #load qlearning agents
# qlearning_returns = []
# agents=[]
# # with open('data/td_qlearning_agents.npy', 'rb') as f:
# #         qlearning_agents = np.load(f, allow_pickle=True).tolist()
# #         print("Qlearning agents: ", len(qlearning_agents))
# qlearn_info = [{"env": env,
#         "step_size": 0.1,
#         "epsilon": 0,
#         "discount": 0.99,
#         'seed': i
#         } for i in range(5)]

# modified td_Control for this exp
# for i in range(5):
#         print("Params:", qlearn_info[i])            
#         all_returns, agent = td_control(env, agent_class=QLearningAgent, info=qlearn_info[i], num_episodes=1, index=i)
#         qlearning_returns.append(all_returns)
#         agents.append(agent)
# print("AVerage qlearning return: ", qlearning_returns, np.mean(qlearning_returns))


# #question 12
# sarsa_returns = []
# agents=[]
# sarsa_info = [{"env": env,
#         "step_size": 0.1,
#         "epsilon": 0.2,
#         "discount": 0.99,
#         'seed': i
#         } for i in range(5)]

# # modified td_Control for this exp
# for i in range(5):
#         print("Params:", sarsa_info[i])
#         # all_returns, agent = td_control(env, agent_class=Sarsa, info=sarsa_info[i], num_episodes=5, index=4)
#         all_returns, agent = td_control(env, agent_class=QLearningAgent, info=sarsa_info[i], num_episodes=5, index=4)
#         sarsa_returns.append(all_returns)
#         agents.append(agent)
# print("12. AVerage sarsa return: ", sarsa_returns, np.mean(sarsa_returns), np.std(sarsa_returns))
# # print("12. AVerage Qlearning return: ", sarsa_returns, np.mean(sarsa_returns), np.std(sarsa_returns))
# plt.figure(figsize=(15, 7))
# plt.grid()
# utl.plot_many(sarsa_returns)
# plt.xlabel('Episodes')
# plt.ylabel('Average of Returns')
# # plt.savefig('td_sarsa_12.png')
# plt.savefig('td_qlearn_12.png')

