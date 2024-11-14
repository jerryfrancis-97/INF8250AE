import random

import matplotlib.pyplot as plt
import numpy as np

import algorithms as algo
import utils as utl
from environment import RaceTrack

# Set seed
seed = 0
np.random.seed(seed)
random.seed(seed)
render_mode = None

env = RaceTrack(track_map='c', render_mode=render_mode)

all_sa_values, all_returns = [], []
for i in range(5):
    sa_values, returns = algo.fv_mc_control(env, epsilon=0.01, num_episodes=1500, discount=0.99)
    all_sa_values.append(sa_values)
    all_returns.append(returns)

with open('data/fv_mc_sa_values.npy', 'wb') as f:
    np.save(f, all_sa_values)
with open('data/fv_mc_returns.npy', 'wb') as g:
    np.save(g, all_returns)

plt.figure(figsize=(15, 7))
plt.grid()
utl.plot_many(all_returns)
plt.savefig('fv_mc_returns.png')

# questions 4
# Load the last state-action values and returns
# with open('data/fv_mc_sa_values.npy', 'rb') as f:
#     all_sa_values = np.load(f, allow_pickle=True).tolist()
#     print("all_sa_values: ", len(all_sa_values))
# last_sa_values = all_sa_values[-1]

# #part 4a
# all_sa_values_4a = []
# returns_for_4a = []
# for i in range(5):
#     sa_values, returns = algo.fv_mc_control(env, epsilon=0, num_episodes=5, discount=0.99)
#     all_sa_values_4a.append(sa_values)
#     returns_for_4a.append(returns)
# with open('data/fv_mc_sa_values_4a.npy', 'wb') as f:
#     np.save(f, all_sa_values_4a)
# with open('data/fv_mc_returns_4a.npy', 'wb') as g:
#     np.save(g, returns_for_4a)

# plt.figure(figsize=(15, 7))
# plt.grid()
# utl.plot_many(returns_for_4a)
# plt.xlabel('Episodes')
# plt.ylabel('Average of Returns')
# plt.title('First-Visit Monte Carlo Control with epsilon=0 for 5 runs')
# plt.savefig('fv_mc_returns_4a.png')


# #part 4b
# all_sa_values_4b = []
# returns_for_4b = []
# for i in range(5):
#     sa_values, returns = algo.fv_mc_control(env, epsilon=0.5, num_episodes=5, discount=0.99)
#     all_sa_values_4b.append(sa_values)
#     returns_for_4b.append(returns)
# with open('data/fv_mc_sa_values_4b.npy', 'wb') as f:
#     np.save(f, all_sa_values_4b)
# with open('data/fv_mc_returns_4b.npy', 'wb') as g:
#     np.save(g, returns_for_4b)

# plt.figure(figsize=(15, 7))
# plt.grid()
# utl.plot_many(returns_for_4b)
# plt.xlabel('Episodes')
# plt.ylabel('Average of Returns')
# plt.title('First-Visit Monte Carlo Control with epsilon=0.5 for 5 runs')
# plt.savefig('fv_mc_returns_4b.png')


