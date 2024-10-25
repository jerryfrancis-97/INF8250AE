import numpy as np

import utils as utl
from environment import RaceTrack
from utils import Action, ActionValueDict, DistributionPolicy, State, StateAction


def fv_mc_estimation(states : list[State], actions: list[Action], rewards: list[float], discount: float) -> ActionValueDict:
    """
        Runs Monte-Carlo prediction for given transitions with the first visit heuristic for estimating the values

        :param list[tuple] states: list of states of an episode generated from generate_episode
        :param list[int] actions: list of actions of an episode generated from generate_episode
        :param list[float] rewards: list of rewards of an episode generated from generate_episode
        :param float discount: discount factor

        :return visited_states_returns (dict[tuple,float]): dictionary where the keys are the unique state-action combinations visited during the episode
        and the value of each key is the estimated discounted return of the first visitation of that key (state-action pair)
    """
    visited_sa_returns = {}

    # TO IMPLEMENT
    # --------------------------------
    rewards = np.array(rewards, dtype=np.float64)
    # print("rewards", rewards)
    gammas  = np.power(discount, np.arange(len(rewards)))
    # print("discount: ", discount)
    # print("gammas: ", gammas)
    rewards *= gammas
    # print("discounted rewards: ", rewards)
    returns = np.cumsum(rewards[::-1])[::-1]
    # print("returns: ", returns)

    for i, (s, a) in enumerate(zip(states, actions)):
        # print("s: ", s)
        # print("a: ", a)
        # print("sa: ", (*s, a))
        # print("returns[i]: ", returns[i])
        if (*s, a) not in visited_sa_returns:
            visited_sa_returns[(*s, a)] = returns[i]

    # print("visited_sa_returns: ", len(visited_sa_returns))
    # --------------------------------
    # print("visited_sa_returns: ", len(visited_sa_returns), visited_sa_returns.items())  


    return visited_sa_returns


def fv_mc_control(env: RaceTrack, epsilon: float, num_episodes: int, discount: float) -> tuple[ActionValueDict, list[float]]:
    """
        Runs Monte-Carlo control, using first-visit Monte-Carlo for policy evaluation and regular policy improvement

        :param RaceTrack env: environment on which to train the agent
        :param float epsilon: epsilon value to use for the epsilon-greedy policy
        :param int num_episodes: number of iterations of policy evaluation + policy improvement
        :param float discount: discount factor

        :return visited_states_returns (dict[tuple,float]): dictionary where the keys are the unique state-action combinations visited during the episode
        and the value of each key is the estimated discounted return of the first visitation of that key (state-action pair)
        :return all_returns (list[float]): list of all the cumulative rewards the agent earned in each episode
    """
    # Initialize memory of estimated state-action returns
    # with open('data/fv_mc_sa_values.npy', 'rb') as f:
    #     all_sa_values = np.load(f, allow_pickle=True).tolist()
    #     print("all_sa_values: ", len(all_sa_values))
    # state_action_values = all_sa_values[-1]

    state_action_values = utl.init_q_and_v()[1]
    all_state_action_values = {}
    all_returns = []

    # TO IMPLEMENT
    # --------------------------------
    policy = utl.make_eps_greedy_policy(state_action_values, epsilon)
    for i in range(num_episodes):
        #policy estimation
        states, actions, rewards = utl.generate_episode(policy, env)
        visited_sa_returns = fv_mc_estimation(states, actions, rewards, discount)
        all_returns.append(sum(rewards))
        
        #policy improvement
        for sa in visited_sa_returns:
            state_action_values[sa] = visited_sa_returns[sa]
        policy = utl.make_eps_greedy_policy(state_action_values, epsilon)

    # --------------------------------

    return state_action_values, all_returns


def is_mc_estimate_with_ratios(
    states: list[State],
    actions: list[Action],
    rewards: list[float],
    target_policy: DistributionPolicy,
    behaviour_policy: DistributionPolicy,
    discount: float
) -> dict[StateAction, list[tuple[float, float]]]:
    """
        Computes Monte-Carlo estimated q-values for each state in an episode in addition to the importance sampling ratio
        associated to that state

        :param list[tuple] states: list of states of an episode generated from generate_episode
        :param list[int] actions: list of actions of an episode generated from generate_episode
        :param list[float] rewards: list of rewards of an episode generated from generate_episode
        :param (int -> list[float]) target_policy: The initial target policy that takes in a state and returns
                                            an action probability distribution (the one we are  learning)
        :param (int -> list[float]) behavior_policy: The behavior policy that takes in a state and returns
                                            an action probability distribution
        :param float discount: discount factor

        :return state_action_returns_and_ratios (dict[tuple,list[tuple]]):
        Keys are all the states visited in the input episode
        Values is a list of tuples. The first index of the tuple is
        the IS estimate of the discounted returns
        of that state in the episode. The second index is the IS ratio
        associated with each of the IS estimates.
        i.e: if state (2, 0, -1, 1) is visited 3 times in the episode and action '7' is always taken in that state,
        state_action_returns_and_ratios[(2, 0, -1, 1, 7)] should be a list of 3 tuples.
    """
    state_action_returns_and_ratios = {}

    # TO IMPLEMENT
    # --------------------------------
    rewards = np.array(rewards, dtype=np.float64)
    gammas  = np.power(discount, np.arange(len(rewards)))
    rewards *= gammas
    returns = np.cumsum(rewards[::-1])[::-1]

    prob_ratios = []
    for i, (s, a) in enumerate(zip(states, actions)):
        target_prob = target_policy(s)[a]
        behavior_prob = behaviour_policy(s)[a]
        prob_ratios.append(target_prob / behavior_prob)
    is_ratios = np.array(prob_ratios, dtype=np.float64)
    is_ratios = np.cumprod(prob_ratios[::-1])[::-1]

    for i, (s, a) in enumerate(zip(states, actions)):
        if (*s, a) not in state_action_returns_and_ratios:
            state_action_returns_and_ratios[(*s, a)] = []
        #every state-action has is returns and ratios
        state_action_returns_and_ratios[(*s, a)].append(( returns[i], is_ratios[i]))
    # --------------------------------

    return state_action_returns_and_ratios


def ev_mc_off_policy_control(env: RaceTrack, behaviour_policy: DistributionPolicy, epsilon: float, num_episodes: int, discount: float):
     # Initialize memory of estimated state-action returns
    state_action_values = utl.init_q_and_v()[1]
    all_state_action_values = {}
    all_returns = []

    # TO IMPLEMENT
    # --------------------------------
    behave_sample_policy = utl.convert_to_sampling_policy(behaviour_policy)
    target_policy = utl.make_eps_greedy_policy_distribution(state_action_values, epsilon)
    every_visit_state_action_values = {}
    from tqdm import trange
    for i in trange(num_episodes):
        #policy estimation
        states, actions, rewards = utl.generate_episode(behave_sample_policy, env)
        state_action_returns_and_ratios = is_mc_estimate_with_ratios(states, actions, rewards, target_policy, behaviour_policy, discount)
        #applying every visit condition to state_action_returns_and_ratios
        for state_action, returns_and_ratios in state_action_returns_and_ratios.items():
            every_vist_return = []
            cumu_is_ratio = 0
            for ret, is_ratio in returns_and_ratios:
                cumu_is_ratio += 1 #ordinary importance sampling
                every_vist_return.append((ret - state_action_values[state_action]) * is_ratio / cumu_is_ratio)
            total_return = np.sum(every_vist_return)
            every_visit_state_action_values[state_action] = total_return
        all_returns.append(sum(rewards))

        #policy improvement
        for sa in every_visit_state_action_values:
            state_action_values[sa] = every_visit_state_action_values[sa]
        target_policy = utl.make_eps_greedy_policy_distribution(state_action_values, epsilon)

    # --------------------------------

    return state_action_values, all_returns
