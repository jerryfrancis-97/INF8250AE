from typing import Tuple, List
import numpy as np
from heroes import Heroes
from helpers import run_trials, save_results_plots

def eps_greedy(
    heroes: Heroes, 
    eps: float, 
    init_value: float = .0
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Perform epsilon-greedy action selection for a bandit problem.

    :param heroes: A bandit problem, instantiated from the Heroes class.
    :param eps: The epsilon value for exploration vs. exploitation.
    :param init_value: Initial estimation of each hero's value.
    :return: 
        - rew_record: The record of rewards at each timestep.
        - avg_ret_record: The average of rewards up to step t. For example: If 
    we define `ret_T` = \sum^T_{t=0}{r_t}, `avg_ret_record` = ret_T / (1+T).
        - tot_reg_record: The total regret up to step t.
        - opt_action_record: Percentage of optimal actions selected.
    """
    
    num_heroes = len(heroes.heroes)
    values = [init_value] * num_heroes    # Initial action values
    rew_record = []                       # Rewards at each timestep
    avg_ret_record = []                   # Average reward up to each timestep
    tot_reg_record = []                   # Total regret up to each timestep
    opt_action_record = []                # Percentage of optimal actions selected
    
    total_rewards = 0
    total_regret = 0

    ######### WRITE YOUR CODE HERE
    optimal_reward = np.max(heroes.true_probability_list)
    optimal_hero_index = np.argmax(heroes.true_probability_list)
    
    total_optimal_actions = 0

    # print(values)
    # print("trials: ", heroes.total_quests)
    # print("optimal rewa & hro index: ", optimal_reward, optimal_hero_index)
    ######### 
    
    for t in range(heroes.total_quests):
        ######### WRITE YOUR CODE HERE
        if np.random.uniform(0, 1) > eps:
            # take greedy action
            best_value = np.max(values)
            # incase of tie
            available_heroes = np.where(values == best_value)[0]
            chosen_hero = np.random.choice(available_heroes)
        else:
            # take random action
            chosen_hero = np.random.randint(num_heroes)

        current_reward = heroes.attempt_quest(chosen_hero)
        values[chosen_hero] += 1/(heroes.heroes[chosen_hero]['n_quests']) * (current_reward - values[chosen_hero])

        regret = optimal_reward - current_reward
        total_regret += regret.item()
        total_rewards += current_reward


        optimal_action_percentage = heroes.heroes[optimal_hero_index]['n_quests']/(t + 1)

        rew_record.append(current_reward)
        avg_ret_record.append(total_rewards/len(rew_record))
        tot_reg_record.append(total_regret)
        opt_action_record.append(optimal_action_percentage)

    #########
    # print("reward rec: ", len(rew_record), rew_record)
    # print("avg reward rec: ", len(avg_ret_record), avg_ret_record) 
    # print("total regert: ", len(tot_reg_record), tot_reg_record) 
    # print("optimal action percentage: ", len(opt_action_record), opt_action_record)
    
    return rew_record, avg_ret_record, tot_reg_record, opt_action_record


if __name__ == "__main__":
    # Define the bandit problem
    heroes = Heroes(total_quests=3000, true_probability_list=[0.35, 0.6, 0.1])


    # Test various epsilon values
    eps_values = [0.2, 0.1, 0.01, 0.]
    results_list = []
    for eps in eps_values:
        rew_rec, avg_ret_rec, tot_reg_rec, opt_act_rec = run_trials(30, 
                                                                    heroes=heroes, bandit_method=eps_greedy, 
                                                                    eps=eps, init_value=0.0)
        
        results_list.append({
            'exp_name': f'eps={eps}',
            'reward_rec': rew_rec,
            'average_rew_rec': avg_ret_rec,
            'tot_reg_rec': tot_reg_rec,
            'opt_action_rec': opt_act_rec
        })

    save_results_plots(results_list, plot_title='Epsilon-Greedy Experiment Results On Various Epsilons', 
                       results_folder='results', pdf_name='epsilon_greedy_various_epsilons.pdf')


    # Test various initial value settings with eps=0.0
    init_values = [0.0, 0.5, 1]
    results_list = []
    for init_val in init_values:
        rew_rec, avg_ret_rec, tot_reg_rec, opt_act_rec = run_trials(30, 
                                                                    heroes=heroes, bandit_method=eps_greedy, 
                                                                    eps=0.0, init_value=init_val)
        
        results_list.append({
            'exp_name': f'init_val={init_val}',
            'reward_rec': rew_rec,
            'average_rew_rec': avg_ret_rec,
            'tot_reg_rec': tot_reg_rec,
            'opt_action_rec': opt_act_rec
        })
    
    save_results_plots(results_list, plot_title='Epsilon-Greedy Experiment Results On Various Initial Values',
                       results_folder='results', pdf_name='epsilon_greedy_various_init_values.pdf')
    