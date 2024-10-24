from typing import Tuple, List
import numpy as np
from heroes import Heroes
from helpers import run_trials, save_results_plots

def softmax(x, tau=1):
    """ Returns softmax probabilities with temperature tau
        Input:  x -- 1-dimensional array
        Output: idx -- chosen index
    """
    
    e_x = np.exp(np.array(x) / tau)
    return e_x / e_x.sum(axis=0)


def gradient_bandit(
    heroes: Heroes, 
    alpha: float, 
    use_baseline: bool = True,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Perform Gradient Bandit action selection for a bandit problem.

    :param heroes: A bandit problem, instantiated from the Heroes class.
    :param alpha: The learning rate.
    :param use_baseline: Whether or not use avg return as baseline.
    :return: 
        - rew_record: The record of rewards at each timestep.
        - avg_ret_record: TThe average of rewards up to step t. For example: If 
    we define `ret_T` = \sum^T_{t=0}{r_t}, `avg_ret_record` = ret_T / (1+T).
        - tot_reg_record: The total regret up to step t.
        - opt_action_record: Percentage of optimal actions selected.
    """

    num_heroes = len(heroes.heroes)
    h = np.array([0]*num_heroes, dtype=float)  # init h (the logits)
    rew_record = []                            # Rewards at each timestep
    avg_ret_record = []                        # Average reward up to each timestep
    tot_reg_record = []                        # Total regret up to each timestep
    opt_action_record = []                     # Percentage of optimal actions selected
    
    reward_bar = 0
    total_rewards = 0
    total_regret = 0

    ######### WRITE YOUR CODE HERE
    optimal_reward = np.max(heroes.true_probability_list)
    # print(type(optimal_reward), optimal_reward)
    optimal_hero_index = np.argmax(heroes.true_probability_list)
    ######### 
    identity_mat = np.eye(num_heroes)

    for t in range(heroes.total_quests):
        ######### WRITE YOUR CODE HERE
        #preference calc
        policy_prob_pi = softmax(h)
        chosen_hero = np.random.choice(num_heroes, p=policy_prob_pi)
        current_reward = heroes.attempt_quest(chosen_hero)

        identity_at_chosen_hero = identity_mat[chosen_hero]
        probs = identity_at_chosen_hero - policy_prob_pi
        reward_bar += 1/(t+1) * (current_reward - reward_bar)

        if use_baseline:
            h = h + alpha * (current_reward - reward_bar) * probs
            # print(f"alpha: {alpha}, h: {h}, {(current_reward - reward_bar)} {(identity_at_chosen_hero - policy_prob_pi)} last term: {(current_reward - reward_bar) * (identity_at_chosen_hero - policy_prob_pi)}")
        else:
            h = h + alpha * current_reward * probs

        regret = optimal_reward - current_reward
        # print(type(current_reward), current_reward, regret)
        total_regret += regret.item()

        optimal_action_percentage = heroes.heroes[optimal_hero_index]['n_quests']/(t + 1)

        total_rewards += current_reward
        rew_record.append(current_reward)
        avg_ret_record.append(total_rewards/len(rew_record))
        tot_reg_record.append(total_regret)
        opt_action_record.append(optimal_action_percentage)
        ######### 
    # print("reward rec: ", len(rew_record), sum(rew_record), rew_record)
    # print("avg reward rec: ", len(avg_ret_record), sum(avg_ret_record), avg_ret_record) 
    # print("total regert: ", len(tot_reg_record), sum(tot_reg_record), tot_reg_record) 
    # print("optimal action percentage: ", len(opt_action_record), sum(opt_action_record), opt_action_record) 

    return rew_record, avg_ret_record, tot_reg_record, opt_action_record

if __name__ == "__main__":
    # Define the bandit problem
    heroes = Heroes(total_quests=3000, true_probability_list=[0.35, 0.6, 0.1])

    # Test various alpha values with baseline
    alpha_values = [0.05, 0.1, 2]
    results_list = []
    for alpha in alpha_values:
        rew_rec, avg_ret_rec, tot_reg_rec, opt_act_rec = run_trials(30,
                                                                    heroes=heroes, bandit_method=gradient_bandit,
                                                                    alpha=alpha, use_baseline=True)
        results_list.append({
            "exp_name": f"alpha={alpha}",
            "reward_rec": rew_rec,
            "average_rew_rec": avg_ret_rec,
            "tot_reg_rec": tot_reg_rec,
            "opt_action_rec": opt_act_rec
        })
    
    save_results_plots(results_list, plot_title="Gradient Bandits (with Baseline) Experiment Results On Various Alpha Values",
                       results_folder='results', pdf_name='gradient_bandit_various_alpha_values_with_baseline.pdf')

    # Test various alpha values without baseline
    alpha_values = [0.05, 0.1, 2]
    results_list = []
    for alpha in alpha_values:
        rew_rec, avg_ret_rec, tot_reg_rec, opt_act_rec = run_trials(30,
                                                                    heroes=heroes, bandit_method=gradient_bandit,
                                                                    alpha=alpha, use_baseline=False)
        results_list.append({
            "exp_name": f"alpha={alpha}",
            "reward_rec": rew_rec,
            "average_rew_rec": avg_ret_rec,
            "tot_reg_rec": tot_reg_rec,
            "opt_action_rec": opt_act_rec
        })

    save_results_plots(results_list, plot_title="Gradient Bandits (without Baseline) Experiment Results On Various Alpha Values",
                       results_folder='results', pdf_name='gradient_bandit_various_alpha_values_without_baseline.pdf')
