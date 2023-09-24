import numpy as np
from bandit_dgp import BanditDGP
from enum import Enum
from pydantic import BaseModel
from scipy.stats import beta
from typing import List, Union

class GreedyMethodsEnum(Enum):
    greedy='greedy'
    optimistic_init='optimistic_init'
    ucb1='ucb1'
    thompson_sampling='thompson_sampling'

class GreedyMethodsInput(BaseModel):
    method: GreedyMethodsEnum = GreedyMethodsEnum.greedy


class MAB():

    def __init__(
        self,
        dgp: BanditDGP,
        pulls: int,
        seed: int=42
        ):

        self.dgp = dgp
        self.outcomes = dgp.pull_arm(pulls)
        self.rng = np.random.default_rng(seed)
        self.n_arms = self.dgp.n_arms
        self.pulls = pulls

        self.perf_avg = np.empty(self.n_arms) * np.nan
        self.num_samples = np.zeros(self.n_arms, dtype=int)
        self.last_sample = np.zeros(self.n_arms) * np.nan
    
    def pull_all_arms_first(self, optimistic_init: bool=False):
        self.reward = []
        for i in range(self.n_arms):
            self.last_sample[i] = self.outcomes[i,0] * (not optimistic_init) + 0.99 * (optimistic_init)
            self.num_samples[i] = 1
            self.perf_avg[i] = self.last_sample[i]
            self.reward += [self.outcomes[i,0]]
        
    def update_perf_avg(self, x_bar_n_1, x_n, n):
        print(x_n, x_bar_n_1, n)
        return (x_bar_n_1 + (x_n - x_bar_n_1)/n)

    def greedy(self, method: str='greedy', priors: np.array=None):
        # validate method
        GreedyMethodsInput(method=method)
        if method=='thompson_sampling':
            if not priors:
                self.priors=np.ones((self.n_arms,2))
            else:
                self.priors=priors
        
        # pull all arms first
        self.pull_all_arms_first(optimistic_init= method=='optimistic_init')

        while sum(self.num_samples) < self.pulls:
            if method in ['greedy', 'optimistic_init']:
                best_perf_arm = self.rng.choice(
                    np.where(self.perf_avg==self.perf_avg.max())[0]
                )            
                print(best_perf_arm, self.perf_avg)
            elif method=='ucb1':
                self.perf_ucb1 = self.perf_avg + np.sqrt(2*np.log(self.pulls)/self.num_samples)
                best_perf_arm = self.rng.choice(
                    np.where(self.perf_ucb1==self.perf_ucb1.max())[0]
                )            
                print(best_perf_arm, self.perf_ucb1)
            elif method=='thompson_sampling':
                self.perf_sample = self.rng.beta(self.priors[:,0], self.priors[:,1])
                best_perf_arm = self.rng.choice(
                    np.where(self.perf_sample==self.perf_sample.max())[0]
                )

            self.last_sample[best_perf_arm] = self.outcomes[
                best_perf_arm,
                self.num_samples[best_perf_arm]
            ]
            self.reward += [self.last_sample[best_perf_arm]]
            self.num_samples[best_perf_arm] += 1
            new_perf_avg = self.update_perf_avg(
                x_bar_n_1=self.perf_avg[best_perf_arm],
                x_n=self.last_sample[best_perf_arm],
                n=self.num_samples[best_perf_arm]
            )
            self.perf_avg[best_perf_arm] = new_perf_avg
            if method=='thompson_sampling':
                self.priors[best_perf_arm,0] += self.last_sample[best_perf_arm]
                self.priors[best_perf_arm,1] += 1-self.last_sample[best_perf_arm]

    def epsilon_greedy(self, epsilon: int=0.1, decay_rate: int=None):
        self.pull_all_arms_first()

        p_pull = np.zeros(self.n_arms)
        while sum(self.num_samples) < self.pulls:
            best_perf_arm_arr = self.perf_avg==self.perf_avg.max()
            n_best_perf_arms = sum(best_perf_arm_arr)
            p_pull[best_perf_arm_arr] = (1-epsilon)/n_best_perf_arms
            p_pull[~best_perf_arm_arr] = epsilon/(self.n_arms - n_best_perf_arms)
            arm_to_pull = self.rng.choice(a=range(self.n_arms),size=1,p=p_pull)[0]

            self.last_sample[arm_to_pull] = self.outcomes[
                arm_to_pull,
                self.num_samples[arm_to_pull]
            ]
            self.reward += [self.last_sample[arm_to_pull]]
            self.num_samples[arm_to_pull] += 1
            new_perf_avg = self.update_perf_avg(
                x_bar_n_1=self.perf_avg[arm_to_pull],
                x_n=self.last_sample[arm_to_pull],
                n=self.num_samples[arm_to_pull]
            )
            
            self.perf_avg[arm_to_pull] = new_perf_avg


bandits = BanditDGP(prob_success=[0.2,0.5,0.75])
mab = MAB(dgp=bandits, pulls=1000)
# mab.epsilon_greedy(epsilon=0.999999)
mab.greedy(method='thompson_sampling')
print(mab.perf_avg, sum(mab.reward))


