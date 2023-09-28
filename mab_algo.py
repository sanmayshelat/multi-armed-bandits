from enum import Enum
from typing import List, Union, Optional

import numpy as np
from pydantic import BaseModel, ValidationError, validator
from scipy.stats._distn_infrastructure import rv_continuous_frozen
from scipy.stats import beta, norm

from bandit_dgp import BanditDGPBernoulli, BanditDGPGaussian

class GreedyMethods(Enum):
    greedy='greedy'
    optimistic_init='optimistic_init'
    ucb1='ucb1'
    thompson_sampling='thompson_sampling'

class GreedyMethodsInput(BaseModel):
    method: GreedyMethods = GreedyMethods.greedy

class MAB():

    def __init__(
        self,
        dgp: Union[BanditDGPBernoulli, BanditDGPGaussian],
        pulls: int,
        seed: int=42,
        known_std: Optional[float]=None
        ):

        self.dgp = dgp
        self.outcomes = dgp.pull_arm(pulls)
        self.rng = np.random.default_rng(seed)
        self.n_arms = self.dgp.n_arms
        self.pulls = pulls

        self.perf_avg = np.empty(self.n_arms) * np.nan
        self.num_samples = np.zeros(self.n_arms, dtype=int)
        self.last_sample = np.zeros(self.n_arms) * np.nan

        self.known_std = known_std
    
    def pull_dgp_arm(self, arm_to_pull):
        arm_outcome_index = self.num_samples[arm_to_pull]
        return self.outcomes[arm_to_pull, arm_outcome_index]

    def pull_all_arms_first(self, method):
        self.reward = []
        for i in range(self.n_arms):
            self.last_sample[i] = self.pull_dgp_arm(i)
            
            if method=='optimistic_init':
                self.last_sample[i] += 0.99
            
            self.num_samples[i] = 1
            self.perf_avg[i] = self.last_sample[i]
            self.reward += [self.outcomes[i,0]]
            
            if method=='thompson_sampling':
                self.priors[i] = self.update_prior(
                        prior=self.priors[i], 
                        x_n=self.last_sample[i]
                    )
                print(i, self.last_sample[i], [k.args[0] for k in self.priors])
            
        
    def update_perf_avg(self, x_bar_n_1, x_n, n):
        return (x_bar_n_1 + (x_n - x_bar_n_1)/n)
    
    def update_prior(self, prior, x_n):
        if prior.dist.name=='beta':
            updated_prior_args = tuple(np.array(prior.args) + np.array([x_n, 1-x_n]))
            updated_prior = beta(updated_prior_args[0], updated_prior_args[1])
        elif prior.dist.name=='norm':
            if not self.known_std:
                raise ValueError("The known std must be provided for this distribution.")
            prior_mean = prior.args[0]
            prior_var = prior.args[1]**2
            posterior_var = 1/(1/prior_var + 1/self.known_std)
            posterior_mean = posterior_var * (prior_mean/prior_var + x_n/self.known_std)
            updated_prior_args = tuple([posterior_mean, np.sqrt(posterior_var)])
            updated_prior = norm(updated_prior_args[0], updated_prior_args[1])
        else:
            raise ValueError("The posterior of this distribution is not available.")
        return updated_prior

    def validate_greedy_methods(self, method):
        GreedyMethodsInput(method=method)
        if isinstance(self.dgp, BanditDGPGaussian) and method=='optimistic_init':
            raise NotImplementedError('Using optimistic_init for Gaussian rewards is not available yet')    

    def validate_priors(self, method, priors):
        if method=='thompson_sampling':
            if not priors:
                raise ValueError('Priors have to be supplied for thompson_sampling')
            elif isinstance(priors, np.ndarray):
                if all(isinstance(i, rv_continuous_frozen) for i in priors):
                    self.priors = priors
            elif isinstance(priors, rv_continuous_frozen):
                self.priors = np.repeat(priors, self.n_arms)
            else:
                raise ValueError('Priors have to be scipy.stats distributions')  

    def greedy(self, method: str='greedy', priors: np.ndarray=None):
        # validate inputs
        self.validate_greedy_methods(method=method)
        self.validate_priors(method=method, priors=priors)
        
        # pull all arms first
        self.pull_all_arms_first(method=method)

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
                self.perf_sample = np.array([i.rvs() for i in self.priors])
                best_perf_arm = self.rng.choice(
                    np.where(self.perf_sample==self.perf_sample.max())[0]
                )
                # print(best_perf_arm, self.perf_sample)

            self.last_sample[best_perf_arm] = self.pull_dgp_arm(best_perf_arm)
            self.reward += [self.last_sample[best_perf_arm]]
            self.num_samples[best_perf_arm] += 1

            new_perf_avg = self.update_perf_avg(
                x_bar_n_1=self.perf_avg[best_perf_arm],
                x_n=self.last_sample[best_perf_arm],
                n=self.num_samples[best_perf_arm]
            )
            self.perf_avg[best_perf_arm] = new_perf_avg

            if method=='thompson_sampling':
                self.priors[best_perf_arm] = self.update_prior(
                    prior=self.priors[best_perf_arm], 
                    x_n=self.last_sample[best_perf_arm]
                )
                print(best_perf_arm, self.perf_sample,self.last_sample[best_perf_arm], [i.args[0] for i in self.priors])
            

    def epsilon_greedy(self, epsilon: int=0.1, decay_rate: int=None):
        self.pull_all_arms_first()

        p_pull = np.zeros(self.n_arms)
        while sum(self.num_samples) < self.pulls:
            best_perf_arm_arr = self.perf_avg==self.perf_avg.max()
            n_best_perf_arms = sum(best_perf_arm_arr)
            p_pull[best_perf_arm_arr] = (1-epsilon)/n_best_perf_arms
            p_pull[~best_perf_arm_arr] = epsilon/(self.n_arms - n_best_perf_arms)
            arm_to_pull = self.rng.choice(a=range(self.n_arms),size=1,p=p_pull)[0]

            self.last_sample[arm_to_pull] = self.pull_dgp_arm(arm_to_pull)
            self.reward += [self.last_sample[arm_to_pull]]
            self.num_samples[arm_to_pull] += 1
            new_perf_avg = self.update_perf_avg(
                x_bar_n_1=self.perf_avg[arm_to_pull],
                x_n=self.last_sample[arm_to_pull],
                n=self.num_samples[arm_to_pull]
            )
            
            self.perf_avg[arm_to_pull] = new_perf_avg


# bandits = BanditDGPBernoulli(prob_success=[0.2,0.5,0.75])
bandits = BanditDGPGaussian([5,10,20], 1)
mab = MAB(dgp=bandits, pulls=2000, known_std=1)
# mab.epsilon_greedy(epsilon=0.999999)
mab.greedy(method='thompson_sampling', priors=norm(10,1))
# mab.greedy()
print(mab.perf_avg, sum(mab.reward))
print([i.args for i in mab.priors])


