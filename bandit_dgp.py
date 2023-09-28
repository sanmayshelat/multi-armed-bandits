from typing import List, Optional, Union

import numpy as np


class BanditDGPBernoulli():
    def __init__(
            self,
            prob_success:Union[List, np.ndarray],
            seed: Optional[int]=None,
            ):
        self.prob_success = prob_success
        self.seed = seed
        self.n_arms = len(prob_success)
        self.rng = np.random.default_rng(seed=self.seed)
        
    def pull_arm(self, draws: Union[int, List, np.ndarray]):
        if isinstance(draws, (int, float)):
            draws = np.repeat(draws, self.n_arms)
        return np.array([self.rng.binomial(n=1, p=p, size=d) for p, d in zip(self.prob_success, draws)])

class BanditDGPGaussian():
    def __init__(
            self,
            mean:Union[List, np.ndarray],
            std:Union[int, float, List, np.ndarray],
            seed: Optional[int]=None,
            ):
        self.mean = mean
        self.n_arms = len(mean)
        if isinstance(std, (int, float)):
            self.std = np.repeat(std, self.n_arms)
        else:
            self.std = std
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)
        
    def pull_arm(self, draws: Union[int, List, np.ndarray]):
        if isinstance(draws, (int, float)):
            draws = np.repeat(draws, self.n_arms)
        return np.array([self.rng.normal(loc=m, scale=s, size=d) for m, s, d in zip(self.mean, self.std, draws)])
