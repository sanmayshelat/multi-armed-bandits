from typing import Union, List, Optional
import numpy as np
from pydantic import BaseModel

# def x


class BanditDGP():
    def __init__(
            self,
            prob_success:Union[List, np.array],
            seed: Optional[int]=None,
            ):
        self.prob_success = prob_success
        self.seed = seed
        self.n_arms = len(prob_success)
        self.rng = np.random.default_rng(seed=self.seed)
        
    def pull_arm(self, draws: Union[int, List, np.array]):
        if isinstance(draws, (int, float)):
            draws = np.repeat(draws, self.n_arms)
        return np.array([self.rng.binomial(n=1, p=p, size=d) for p, d in zip(self.prob_success, draws)])
