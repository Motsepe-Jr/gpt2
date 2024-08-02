import math
from dataclasses import dataclass
from typing import List

@dataclass
class CosineScheduler:
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 1000
    max_iters: int = 100_000

    def __call__(self, iteration: int) -> float:
        """
        Calculate the learning rate based on cosine annealing with linear warmup.
        
        Args:
            iteration (int): Current iteration number.
        
        Returns:
            float: Calculated learning rate.
        """
        if iteration < self.warmup_iters:
            return self.max_lr * iteration / self.warmup_iters

        if iteration > self.max_iters:
            return self.min_lr

        decay_ratio = (iteration - self.warmup_iters) / (self.max_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1

        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

@dataclass
class ExponentialScheduler:
    max_lr: float = 3e-4
    gamma: float = 0.1

    def __call__(self, iteration: int) -> float:
        """
        Calculate the learning rate based on exponential decay.
        
        Args:
            iteration (int): Current iteration number.
        
        Returns:
            float: Calculated learning rate.
        """
        return self.max_lr * (self.gamma ** iteration)
    
@dataclass
class ConstantLR:
    max_lr: float = 3e-4

    def __call__(self, iteration: int) -> float:
        """
        Keep learning rate constant across entire training steps
        
        Args:
        
        Returns:
            float: Calculated learning rate.
        """
        return self.max_lr 

@dataclass
class StepLR:
    max_lr: float = 3e-4
    max_iters: int = 100_000
    gamma: float = 0.1

    def __call__(self, iteration: int) -> float:
        """
        Calculate the learning rate based on step decay.
        
        Args:
            iteration (int): Current iteration number.
        
        Returns:
            float: Calculated learning rate.
        """
        return self.max_lr * (self.gamma ** (iteration // self.max_iters))

@dataclass
class MultiStepLR:
    max_lr: float = 3e-4
    max_iters: int = 100_000
    gamma: float = 0.1
    milestones: List[int] = (10, 30)

    def __call__(self, iteration: int) -> float:
        """
        Calculate the learning rate based on multi-step decay.
        
        Args:
            iteration (int): Current iteration number.
        
        Returns:
            float: Calculated learning rate.
        """
        return self.max_lr * (self.gamma ** sum([iteration >= m for m in self.milestones]))

@dataclass
class LinearLR:
    max_lr: float = 3e-4
    max_iters: int = 100_000

    def __call__(self, iteration: int) -> float:
        """
        Calculate the learning rate based on linear decay.
        
        Args:
            iteration (int): Current iteration number.
        
        Returns:
            float: Calculated learning rate.
        """
        return self.max_lr * (1 - iteration / self.max_iters)

@dataclass
class PolynomialLR:
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    max_iters: int = 100_000
    power: float = 1.0

    def __call__(self, iteration: int) -> float:
        """
        Calculate the learning rate based on polynomial decay.
        
        Args:
            iteration (int): Current iteration number.
        
        Returns:
            float: Calculated learning rate.
        """
        return (self.max_lr - self.min_lr) * (1 - iteration / self.max_iters) ** self.power + self.min_lr

@dataclass
class CyclicLR:
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    max_iters: int = 100_000

    def __call__(self, iteration: int) -> float:
        """
        Calculate the learning rate based on cyclic learning rate policy.
        
        Args:
            iteration (int): Current iteration number.
        
        Returns:
            float: Calculated learning rate.
        """
        cycle = math.floor(1 + iteration / (2 * self.max_iters))
        x = abs(iteration / self.max_iters - 2 * cycle + 1)
        return self.min_lr + (self.max_lr - self.min_lr) * max(0, (1 - x))

@dataclass
class OneCycleLR:
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    max_iters: int = 100_000
    pct_start: float = 0.3

    def __call__(self, iteration: int) -> float:
        """
        Calculate the learning rate based on one cycle policy.
        
        Args:
            iteration (int): Current iteration number.
        
        Returns:
            float: Calculated learning rate.
        """
        if iteration / self.max_iters <= self.pct_start:
            return self.min_lr + (self.max_lr - self.min_lr) * (iteration / (self.pct_start * self.max_iters))
        else:
            return self.max_lr - (self.max_lr - self.min_lr) * ((iteration - self.pct_start * self.max_iters) / ((1 - self.pct_start) * self.max_iters))

@dataclass
class CosineAnnealingWarmRestarts:
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 10
    T_mult: int = 1

    def __call__(self, iteration: int) -> float:
        """
        Calculate the learning rate based on cosine annealing with warm restarts.
        
        Args:
            iteration (int): Current iteration number.
        
        Returns:
            float: Calculated learning rate.
        """
        T_cur = iteration % self.warmup_steps
        T_i = self.warmup_steps
        while iteration >= T_i:
            iteration -= T_i
            T_i *= self.T_mult
        return self.min_lr + (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * T_cur / self.warmup_steps)) / 2