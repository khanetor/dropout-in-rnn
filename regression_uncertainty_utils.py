"""Utilities to find uncertainty in regression tasks"""
from torch import Tensor
import torch


def aleatoric_uncertainty(log_var: Tensor) -> Tensor:
    return torch.exp(log_var).mean(axis=0)

def epistemic_uncertainty(mean: Tensor) -> Tensor:
    return torch.var(mean, axis=0)

def uncertainty(mean: Tensor, log_var: Tensor) -> Tensor:
    return epistemic_uncertainty(mean) + aleatoric_uncertainty(log_var)

def prediction_avg(mean: Tensor) -> Tensor:
    return torch.mean(mean, axis=0)