"""Utilities to find uncertainty in classification tasks"""
from torch import Tensor
import torch


def score_avg(props: Tensor):
    return props.mean(axis=0)

def aleatoric_uncertainty(props: Tensor):
    aleatoric = props * (1-props)
    aleatoric = aleatoric.mean(axis=0)
    return aleatoric

def epistemic_uncertainty(props: Tensor):
    return torch.var(props, dim=0, unbiased=False)

def uncertainty_avg(props: Tensor):
    return aleatoric_uncertainty(props) + epistemic_uncertainty(props)
