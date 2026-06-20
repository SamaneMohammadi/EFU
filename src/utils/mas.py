"""Memory Aware Synapses (MAS) parameter importance for EFU's unlearning loss.

Importance Omega_j is the average L2-norm sensitivity of the model output to each
parameter. EFU regularizes with the *inverted* importance (1 - Omega) so that
unlearning preferentially changes parameters least important for the retained
behavior, masking parameter-level traces of unlearning.
"""

import copy

import torch


def _compute_weight_importance(model, data):
    model.train()
    importance = {n: torch.zeros_like(p, device=model.device)
                  for n, p in model.named_parameters() if p.requires_grad}
    total_samples = 0
    for x, _ in data:
        x = x.to(model.device)
        model.zero_grad()
        logits = model(x)
        logits.pow(2).sum().backward()
        for name, param in model.named_parameters():
            if name in importance and param.grad is not None:
                importance[name] += param.grad.detach().abs().sum()
        total_samples += x.shape[0]
    for n, i in importance.items():
        importance[n] /= max(total_samples, 1)
        lo, hi = i.min(), i.max()
        if hi > lo:
            importance[n] = (i - lo) / (hi - lo + 1e-8)
    return {n: 1 - i for n, i in importance.items()}      # inverted importance


def get_weights_importance(model, data):
    _model = copy.deepcopy(model).to(model.device)
    importance = _compute_weight_importance(_model, data)
    _model.to('cpu')
    del _model
    return importance
