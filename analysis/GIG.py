import torch
from torch import nn
import math
import numpy as np

# Modified by Mohamed Ali Al-Badri from:
# https://github.com/PAIR-code/saliency/blob/master/saliency/core/guided_ig.py
# as a PyTorch implementation of Guided Integrated Gradients instead of numpy,
# with modifications for memory efficiency.

class GuidedIG_torch:
    r"""Implements Guided Integrated Gradients method.
    
    Guided IG is an extension of Integrated Gradients, which computes the 
    attributions along a straight-line path between a baseline input and 
    the input of interest. Guided IG incorporates a guided backpropagation 
    approach to limit the attribution to meaningful and positive influences. 

    For more information, see the original paper:
    https://arxiv.org/abs/2106.09788
    """
    def __init__(self, model, device='cuda', target=3, EPSILON=1e-9):
        self.model = model
        self.model.eval() # set the model to evaluation mode
        self.device = torch.device(device)
        self.target = target
        self.EPSILON = EPSILON

    def grad_func(self, x):
        """Returns gradients of the target output with respect to the input."""
        x = x.clone().float().to(self.device)
        x.requires_grad_(True)

        output = self.model(x)[0][self.target] # Forward pass through the model
        
        grads = torch.autograd.grad(output, x, create_graph=True, allow_unused=True)[0]
        del x
        return grads

    def l1_distance(self, x1, x2):
        """Returns L1 distance between two points."""
        return torch.abs(x1 - x2).sum()

    def translate_x_to_alpha(self, x, x_input, x_baseline):
        """Translates a point on the straight-line path to its corresponding alpha value."""
        with torch.no_grad():
            x_input = x_input.to(self.device).double()
            x_baseline = x_baseline.to(self.device).double()

            return torch.where(x_input - x_baseline != 0,
                               (x - x_baseline) / (x_input - x_baseline),
                               torch.tensor(float('nan'), dtype=torch.float64, device=self.device))

    def translate_alpha_to_x(self, alpha, x_input, x_baseline):
        """Translates alpha to the point coordinates within the straight-line interval."""
        assert 0 <= alpha <= 1.0
        return x_baseline + (x_input - x_baseline) * alpha

    def guided_ig_impl(self, x_input, x_baseline, steps=200, fraction=0.25, max_dist=0.02):
        """Calculates and returns Guided IG attribution.

        Args:
            x_input: model input that should be explained.
            x_baseline: chosen baseline for the input explanation.
            grad_func: gradient function that accepts a model input and returns
                the corresponding output gradients. In case of many class model, it is
                responsibility of the implementer of the function to return gradients
                for the specific class of interest.
                steps: the number of Riemann sum steps for path integral approximation.
                fraction: the fraction of features [0, 1] that should be selected and
                changed at every approximation step. E.g., value `0.25` means that 25% of
                the input features with smallest gradients are selected and changed at
                every step.
            max_dist: the relative maximum L1 distance [0, 1] that any feature can
                deviate from the straight line path. Value `0` allows no deviation and,
                therefore, corresponds to the Integrated Gradients method that is
                calculated on the straight-line path. Value `1` corresponds to the
                unbounded Guided IG method, where the path can go through any point within
                the baseline-input hyper-rectangular.
            """



        x_input = torch.tensor(x_input, dtype=torch.float64, device=self.device)
        x_baseline = torch.tensor(x_baseline, dtype=torch.float64, device=self.device)
        x = x_baseline.clone()
        l1_total = self.l1_distance(x_input, x_baseline)
        attr = torch.zeros_like(x_input, dtype=torch.float64, device=self.device)

        # If the input is equal to the baseline, then the attribution is zero.
        total_diff = x_input - x_baseline
        if torch.abs(total_diff).sum() == 0:
            return attr

        # Iterate through every step.
        for step in range(steps):
            # Calculate gradients and make a copy.
            x.requires_grad_(True)
            grad_actual = self.grad_func(x)
            grad = grad_actual.clone().detach()
            x.requires_grad_(False)

            # Calculate current step alpha and the ranges of allowed values for this step.
            alpha = (step + 1.0) / steps
            alpha_min = max(alpha - max_dist, 0.0)
            alpha_max = min(alpha + max_dist, 1.0)
            x_min = self.translate_alpha_to_x(alpha_min, x_input, x_baseline)
            x_max = self.translate_alpha_to_x(alpha_max, x_input, x_baseline)

            # The goal of every step is to reduce L1 distance to the input.
            # `l1_target` is the desired L1 distance after completion of this step.
            l1_target = l1_total * (1 - (step + 1) / steps)

            # Iterate until the desired L1 distance has been reached.
            gamma = float('inf')
            while gamma > 1.0:
                x_old = x.clone()
                x_alpha = self.translate_x_to_alpha(x, x_input, x_baseline)
                x_alpha[torch.isnan(x_alpha)] = alpha_max

                # All features that fell behind the [alpha_min, alpha_max] interval in terms of alpha
                # should be assigned the x_min values.
                x[x_alpha < alpha_min] = x_min[x_alpha < alpha_min]

                # Calculate the current L1 distance from the input.
                l1_current = self.l1_distance(x, x_input)

                # If the current L1 distance is close enough to the desired one,
                # update the attribution and proceed to the next step.
                if math.isclose(l1_target, l1_current, rel_tol=self.EPSILON, abs_tol=self.EPSILON):
                    attr += (x - x_old) * grad_actual
                    break

                # Features that reached `x_max` should not be included in the selection.
                # Assign very high gradients to them so they are excluded.
                grad[x == x_max] = float('inf')

                # Select features with the lowest absolute gradient.
                threshold = torch.quantile(torch.abs(grad), fraction, dim=0, keepdim=True)
                s = torch.logical_and(torch.abs(grad) <= threshold, grad != float('inf'))

                # Find by how much the L1 distance can be reduced by changing only the selected features.
                l1_s = (torch.abs(x - x_max) * s).sum()

                # Calculate the ratio `gamma` that shows how much the selected features should
                # be changed toward `x_max` to close the gap between the current L1 and target L1.
                if l1_s > 0:
                    gamma = (l1_current - l1_target) / l1_s
                else:
                    gamma = float('inf')

                if gamma > 1.0:
                    # Gamma higher than 1.0 means that changing selected features is not
                    # enough to close the gap. Therefore, change them as much as possible to
                    # stay in the valid range.
                    x[s] = x_max[s]
                else:
                    assert gamma > 0, gamma
                    x[s] = self.translate_alpha_to_x(gamma, x_max, x)[s]

                # Update attribution to reflect changes in `x`.
                attr += (x - x_old) * grad_actual
                
            # at the end of the loop, clear the gradients because they're using up memory
            grad_actual = grad_actual.detach()
            grad = grad.detach()
            del grad_actual, grad
        
        # move all the values to CPU; GPU overloaded
        attr = attr.detach().cpu()
        x_input = x_input.detach().cpu()
        x_baseline = x_baseline.detach().cpu()
        torch.cuda.empty_cache()
        
        return attr
