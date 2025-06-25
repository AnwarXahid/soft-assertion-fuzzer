import torch
import torch.nn as nn

"""
Source: https://github.com/sunzeyeah/RLHF/blob/cd1a6d54971eb0513f38974aa6dcca53aa2f3174/src/models/loss.py
"""

class PairWiseLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    """

    def forward(self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(chosen_reward - reject_reward)

        ## Inserted code
        return probs
        ## Inserted code

        log_probs = torch.log(probs)
        loss = -log_probs.mean()
        return loss
