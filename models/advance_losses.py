import torch
from torch.nn import functional as F
def combine_policy_logits_to_log_probs(
        behavior_policy_logits: torch.Tensor,
        actions: torch.Tensor,
        actions_taken_mask: torch.Tensor
) -> torch.Tensor:
    """
    Combines all policy_logits at a given step to get a single action_log_probs value for that step
    Initial shape: time, batch, 1, players, x, y, n_actions
    Returned shape: time, batch, players
    """
    # Get the action probabilities
    probs = F.softmax(behavior_policy_logits, dim=-1)
    # Ignore probabilities for actions that were not used
    probs = actions_taken_mask * probs
    # Select the probabilities for actions that were taken by stacked agents and sum these
    selected_probs = torch.gather(probs, -1, actions)
    # Convert the probs to conditional probs, since we sample without replacement
    remaining_probability_density = 1. - torch.cat([
        torch.zeros(
            (*selected_probs.shape[:-1], 1),
            device=selected_probs.device,
            dtype=selected_probs.dtype
        ),
        selected_probs[..., :-1].cumsum(dim=-1)
    ], dim=-1)
    # Avoid division by zero
    remaining_probability_density = remaining_probability_density + torch.where(
        remaining_probability_density == 0,
        torch.ones_like(remaining_probability_density),
        torch.zeros_like(remaining_probability_density)
    )
    conditional_selected_probs = selected_probs / remaining_probability_density
    # Remove 0-valued conditional_selected_probs in order to eliminate neg-inf valued log_probs
    conditional_selected_probs = conditional_selected_probs + torch.where(
        conditional_selected_probs == 0,
        torch.ones_like(conditional_selected_probs),
        torch.zeros_like(conditional_selected_probs)
    )
    log_probs = torch.log(conditional_selected_probs)
    # Sum over actions, y and x dimensions to combine log_probs from different actions
    # Squeeze out action_planes dimension as well
    return torch.flatten(log_probs, start_dim=-3, end_dim=-1).sum(dim=-1).squeeze(dim=-2)


def combine_policy_entropy(
        policy_logits: torch.Tensor,
        actions_taken_mask: torch.Tensor
) -> torch.Tensor:
    """
    Computes and combines policy entropy for a given step.
    NB: We are just computing the sum of individual entropies, not the joint entropy, because I don't think there is
    an efficient way to compute the joint entropy?
    Initial shape: time, batch, action_planes, players, x, y, n_actions
    Returned shape: time, batch, players
    """
    policy = F.softmax(policy_logits, dim=-1)
    log_policy = F.log_softmax(policy_logits, dim=-1)
    log_policy_masked_zeroed = torch.where(
        log_policy.isneginf(),
        torch.zeros_like(log_policy),
        log_policy
    )
    entropies = (policy * log_policy_masked_zeroed).sum(dim=-1)
    assert actions_taken_mask.shape == entropies.shape
    entropies_masked = entropies * actions_taken_mask.float()
    # Sum over y, x, and action_planes dimensions to combine entropies from different actions
    return entropies_masked.sum(dim=-1).sum(dim=-1).squeeze(dim=-2)
if __name__=='__main__':
    c=combine_policy_logits_to_log_probs(
       torch.Tensor(torch.rand((6,6,12))),
   torch.zeros((6,6,1),dtype=torch.long),
    torch.zeros((6,6,12))
    )
    v=0